#!/bin/bash

################################################################################
# EBT Sudoku RL (GRPO/GSPO) 训练脚本
# ─────────────────────────────────────────────────────────────────────────────
# 基于 SFT v3 最佳 checkpoint 进行 GRPO 强化学习后训练。
#
# 2026-05-31 稳定性默认值来自 d26-ctx2048-sudoku-rl-gspo-20260529:
#   - 有效信号集中在 step 60-120，step 190 后出现明显退化。
#   - 默认使用短跑 160 steps，并保存 50/100/150 附近候选。
#   - trajectory 固定写到当前 run 根目录，避免落到 repo 根目录 ./trajectories。
#   - GSPO clip 对齐 verl tight sequence-ratio recipe，避免 clip 永远失效。
#   - clue violation 在 reward 层 hard-gate blank/full-solve credit。
#
# 算法: Energy-GSPO / Group Relative Policy Optimization
#   - 每个 prompt 生成 N 个 completion
#   - 多组件奖励 (format + accuracy + validity + full_solve)
#   - energy-ratio clipped policy gradient + KL/energy anchor
#
# 参考: d1/diffu-grpo (adapted for EBT's energy-based paradigm)
################################################################################

set -e

### 基础配置 ###
export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### SFT 权重 (RL 起点) ###
# TODO: 替换为实际的 SFT v3 最佳 checkpoint 路径
SFT_CKPT="${SFT_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-0.6-20260508/sft_train/checkpoints/s=step=2990-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4782.ckpt}"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_SFT_DATA_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"

# Offline 模式
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Sudoku V2 数据缓存目录
export SUDOKU_DATA_DIR_V2="${SUDOKU_DATA_DIR_V2:-/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2}"

# RL 实验 ID。允许 rjob/复现实验通过环境变量固定输出目录；默认行为保持不变。
export EXP_ID="${EXP_ID:-d26-ctx2048-sudoku-rl-gspo-$(date +%Y%m%d)}"
# export EXP_ID="d26-ctx2048-sudoku-rl-reinforce-$(date +%Y%m%d)"

################################################################################
# GRPO 超参数
################################################################################
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"            # more samples per prompt to keep GRPO group variance
# MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-165}"     # 9x9 board ≤162 chars; tail tokens add noise
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-210}"     # 9x9 board ≤162 chars; tail tokens add noise

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-192}"
# 2026-07-03 explore-anchor still hit LOCAL_ZERO=4/10 around step 33:
# KL stayed low, so the bottleneck is rollout diversity rather than anchor
# drift. Use a larger group and moderately wider sampling while slowing updates.
TEMPERATURE="${TEMPERATURE:-0.80}"
TOP_P="${TOP_P:-0.90}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}" # sub-batch for generation (VRAM management)

NUM_ITERATIONS="${NUM_ITERATIONS:-1}"              # legacy loss recomputation; keep 1 when using GSPO_UPDATE_EPOCHS
GSPO_UPDATE_EPOCHS="${GSPO_UPDATE_EPOCHS:-1}"  # keep 1 by default; previous 2-epoch long run degraded after early gains
EPSILON="${EPSILON:-0.2}"                   # PPO clip range (only used by energy_gspo)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-3e-4}"    # verl GSPO sequence-ratio clip
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"
BETA="${BETA:-2.0}"                         # stronger anchor for the higher-exploration rollout
ENERGY_KL_MODE="${ENERGY_KL_MODE:-symmetric_huber}"
ENERGY_KL_HUBER_DELTA="${ENERGY_KL_HUBER_DELTA:-0.5}"
# 2026-07-03 conservative restart: the 20260702-233012 fusion run repeatedly
# oscillated from usable Sudoku reward to format/clue-only batches with high
# pre-clip grad_norm, so shrink both AdamW and Muon update scales by default.
LEARNING_RATE="${LEARNING_RATE:-1e-7}"       # AdamW base/fallback LR; Muon matrices use MUON_LR
# RL_LOSS_TYPE="energy_reinforce"  # removes 1/|y| dilution
RL_LOSS_TYPE="${RL_LOSS_TYPE:-energy_gspo}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.5}"
MAX_GRAD_PER_PARAM="${MAX_GRAD_PER_PARAM:-0.005}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"

# ── Optimizer (v3 P0+P1+muon) ─────────────────────────────────────────────────
# adamw      : v3 P0 multi-group AdamW (transformer/v2e/scalar/other split LR)
# muon_adamw : Muon for transformer matrices + AdamW for the rest (matches SFT)
RL_OPTIMIZER="${RL_OPTIMIZER:-muon_adamw}"  # SFT-validated hybrid optimizer: Muon matrices + AdamW lanes
MUON_LR="${MUON_LR:-2e-6}"                  # Muon peak LR for transformer matrices; scheduler scales it after warmup
ADAMW_VOCAB_TO_EMBED_LR="${ADAMW_VOCAB_TO_EMBED_LR:--1}"  # <=0 => LEARNING_RATE * 0.5
ADAMW_SCALAR_LR="${ADAMW_SCALAR_LR:--1}"                  # <=0 => LEARNING_RATE * 2.0
ADAMW_OTHER_LR="${ADAMW_OTHER_LR:--1}"                    # <=0 => LEARNING_RATE
ADVANTAGE_NORM="${ADVANTAGE_NORM:-group_mean_global_std}"  # batch-wide std prevents tiny intra-group std blow-ups
GLOBAL_STD_MIN="${GLOBAL_STD_MIN:-0.2}"
# Skip rollout batches with no usable reward signal. The trainer returns a
# graph-attached zero loss and clears every grad to None before optimizer.step,
# so Muon/AdamW/weight_decay do not move parameters on skipped steps.
SKIP_DEGENERATE_THRESHOLD="${SKIP_DEGENERATE_THRESHOLD:-0.9}"
MIN_REWARD_STD_TO_UPDATE="${MIN_REWARD_STD_TO_UPDATE:-1e-4}"
MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE="${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE:-0.0}"
# Skip nearly malformed rollout batches that still have nonzero reward std.
# The 20260703 conservative run hit such a batch at step 25
# (reward_mean≈0.004, format≈0.004, ref_energy_kl≈1.65e-2).
MIN_REWARD_MEAN_TO_UPDATE="${MIN_REWARD_MEAN_TO_UPDATE:-0.5}"
MIN_REWARD_FORMAT_TO_UPDATE="${MIN_REWARD_FORMAT_TO_UPDATE:-0.25}"
SKIP_CONSENSUS="${SKIP_CONSENSUS:-local}"    # DDP-safe: bad ranks zero out; all-bad batches skip globally

# MAX_STEPS="${MAX_STEPS:-160}"                # analyzed run peaked at 60-120 and degraded around 150-190
MAX_STEPS="${MAX_STEPS:-1000}"                # analyzed run peaked at 60-120 and degraded around 150-190
           
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SAVE_TOP_K="${SAVE_TOP_K:-2}"
SEED="${SEED:-42}"

TRAJ_LOG_INTERVAL="${TRAJ_LOG_INTERVAL:-25}"
TRAJ_NUM_SAMPLES="${TRAJ_NUM_SAMPLES:-4}"
COLLAPSE_CHECK_WINDOW="${COLLAPSE_CHECK_WINDOW:-2}"

# Current EBT forward debug is opt-in; keep train.log compact by default.
export DBG_RL_FORWARD="${DBG_RL_FORWARD:-0}"
export DBG_RL_FORWARD_INTERVAL="${DBG_RL_FORWARD_INTERVAL:-200}"
export RL_PHASE_LOG="${RL_PHASE_LOG:-1}"       # concise rank-0 phase markers + heartbeat.json
export TRAJ_PRINT_TEXT="${TRAJ_PRINT_TEXT:-1}" # JSONL keeps full text; stdout stays compact by default
export TRAJ_TEXT_CHARS="${TRAJ_TEXT_CHARS:-180}"

################################################################################
# GPU 配置
################################################################################
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

################################################################################
# 数据集检查
################################################################################
echo ""
echo "=================================="
echo "  检查 Sudoku V2 数据集 (RL 复用)"
echo "=================================="
echo "数据集路径: ${SUDOKU_DATA_DIR_V2}"

MISSING=()
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_train.pt" ] && MISSING+=("rrn_train.pt")
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_val.pt" ] && MISSING+=("rrn_val.pt")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "✗ 缺少以下 V2 数据文件:"
    for f in "${MISSING[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "请先运行 V2 数据准备脚本:"
    echo "  python openebm/elm/data/prepare_sudoku_data_v2.py --data_dir ${SUDOKU_DATA_DIR_V2}"
    exit 1
fi

echo "✓ rrn_train.pt"
echo "✓ rrn_val.pt"

# 检查 SFT checkpoint
if [ ! -f "${SFT_CKPT}" ]; then
    echo "✗ SFT Checkpoint 不存在: ${SFT_CKPT}"
    echo "  请设置 SFT_CKPT 环境变量指向 SFT v3 最佳 checkpoint"
    exit 1
fi
echo "✓ SFT Checkpoint: ${SFT_CKPT}"
echo ""

################################################################################
# 实验目录布局
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${SFT_CKPT}"
exp_init_sft "$0"
export RUN_NAME="${EXP_ID}-sudoku-rl-grpo"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

# 轨迹输出目录跟着实验 run 根目录走。TrajectoryLogger 会写入:
#   ${TRAJ_OUTPUT_DIR}/trajectories/step_*/rollout_samples.jsonl
#   ${TRAJ_OUTPUT_DIR}/collapse_dump/step_*.jsonl
# 例如:
#   /mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-gspo-20260529/trajectories
TRAJ_OUTPUT_DIR="${TRAJ_OUTPUT_DIR:-${EXP_DIR}}"
export TRAJ_OUTPUT_DIR

exp_save_hparams "${EXP_SFT_DIR}" \
    "model_size=${MODEL_SIZE}" \
    "algorithm=grpo" \
    "sft_checkpoint=${SFT_CKPT}" \
    "num_generations=${NUM_GENERATIONS}" \
    "max_completion_length=${MAX_COMPLETION_LENGTH}" \
    "temperature=${TEMPERATURE}" \
    "epsilon=${EPSILON}" \
    "clip_ratio_low=${CLIP_RATIO_LOW}" \
    "clip_ratio_high=${CLIP_RATIO_HIGH}" \
    "gspo_update_epochs=${GSPO_UPDATE_EPOCHS}" \
    "beta=${BETA}" \
    "energy_kl_mode=${ENERGY_KL_MODE}" \
    "energy_kl_huber_delta=${ENERGY_KL_HUBER_DELTA}" \
    "learning_rate=${LEARNING_RATE}" \
    "rl_optimizer=${RL_OPTIMIZER}" \
    "muon_lr=${MUON_LR}" \
    "adamw_vocab_to_embed_lr=${ADAMW_VOCAB_TO_EMBED_LR}" \
    "adamw_scalar_lr=${ADAMW_SCALAR_LR}" \
    "adamw_other_lr=${ADAMW_OTHER_LR}" \
    "warmup_steps=${WARMUP_STEPS}" \
    "gradient_clip_val=${GRADIENT_CLIP_VAL}" \
    "max_grad_per_param=${MAX_GRAD_PER_PARAM}" \
    "advantage_norm=${ADVANTAGE_NORM}" \
    "global_std_min=${GLOBAL_STD_MIN}" \
    "skip_degenerate_threshold=${SKIP_DEGENERATE_THRESHOLD}" \
    "min_reward_std_to_update=${MIN_REWARD_STD_TO_UPDATE}" \
    "min_unique_completion_ratio_to_update=${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE}" \
    "min_reward_mean_to_update=${MIN_REWARD_MEAN_TO_UPDATE}" \
    "min_reward_format_to_update=${MIN_REWARD_FORMAT_TO_UPDATE}" \
    "skip_consensus=${SKIP_CONSENSUS}" \
    "rl_loss_type=${RL_LOSS_TYPE}" \
    "top_p=${TOP_P}" \
    "generation_batch_size=${GENERATION_BATCH_SIZE}" \
    "traj_output_dir=${TRAJ_OUTPUT_DIR}" \
    "traj_log_interval=${TRAJ_LOG_INTERVAL}" \
    "traj_num_samples=${TRAJ_NUM_SAMPLES}" \
    "rl_phase_log=${RL_PHASE_LOG}" \
    "traj_print_text=${TRAJ_PRINT_TEXT}" \
    "collapse_check_window=${COLLAPSE_CHECK_WINDOW}" \
    "max_steps=${MAX_STEPS}" \
    "num_gpus=${NUM_GPUS}" \
    "save_top_k=${SAVE_TOP_K}"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  EBT Sudoku RL (GRPO) 训练"
echo "=================================="
echo "SFT checkpoint:          ${SFT_CKPT}"
echo "Algorithm:               GRPO (${RL_LOSS_TYPE})"
echo "GSPO update epochs:      ${GSPO_UPDATE_EPOCHS}"
echo "Num generations:         ${NUM_GENERATIONS}"
echo "Max completion length:   ${MAX_COMPLETION_LENGTH}"
echo "Temperature:             ${TEMPERATURE}"
echo "Top-p:                   ${TOP_P}"
echo "Epsilon (clip):          ${EPSILON}"
echo "Clip low/high:           ${CLIP_RATIO_LOW}/${CLIP_RATIO_HIGH}"
echo "Beta (KL anchor):        ${BETA}"
echo "Energy KL mode:          ${ENERGY_KL_MODE} (delta=${ENERGY_KL_HUBER_DELTA})"
echo "Learning rate:           ${LEARNING_RATE}"
echo "Warmup steps:            ${WARMUP_STEPS}"
echo "Gradient clip (global):  ${GRADIENT_CLIP_VAL}"
echo "Max grad per param:      ${MAX_GRAD_PER_PARAM}"
echo "Advantage norm:          ${ADVANTAGE_NORM}"
echo "Skip degen threshold:    ${SKIP_DEGENERATE_THRESHOLD}"
echo "Min reward std update:   ${MIN_REWARD_STD_TO_UPDATE}"
echo "Min unique ratio update: ${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE}"
echo "Skip consensus:          ${SKIP_CONSENSUS}"
echo "Max steps:               ${MAX_STEPS}"
echo "Val interval:            ${VAL_CHECK_INTERVAL}"
echo "Trajectory dir:          ${TRAJ_OUTPUT_DIR}/trajectories"
echo "Heartbeat file:          ${TRAJ_OUTPUT_DIR}/logs/heartbeat.json"
echo "Forward debug:           DBG_RL_FORWARD=${DBG_RL_FORWARD} interval=${DBG_RL_FORWARD_INTERVAL}"
echo "Phase log:               RL_PHASE_LOG=${RL_PHASE_LOG}"
echo "GPUs:                    ${NUM_GPUS}"
echo "Log file:                ${LOG_FILE}"
echo ""
if [ "${SKIP_CONFIRM:-0}" != "1" ]; then
    read -p "按 Enter 开始训练，或 Ctrl+C 取消..."
fi

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT Sudoku RL (GRPO) 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# SFT From:        ${SFT_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
# Traj Dir:        ${TRAJ_OUTPUT_DIR}/trajectories
#
# GRPO Config:
#   num_generations=${NUM_GENERATIONS}
#   max_completion_length=${MAX_COMPLETION_LENGTH}
#   temperature=${TEMPERATURE}
#   top_p=${TOP_P}
#   epsilon=${EPSILON}
#   clip_ratio_low=${CLIP_RATIO_LOW}
#   clip_ratio_high=${CLIP_RATIO_HIGH}
#   beta=${BETA}
#   energy_kl_mode=${ENERGY_KL_MODE}
#   optimizer=${RL_OPTIMIZER}
#   adamw_base_lr=${LEARNING_RATE}
#   muon_lr=${MUON_LR}
#   adamw_vocab_to_embed_lr=${ADAMW_VOCAB_TO_EMBED_LR}
#   adamw_scalar_lr=${ADAMW_SCALAR_LR}
#   adamw_other_lr=${ADAMW_OTHER_LR}
#   max_steps=${MAX_STEPS}
#   traj_dir=${TRAJ_OUTPUT_DIR}/trajectories
#   skip_consensus=${SKIP_CONSENSUS}
#   min_reward_mean_to_update=${MIN_REWARD_MEAN_TO_UPDATE}
#   min_reward_format_to_update=${MIN_REWARD_FORMAT_TO_UPDATE}
#   heartbeat=${TRAJ_OUTPUT_DIR}/logs/heartbeat.json
################################################################################

LOG_HEADER

{
echo "================================================================================"
echo "[SYSTEM INFO]"
echo "================================================================================"
echo "Hostname:         $(hostname)"
echo "User:             $(whoami)"
echo "Python:           $(python3 --version 2>&1)"
echo "PyTorch:          $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA Available:   $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU Count:        $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""
echo "================================================================================"
echo "[开始 RL 训练]"
echo "================================================================================"
echo ""

set +e

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m openebm.elm.rl.train_rl_sudoku \
    --sft_checkpoint "${SFT_CKPT}" \
    --num_generations ${NUM_GENERATIONS} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --generation_batch_size ${GENERATION_BATCH_SIZE} \
    --num_iterations ${NUM_ITERATIONS} \
    --gspo_update_epochs ${GSPO_UPDATE_EPOCHS} \
    --epsilon ${EPSILON} \
    --clip_ratio_low ${CLIP_RATIO_LOW} \
    --clip_ratio_high ${CLIP_RATIO_HIGH} \
    --beta ${BETA} \
    --energy_kl_mode ${ENERGY_KL_MODE} \
    --energy_kl_huber_delta ${ENERGY_KL_HUBER_DELTA} \
    --learning_rate ${LEARNING_RATE} \
    --rl_loss_type ${RL_LOSS_TYPE} \
    --weight_decay ${WEIGHT_DECAY} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --max_grad_per_param ${MAX_GRAD_PER_PARAM} \
    --warmup_steps ${WARMUP_STEPS} \
    --rl_optimizer ${RL_OPTIMIZER} \
    --muon_lr ${MUON_LR} \
    --adamw_vocab_to_embed_lr ${ADAMW_VOCAB_TO_EMBED_LR} \
    --adamw_scalar_lr ${ADAMW_SCALAR_LR} \
    --adamw_other_lr ${ADAMW_OTHER_LR} \
    --advantage_norm ${ADVANTAGE_NORM} \
    --global_std_min ${GLOBAL_STD_MIN} \
    --skip_degenerate_threshold ${SKIP_DEGENERATE_THRESHOLD} \
    --min_reward_std_to_update ${MIN_REWARD_STD_TO_UPDATE} \
    --min_unique_completion_ratio_to_update ${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE} \
    --min_reward_mean_to_update ${MIN_REWARD_MEAN_TO_UPDATE} \
    --min_reward_format_to_update ${MIN_REWARD_FORMAT_TO_UPDATE} \
    --skip_consensus ${SKIP_CONSENSUS} \
    --max_steps ${MAX_STEPS} \
    --val_check_interval ${VAL_CHECK_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --save_top_k ${SAVE_TOP_K} \
    --seed ${SEED} \
    --data_dir "${SUDOKU_DATA_DIR_V2}" \
    --num_gpus ${NUM_GPUS} \
    --float_precision "32-true" \
    --wandb_project "nlp_sudoku_rl" \
    --wandb_mode "${WANDB_MODE}" \
    --run_name "${RUN_NAME}_${current_time}" \
    --checkpoint_dir "${EXP_CKPT_DIR}" \
    --traj_log_interval ${TRAJ_LOG_INTERVAL} \
    --traj_output_dir "${TRAJ_OUTPUT_DIR}" \
    --traj_num_samples ${TRAJ_NUM_SAMPLES} \
    --collapse_check_window ${COLLAPSE_CHECK_WINDOW}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_SFT_DIR}" "sudoku_rl_grpo_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ Sudoku RL (GRPO) 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ Sudoku RL (GRPO) 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi
exit "$TRAIN_EXIT_CODE"

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "日志已保存到: ${LOG_FILE}"
echo "轨迹 JSONL:  ${TRAJ_OUTPUT_DIR}/trajectories"

if [ "${ANALYZE_AFTER_TRAIN:-1}" = "1" ] && [ -f "${SCRIPT_DIR}/../scripts/analyze_rl_run.py" ]; then
    echo "正在生成 RL 分析报告..."
    python "${SCRIPT_DIR}/../scripts/analyze_rl_run.py" "${EXP_SFT_DIR}" || true
fi

exit "$TRAIN_EXIT_CODE"
