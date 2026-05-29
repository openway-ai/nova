#!/bin/bash

################################################################################
# EBT Sudoku RL (GRPO) 训练脚本
# ─────────────────────────────────────────────────────────────────────────────
# 基于 SFT v3 最佳 checkpoint 进行 GRPO 强化学习后训练。
#
# 算法: Group Relative Policy Optimization (GRPO)
#   - 每个 prompt 生成 N 个 completion
#   - 多组件奖励 (format + accuracy + validity + full_solve)
#   - PPO-clipped policy gradient + KL penalty
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

# RL 实验 ID
export EXP_ID="d26-ctx2048-sudoku-rl-gspo-$(date +%Y%m%d)"
# export EXP_ID="d26-ctx2048-sudoku-rl-reinforce-$(date +%Y%m%d)"

################################################################################
# GRPO 超参数
################################################################################
NUM_GENERATIONS="${NUM_GENERATIONS:-12}"            # 12 is a stable smoke default; use 16 after curves are healthy
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-165}"     # 9x9 board ≤162 chars; tail tokens add noise
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-192}"
TEMPERATURE="${TEMPERATURE:-0.75}"                  # lower than 0.9: reduce invalid-format exploration
TOP_P="${TOP_P:-0.85}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-6}" # sub-batch for generation (VRAM management)

NUM_ITERATIONS="${NUM_ITERATIONS:-1}"              # legacy loss recomputation; keep 1 when using GSPO_UPDATE_EPOCHS
GSPO_UPDATE_EPOCHS="${GSPO_UPDATE_EPOCHS:-1}"  # true optimizer steps per rollout for energy_gspo
EPSILON="${EPSILON:-0.2}"                   # PPO clip range (only used by energy_gspo)
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"     # verl/DAPO-style asymmetric GSPO clip
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
BETA="${BETA:-0.3}"                         # stronger anchor after negative energy-drift collapse
ENERGY_KL_MODE="${ENERGY_KL_MODE:-symmetric_huber}"
ENERGY_KL_HUBER_DELTA="${ENERGY_KL_HUBER_DELTA:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"       # lower than collapsed v2/v3 runs
# RL_LOSS_TYPE="energy_reinforce"  # removes 1/|y| dilution
RL_LOSS_TYPE="${RL_LOSS_TYPE:-energy_gspo}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.5}"
MAX_GRAD_PER_PARAM="${MAX_GRAD_PER_PARAM:-0.03}"
WARMUP_STEPS="${WARMUP_STEPS:-80}"

# ── Optimizer (v3 P0+P1+muon) ─────────────────────────────────────────────────
# adamw      : v3 P0 multi-group AdamW (transformer/v2e/scalar/other split LR)
# muon_adamw : Muon for transformer matrices + AdamW for the rest (matches SFT)
RL_OPTIMIZER="${RL_OPTIMIZER:-adamw}"       # keep AdamW first; Muon can amplify collapse in short RL runs
MUON_LR="${MUON_LR:-2e-4}"                  # default = SFT muon_lr (2e-3) / 10
ADVANTAGE_NORM="${ADVANTAGE_NORM:-group_mean_global_std}"  # batch-wide std prevents tiny intra-group std blow-ups
GLOBAL_STD_MIN="${GLOBAL_STD_MIN:-0.2}"
SKIP_DEGENERATE_THRESHOLD="${SKIP_DEGENERATE_THRESHOLD:-0.85}"
MIN_REWARD_STD_TO_UPDATE="${MIN_REWARD_STD_TO_UPDATE:-0.01}"
MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE="${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE:-0.0}"

MAX_STEPS="${MAX_STEPS:-600}"                # smoke first; raise only after analysis_overview is healthy
           
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-25}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
SEED="${SEED:-42}"

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
    "warmup_steps=${WARMUP_STEPS}" \
    "gradient_clip_val=${GRADIENT_CLIP_VAL}" \
    "max_grad_per_param=${MAX_GRAD_PER_PARAM}" \
    "advantage_norm=${ADVANTAGE_NORM}" \
    "global_std_min=${GLOBAL_STD_MIN}" \
    "skip_degenerate_threshold=${SKIP_DEGENERATE_THRESHOLD}" \
    "min_reward_std_to_update=${MIN_REWARD_STD_TO_UPDATE}" \
    "min_unique_completion_ratio_to_update=${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE}" \
    "rl_loss_type=${RL_LOSS_TYPE}" \
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
echo "Max steps:               ${MAX_STEPS}"
echo "Val interval:            ${VAL_CHECK_INTERVAL}"
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
#
# GRPO Config:
#   num_generations=${NUM_GENERATIONS}
#   max_completion_length=${MAX_COMPLETION_LENGTH}
#   temperature=${TEMPERATURE}
#   epsilon=${EPSILON}
#   clip_ratio_low=${CLIP_RATIO_LOW}
#   clip_ratio_high=${CLIP_RATIO_HIGH}
#   beta=${BETA}
#   energy_kl_mode=${ENERGY_KL_MODE}
#   learning_rate=${LEARNING_RATE}
#   max_steps=${MAX_STEPS}
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

set -e

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
    --advantage_norm ${ADVANTAGE_NORM} \
    --global_std_min ${GLOBAL_STD_MIN} \
    --skip_degenerate_threshold ${SKIP_DEGENERATE_THRESHOLD} \
    --min_reward_std_to_update ${MIN_REWARD_STD_TO_UPDATE} \
    --min_unique_completion_ratio_to_update ${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE} \
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
    --checkpoint_dir "${EXP_CKPT_DIR}"

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_SFT_DIR}" "sudoku_rl_grpo_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ Sudoku RL (GRPO) 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ Sudoku RL (GRPO) 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"

if [ "${ANALYZE_AFTER_TRAIN:-1}" = "1" ] && [ -f "${SCRIPT_DIR}/../scripts/analyze_rl_run.py" ]; then
    echo "正在生成 RL 分析报告..."
    python "${SCRIPT_DIR}/../scripts/analyze_rl_run.py" "${EXP_SFT_DIR}" || true
fi
