#!/bin/bash

################################################################################
# EBT GSM8K RL (GRPO) 训练脚本
# ─────────────────────────────────────────────────────────────────────────────
# 基于 SFT checkpoint 对 GSM8K 数学推理任务进行 GRPO 强化学习后训练。
#
# 算法: Energy-REINFORCE with KL anchor
#   - 每个 prompt 生成 N 个 completion
#   - 奖励 = format(0.3) + exact_match(0.7) + length_penalty(-0.1)
#   - sequence-level energy-KL anchor (防 policy collapse)
#
# 日志与轨迹采样:
#   - 每 TRAJ_LOG_INTERVAL 步打印 [DBG-RL-PERIODIC] 样本
#   - JSONL 持久化到 ${EXP_DIR}/trajectories/
#   - collapse 预警: unique_ratio < 0.3 持续 K 步 → [WARN-COLLAPSE] + dump
#
# 参考: run_ebt_sudoku_rl.sh (风格对齐)
################################################################################

set -e

### 基础配置 ###
export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### SFT 权重 (RL 起点) ###
# TODO: 替换为实际的 SFT checkpoint 路径
SFT_CKPT="${SFT_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-0.6-20260508/sft_train/checkpoints/s=step=2990-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4782.ckpt}"

### GSM8K 数据路径 ###
# 直接复用 run_ebt_sft.sh pipeline 已下载好的 HF 缓存 (openai/gsm8k 完整数据)
#   train: 7473 examples (gsm8k-train.arrow)
#   test:  1319 examples (gsm8k-test.arrow) — RL 的 "val" 自动映射到 HF test split
# 数据集加载逻辑见 gsm8k_dataset_rl.py:_load_gsm8k
# 通过传入 HF cache 根目录, dataset 会自动定位 openai___gsm8k/main/<ver>/<hash>/gsm8k-<split>.arrow
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-${NANOCHAT_SFT_DATA_DIR:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data}/gsm8k}"

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

# RL 实验 ID
# export EXP_ID="d26-ctx2048-gsm8k-rl-gspo-$(date +%Y%m%d)"
export EXP_ID="d26-ctx2048-gsm8k-rl-reinforce-$(date +%Y%m%d)"

################################################################################
# GRPO 超参数
# 设计参考:
#   - v2 sudoku RL 经验: KL anchor(BETA≥0.05) + 低LR 防 collapse
#   - GSM8K max_completion_length 更大 (512 vs 180): 数学推理需要 CoT 空间
#   - group_mean_global_std: 防止 advantage 在低多样性批次中爆炸
################################################################################
NUM_GENERATIONS=8             # completions per prompt (GSM8K 比 sudoku 简单，8 足够)

# MAX_COMPLETION_LENGTH=512     # CoT 推理需要更长空间 (数学 ≤512 tokens 够用)
# MAX_PROMPT_LENGTH=256         # GSM8K 题目通常较短

MAX_COMPLETION_LENGTH=320     # ↓ 1024 → 320: GSM8K answer ≤200 tok; O(L²) generation, 16x speedup
MAX_PROMPT_LENGTH=192         # ↓ 512 → 192: actual GSM8K prompt is ~120 tok

TEMPERATURE=0.9
TOP_P=0.9
GENERATION_BATCH_SIZE=8       # ↑ 4 → 8: more parallel sampling, fits VRAM after length reduction

NUM_ITERATIONS=1              # GRPO inner loop
EPSILON=0.2                   # PPO clip range (仅 energy_gspo 用)
BETA=0.2                      # KL anchor: aligned with sudoku v3. v2 evidence shows BETA<0.1 → grad_norm 20× drift
LEARNING_RATE=2e-6            # 同 sudoku v2; 能量梯度幅度大，不宜太高
RL_LOSS_TYPE="energy_reinforce"
# RL_LOSS_TYPE="energy_gspo"

WEIGHT_DECAY=0.01
GRADIENT_CLIP_VAL=1.0         # v3 P1: 0.3 → 1.0 (KL anchor BETA=0.2 主导稳定; per-param clip 之前压制了学习信号)
MAX_GRAD_PER_PARAM=0.05       # v3 P1: 0.02 → 0.05 (v2 实测真实梯度信号需要更大空间)
WARMUP_STEPS=50

# ── Optimizer (v3 P0+P1+muon) ─────────────────────────────────────────────────
# adamw      : v3 P0 multi-group AdamW (transformer/v2e/scalar/other split LR)
# muon_adamw : Muon for transformer matrices + AdamW for the rest (matches SFT)
RL_OPTIMIZER="${RL_OPTIMIZER:-adamw}"          # set to "muon_adamw" to enable Muon hybrid
MUON_LR="${MUON_LR:-2e-4}"                  # default = SFT muon_lr (2e-3) / 10
ADVANTAGE_NORM="group_mean_global_std"
GLOBAL_STD_MIN=0.1
SKIP_DEGENERATE_THRESHOLD=0.9

MAX_STEPS=5000                 # 首次验证: 短训 500 步观察 reward 是否上升
VAL_CHECK_INTERVAL=100
LOG_INTERVAL=5
SAVE_TOP_K=2
SEED=42

################################################################################
# 轨迹采样与日志配置
################################################################################
TRAJ_LOG_INTERVAL=50          # 每 50 step 打印 [DBG-RL-PERIODIC] 样本
TRAJ_NUM_SAMPLES=2            # 每次打印 2 条 completion
COLLAPSE_CHECK_WINDOW=5       # K 步连续 degenerate → WARN-COLLAPSE

# LOG_LEVEL: DEBUG / INFO / WARN / ERROR
export LOG_LEVEL="INFO"

################################################################################
# GPU 配置
################################################################################
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-6}

################################################################################
# 数据与 checkpoint 检查
################################################################################
echo ""
echo "=================================="
echo "  检查 GSM8K 数据与 SFT checkpoint"
echo "=================================="

# HF cache 根目录下应能找到 openai___gsm8k/main/*/*/gsm8k-{train,test}.arrow
GSM8K_TRAIN_ARROW=$(ls ${GSM8K_DATA_PATH}/openai___gsm8k/main/*/*/gsm8k-train.arrow 2>/dev/null | head -1)
GSM8K_TEST_ARROW=$(ls ${GSM8K_DATA_PATH}/openai___gsm8k/main/*/*/gsm8k-test.arrow 2>/dev/null | head -1)

if [ -z "${GSM8K_TRAIN_ARROW}" ] || [ -z "${GSM8K_TEST_ARROW}" ]; then
    echo "✗ GSM8K HF cache 不完整: ${GSM8K_DATA_PATH}"
    echo "  期望路径: \${GSM8K_DATA_PATH}/openai___gsm8k/main/*/*/gsm8k-{train,test}.arrow"
    echo "  请先跑过 SFT pipeline 以生成 HF cache, 或设置 GSM8K_DATA_PATH 指向其他 HF cache 根"
    exit 1
fi
echo "✓ GSM8K HF cache 根:  ${GSM8K_DATA_PATH}"
echo "  train arrow:  ${GSM8K_TRAIN_ARROW}"
echo "  test arrow:   ${GSM8K_TEST_ARROW}"

if [ ! -f "${SFT_CKPT}" ]; then
    echo "✗ SFT checkpoint 不存在: ${SFT_CKPT}"
    echo "  请设置 SFT_CKPT 环境变量"
    exit 1
fi
echo "✓ SFT checkpoint: ${SFT_CKPT}"
echo ""

################################################################################
# 实验目录布局
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${SFT_CKPT}"
exp_init_sft "$0"
export RUN_NAME="${EXP_ID}-gsm8k-rl-grpo"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

# 轨迹输出目录跟着实验目录走 (EXP_SFT_DIR 由 exp_init_sft 自动 export, 含 v{N} 后缀)
TRAJ_OUTPUT_DIR="${EXP_SFT_DIR}/trajectories"
export TRAJ_OUTPUT_DIR

exp_save_hparams "${EXP_SFT_DIR}" \
    "model_size=${MODEL_SIZE}" \
    "algorithm=grpo" \
    "task=gsm8k" \
    "sft_checkpoint=${SFT_CKPT}" \
    "num_generations=${NUM_GENERATIONS}" \
    "max_completion_length=${MAX_COMPLETION_LENGTH}" \
    "temperature=${TEMPERATURE}" \
    "epsilon=${EPSILON}" \
    "beta=${BETA}" \
    "learning_rate=${LEARNING_RATE}" \
    "warmup_steps=${WARMUP_STEPS}" \
    "gradient_clip_val=${GRADIENT_CLIP_VAL}" \
    "max_grad_per_param=${MAX_GRAD_PER_PARAM}" \
    "advantage_norm=${ADVANTAGE_NORM}" \
    "skip_degenerate_threshold=${SKIP_DEGENERATE_THRESHOLD}" \
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
echo "  EBT GSM8K RL (GRPO) 训练"
echo "=================================="
echo "SFT checkpoint:          ${SFT_CKPT}"
echo "GSM8K data:              ${GSM8K_DATA_PATH}"
echo "Algorithm:               GRPO (${RL_LOSS_TYPE})"
echo "Num generations:         ${NUM_GENERATIONS}"
echo "Max completion length:   ${MAX_COMPLETION_LENGTH}"
echo "Temperature:             ${TEMPERATURE}"
echo "Beta (KL anchor):        ${BETA}"
echo "Learning rate:           ${LEARNING_RATE}"
echo "Warmup steps:            ${WARMUP_STEPS}"
echo "Gradient clip (global):  ${GRADIENT_CLIP_VAL}"
echo "Max grad per param:      ${MAX_GRAD_PER_PARAM}"
echo "Advantage norm:          ${ADVANTAGE_NORM}"
echo "Skip degen threshold:    ${SKIP_DEGENERATE_THRESHOLD}"
echo "Max steps:               ${MAX_STEPS}"
echo "Val interval:            ${VAL_CHECK_INTERVAL}"
echo "GPUs:                    ${NUM_GPUS}"
echo "Traj log interval:       ${TRAJ_LOG_INTERVAL}"
echo "Traj output dir:         ${TRAJ_OUTPUT_DIR}"
echo "Log level:               ${LOG_LEVEL}"
echo "Log file:                ${LOG_FILE}"
echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT GSM8K RL (GRPO) 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# SFT From:        ${SFT_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
# Traj Dir:        ${TRAJ_OUTPUT_DIR}
#
# GRPO Config:
#   task=gsm8k
#   num_generations=${NUM_GENERATIONS}
#   max_completion_length=${MAX_COMPLETION_LENGTH}
#   temperature=${TEMPERATURE}
#   beta=${BETA}
#   learning_rate=${LEARNING_RATE}
#   rl_loss_type=${RL_LOSS_TYPE}
#   max_steps=${MAX_STEPS}
#
# Log grep patterns:
#   [GRPO]            — per-step training metrics
#   [DBG-RL-PERIODIC] — periodic completion samples (every TRAJ_LOG_INTERVAL steps)
#   [WARN-COLLAPSE]   — collapse detection alert
#   [METRIC]          — aligned columnar metrics line
#   [INFO]            — informational messages
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
echo "[开始 GSM8K RL 训练]"
echo "================================================================================"
echo ""

set -e

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m openebm.elm.rl.train_rl_gsm8k \
    --sft_checkpoint "${SFT_CKPT}" \
    --data_path "${GSM8K_DATA_PATH}" \
    --num_generations ${NUM_GENERATIONS} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --generation_batch_size ${GENERATION_BATCH_SIZE} \
    --num_iterations ${NUM_ITERATIONS} \
    --epsilon ${EPSILON} \
    --beta ${BETA} \
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
    --max_steps ${MAX_STEPS} \
    --val_check_interval ${VAL_CHECK_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --save_top_k ${SAVE_TOP_K} \
    --seed ${SEED} \
    --num_gpus ${NUM_GPUS} \
    --float_precision "32-true" \
    --wandb_project "nlp_gsm8k_rl" \
    --wandb_mode "${WANDB_MODE}" \
    --run_name "${RUN_NAME}_${current_time}" \
    --checkpoint_dir "${EXP_CKPT_DIR}" \
    --traj_log_interval ${TRAJ_LOG_INTERVAL} \
    --traj_output_dir "${TRAJ_OUTPUT_DIR}" \
    --traj_num_samples ${TRAJ_NUM_SAMPLES} \
    --collapse_check_window ${COLLAPSE_CHECK_WINDOW}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_SFT_DIR}" "gsm8k_rl_grpo_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ GSM8K RL (GRPO) 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ GSM8K RL (GRPO) 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
echo "轨迹 JSONL:  ${TRAJ_OUTPUT_DIR}"

################################################################################
# 日志分析速查
################################################################################
# grep 模式:
#   实时 loss/reward:   grep -E '\[GRPO\] step=' ${LOG_FILE}
#   周期采样:           grep '\[DBG-RL-PERIODIC\]' ${LOG_FILE}
#   collapse 预警:      grep '\[WARN-COLLAPSE\]' ${LOG_FILE}
#   对齐列 metrics:     grep '\[METRIC\]' ${LOG_FILE}
#
# 轨迹分析示例:
#   # 查看 step 50 的所有 completion 及 reward
#   cat ${TRAJ_OUTPUT_DIR}/step_000050/rollout_samples.jsonl | python3 -c \
#     "import sys,json; [print(json.loads(l)['reward'], json.loads(l)['completion'][:100]) for l in sys.stdin]"
#
#   # 统计各 step 的 exact_match 准确率
#   grep -h '' ${TRAJ_OUTPUT_DIR}/*/summary.json | python3 -c \
#     "import sys,json; [print(json.loads(l)['step'], json.loads(l).get('reward_mean','?')) for l in sys.stdin]"
#
#   # 找到所有 collapse dump
#   ls ${TRAJ_OUTPUT_DIR}/../collapse_dump/
################################################################################
