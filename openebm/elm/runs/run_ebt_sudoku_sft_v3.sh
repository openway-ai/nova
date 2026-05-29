#!/bin/bash

################################################################################
# EBT Sudoku SFT V3 训练脚本
# ─────────────────────────────────────────────────────────────────────────────
# v3 vs v2 (no synthetic CoT, no Sakana CTC, no KD):
#   - 自适应 sudoku_ratio (`three_phase_with_guard`)              [P3+P4]
#       warmup 0.6 → focus 0.85 → consolidate 0.72; phase cutoffs scale with MAX_STEPS
#       guard: valid_loss_sft 漂移 > 0.03 时下调 0.05 (floor 0.50)
#   - RRN-train 难度感知重采样 (`three_phase`)                   [P5]
#       hard 17-22 在 focus 阶段权重 0.60
#   - blank-position 损失加权 K=2.0                              [P1]
#   - MCMC stays at 2, randomize=0 by default (OOM-safe); override if memory allows
#   - Prompt 5→20, response 3→8, 新增 flat 格式 (70/30 grid/flat)[P8 user req]
#   - 报告 valid_loss_sft / valid_loss_sudoku / valid_loss_balanced
#   - SAVE_TOP_K 2 → 5 + periodic ckpt, 保留更多 RL 起点候选
################################################################################

set -e

### 基础配置 ###


export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 预训练权重 (与 v2 一致) ###
# PRETRAIN_CKPT="${PRETRAIN_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-20260426/sft_train.v4/checkpoints/s=step=562-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.2264.ckpt}"

# 最新core最高的 sft权重 core 0.20
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/mnt/shared-storage-user/luyudong/nova-sft/nova/logs/ebt_runs/d26-ctx2048-20260513/sft_train.v2/checkpoints/s=step=1703-d26-ctx2048-lr0.00024-bs1x32-muon_adamw-valid_loss=valid_loss=0.7257.ckpt}"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_SFT_DATA_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

# Offline 模式
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Sudoku V2 数据缓存目录 (v3 复用 v2 数据缓存, 数据集本身不变)
export SUDOKU_DATA_DIR_V2="${SUDOKU_DATA_DIR_V2:-/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2}"

# v3: 初始 sudoku_ratio (warmup 阶段值; AdaptiveRatioCallback 接管之后的调度)
export SUDOKU_RATIO="${SUDOKU_RATIO:-0.6}"

# v3: 用单独 EXP_ID 防止与 v2 (0.6 / 0.4 / 0.8 sweep) 目录冲突
export EXP_ID="${EXP_ID:-d26-ctx2048-sudoku-mixed-v3p1-$(date +%Y%m%d)}"

################################################################################
# EBT 核心超参数 (v3: MCMC_NUM_STEPS 已回退至 2, randomize=0 以缓解 OOM)
################################################################################
MCMC_STEP_SIZE="${MCMC_STEP_SIZE:-500.0}"
MCMC_STEP_SIZE_LR_MULTIPLIER="${MCMC_STEP_SIZE_LR_MULTIPLIER:-750}"
MCMC_NUM_STEPS="${MCMC_NUM_STEPS:-2}"              # OOM-safe default; try 3 only after a memory smoke test.
RANDOMIZE_MCMC_NUM_STEPS="${RANDOMIZE_MCMC_NUM_STEPS:-0}"
EBT_TYPE="${EBT_TYPE:-time_embed}"
NORMALIZE_INITIAL_CONDITION="${NORMALIZE_INITIAL_CONDITION:-true}"
DENOISING_INITIAL_CONDITION="${DENOISING_INITIAL_CONDITION:-random_noise}"
MCMC_STEP_SIZE_LEARNABLE="${MCMC_STEP_SIZE_LEARNABLE:-true}"
NO_MCMC_DETACH="${NO_MCMC_DETACH:-false}"

################################################################################
# v3 新增: 自适应调度 + 难度调度 + blank 位置加权
################################################################################
SUDOKU_RATIO_SCHEDULE="${SUDOKU_RATIO_SCHEDULE:-three_phase_with_guard}"   # P3+P4
SUDOKU_DIFFICULTY_SCHEDULE="${SUDOKU_DIFFICULTY_SCHEDULE:-three_phase}"    # P5
SUDOKU_BLANK_LOSS_WEIGHT="${SUDOKU_BLANK_LOSS_WEIGHT:-2.0}"                # P1: K=2.0
SUDOKU_WARMUP_RATIO="${SUDOKU_WARMUP_RATIO:-0.60}"
SUDOKU_FOCUS_RATIO="${SUDOKU_FOCUS_RATIO:-0.85}"
SUDOKU_CONSOLIDATE_RATIO="${SUDOKU_CONSOLIDATE_RATIO:-0.72}"
SUDOKU_GUARD_SFT_BASELINE="${SUDOKU_GUARD_SFT_BASELINE:-1.121}"
SUDOKU_GUARD_TOLERANCE="${SUDOKU_GUARD_TOLERANCE:-0.03}"
SUDOKU_RATIO_FLOOR="${SUDOKU_RATIO_FLOOR:-0.50}"
SUDOKU_RATIO_CEIL="${SUDOKU_RATIO_CEIL:-0.85}"
SUDOKU_RATIO_STEP="${SUDOKU_RATIO_STEP:-0.05}"
SUDOKU_WARMUP_BUCKET_WEIGHTS="${SUDOKU_WARMUP_BUCKET_WEIGHTS:-0.33 0.33 0.34}"
SUDOKU_FOCUS_BUCKET_WEIGHTS="${SUDOKU_FOCUS_BUCKET_WEIGHTS:-0.60 0.28 0.12}"
SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS="${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS:-0.45 0.35 0.20}"
DATASET_SEED="${DATASET_SEED:-20260529}"

################################################################################
# Batch 配置 (与 v2 一致)
################################################################################
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
# DEVICE_BATCH_SIZE=2   # v3 优化: 1→2 (P2 回退后腾出显存, 减少 micro-step 数)
# GRAD_ACCUM=16         # v3 优化: 32→16 (保持 global bs = 8*2*16 = 256 不变)

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"


################################################################################
# SFT 学习率配置 (与 v2 一致)
################################################################################
PEAK_LR="${PEAK_LR:-0.00005}"
SFT_MUON_LR="${SFT_MUON_LR:-0.002}"
SFT_EMBEDDING_LR="${SFT_EMBEDDING_LR:-0.03}"
SFT_VOCAB_TO_EMBED_LR="${SFT_VOCAB_TO_EMBED_LR:-0.001}"
SFT_SCALAR_LR="${SFT_SCALAR_LR:-0.004}"

################################################################################
# 优化器配置 (与 v2 一致)
################################################################################
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BETA1="${BETA1:-0.8}"
BETA2="${BETA2:-0.95}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"

################################################################################
# 训练步数 (与 v2 一致)
################################################################################
MAX_STEPS="${MAX_STEPS:-4500}"
MAX_SCHEDULING_STEPS="${MAX_SCHEDULING_STEPS:-${MAX_STEPS}}"
SUDOKU_PHASE1_STEPS="${SUDOKU_PHASE1_STEPS:-$((MAX_STEPS / 6))}"
SUDOKU_PHASE2_STEPS="${SUDOKU_PHASE2_STEPS:-$((MAX_STEPS * 4 / 5))}"

################################################################################
# 验证与数据加载配置
################################################################################
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
VAL_STEPS="${VAL_STEPS:-200}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"

################################################################################
# 优化选项配置 (与 v2 一致)
################################################################################
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.20}"
FINAL_LR_FRAC="${FINAL_LR_FRAC:-0.05}"
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio ${WARMUP_RATIO} --warmdown_ratio ${WARMDOWN_RATIO} --final_lr_frac ${FINAL_LR_FRAC} --optimizer muon_adamw --muon_lr ${SFT_MUON_LR} --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr ${SFT_EMBEDDING_LR} --adamw_vocab_to_embed_lr ${SFT_VOCAB_TO_EMBED_LR} --adamw_scalar_lr ${SFT_SCALAR_LR} --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################
COMPILE_FLAGS="${COMPILE_FLAGS:---compile_model --compile_mode full}"
WANDB_FLAGS=""

################################################################################
# Checkpoint 管理: 保留更多候选，便于后续用 sample eval 选择 RL 起点
################################################################################
SAVE_TOP_K="${SAVE_TOP_K:-5}"
SAVE_PERIODIC_STEPS="${SAVE_PERIODIC_STEPS:-500}"
RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN="${RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN:-1}"
SUDOKU_CKPT_SWEEP_MAX_CKPTS="${SUDOKU_CKPT_SWEEP_MAX_CKPTS:-8}"
SUDOKU_CKPT_SWEEP_NUM_SAMPLES="${SUDOKU_CKPT_SWEEP_NUM_SAMPLES:-256}"
SUDOKU_CKPT_SWEEP_PROMPT_STYLE="${SUDOKU_CKPT_SWEEP_PROMPT_STYLE:-fixed}"

################################################################################
# 数据集检查 (与 v2 一致)
################################################################################
echo ""
echo "=================================="
echo "  检查 Sudoku V2 数据集 (v3 复用)"
echo "=================================="
echo "数据集路径: ${SUDOKU_DATA_DIR_V2}"

MISSING=()
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_train.pt" ] && MISSING+=("rrn_train.pt")
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_val.pt" ] && MISSING+=("rrn_val.pt")
[ ! -f "${SUDOKU_DATA_DIR_V2}/satnet_test.pt" ] && MISSING+=("satnet_test.pt")

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
echo "✓ satnet_test.pt"

# 检查 checkpoint
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "✗ Checkpoint 不存在: ${PRETRAIN_CKPT}"
    exit 1
fi
echo "✓ Checkpoint: ${PRETRAIN_CKPT}"
echo ""

################################################################################
# 实验目录布局
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${PRETRAIN_CKPT}"
exp_init_sft "$0"
export RUN_NAME="${EXP_ID}-sudoku-mixed-sft-v3"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

exp_save_hparams "${EXP_DIR}/sft_train" \
    "model_size=${MODEL_SIZE}" \
    "context_length=${CONTEXT_LENGTH}" \
    "peak_lr=${PEAK_LR}" \
    "sft_muon_lr=${SFT_MUON_LR}" \
    "sft_embedding_lr=${SFT_EMBEDDING_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "device_batch_size=${DEVICE_BATCH_SIZE}" \
    "grad_accum=${GRAD_ACCUM}" \
    "num_gpus=${NUM_GPUS}" \
    "max_steps=${MAX_STEPS}" \
    "max_scheduling_steps=${MAX_SCHEDULING_STEPS}" \
    "warmup_ratio=${WARMUP_RATIO}" \
    "warmdown_ratio=${WARMDOWN_RATIO}" \
    "final_lr_frac=${FINAL_LR_FRAC}" \
    "mcmc_step_size=${MCMC_STEP_SIZE}" \
    "mcmc_lr_multiplier=${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
    "mcmc_num_steps=${MCMC_NUM_STEPS}" \
    "randomize_mcmc_num_steps=${RANDOMIZE_MCMC_NUM_STEPS}" \
    "pretrain_ckpt=${PRETRAIN_CKPT}" \
    "optimizer=muon_adamw" \
    "dataset=sudoku_mixed_v3" \
    "sudoku_ratio_initial=${SUDOKU_RATIO}" \
    "sudoku_ratio_schedule=${SUDOKU_RATIO_SCHEDULE}" \
    "sudoku_phase1_steps=${SUDOKU_PHASE1_STEPS}" \
    "sudoku_phase2_steps=${SUDOKU_PHASE2_STEPS}" \
    "sudoku_warmup_ratio=${SUDOKU_WARMUP_RATIO}" \
    "sudoku_focus_ratio=${SUDOKU_FOCUS_RATIO}" \
    "sudoku_consolidate_ratio=${SUDOKU_CONSOLIDATE_RATIO}" \
    "sudoku_guard_sft_baseline=${SUDOKU_GUARD_SFT_BASELINE}" \
    "sudoku_guard_tolerance=${SUDOKU_GUARD_TOLERANCE}" \
    "sudoku_difficulty_schedule=${SUDOKU_DIFFICULTY_SCHEDULE}" \
    "sudoku_warmup_bucket_weights=${SUDOKU_WARMUP_BUCKET_WEIGHTS}" \
    "sudoku_focus_bucket_weights=${SUDOKU_FOCUS_BUCKET_WEIGHTS}" \
    "sudoku_consolidate_bucket_weights=${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS}" \
    "dataset_seed=${DATASET_SEED}" \
    "sudoku_blank_loss_weight=${SUDOKU_BLANK_LOSS_WEIGHT}" \
    "save_top_k=${SAVE_TOP_K}" \
    "save_periodic_steps=${SAVE_PERIODIC_STEPS}" \
    "run_sudoku_ckpt_sweep_after_train=${RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN}" \
    "sudoku_ckpt_sweep_max_ckpts=${SUDOKU_CKPT_SWEEP_MAX_CKPTS}" \
    "sudoku_ckpt_sweep_num_samples=${SUDOKU_CKPT_SWEEP_NUM_SAMPLES}" \
    "sudoku_ckpt_sweep_prompt_style=${SUDOKU_CKPT_SWEEP_PROMPT_STYLE}"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  EBT Sudoku SFT V3 训练"
echo "=================================="
echo "Pretrain ckpt:           ${PRETRAIN_CKPT}"
echo "Dataset:                 sudoku_mixed (v3)"
echo "sudoku_ratio (initial):  ${SUDOKU_RATIO}"
echo "sudoku_ratio_schedule:   ${SUDOKU_RATIO_SCHEDULE}    [P3+P4]"
echo "phase cutoffs:           ${SUDOKU_PHASE1_STEPS}, ${SUDOKU_PHASE2_STEPS}"
echo "phase ratios:            ${SUDOKU_WARMUP_RATIO}, ${SUDOKU_FOCUS_RATIO}, ${SUDOKU_CONSOLIDATE_RATIO}"
echo "difficulty_schedule:     ${SUDOKU_DIFFICULTY_SCHEDULE}            [P5]"
echo "bucket weights:          warm=[${SUDOKU_WARMUP_BUCKET_WEIGHTS}] focus=[${SUDOKU_FOCUS_BUCKET_WEIGHTS}] consolidate=[${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS}]"
echo "dataset seed:            ${DATASET_SEED}"
echo "blank_loss_weight:       ${SUDOKU_BLANK_LOSS_WEIGHT}              [P1]"
echo "mcmc_num_steps:          ${MCMC_NUM_STEPS} (rand=${RANDOMIZE_MCMC_NUM_STEPS})        [P2]"
echo "Peak LR:                 ${PEAK_LR}"
echo "LR schedule:             warmup=${WARMUP_RATIO}, warmdown=${WARMDOWN_RATIO}, final_lr_frac=${FINAL_LR_FRAC}"
echo "Weight decay:            ${WEIGHT_DECAY}"
echo "Max steps:               ${MAX_STEPS}"
echo "Max scheduling steps:    ${MAX_SCHEDULING_STEPS}"
echo "Val interval:            ${VAL_CHECK_INTERVAL}"
echo "Val steps:               ${VAL_STEPS}"
echo "Limit val batches:       ${LIMIT_VAL_BATCHES}"
echo "Save top-k:              ${SAVE_TOP_K}"
echo "Save periodic steps:     ${SAVE_PERIODIC_STEPS}"
echo "Post-train ckpt sweep:   ${RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN} (max=${SUDOKU_CKPT_SWEEP_MAX_CKPTS}, samples=${SUDOKU_CKPT_SWEEP_NUM_SAMPLES})"
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
#                    EBT Sudoku SFT V3 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Pretrain From:   ${PRETRAIN_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# v3 (no synthetic CoT, no KD, no Sakana CTC):
#   P1 blank loss weight K=${SUDOKU_BLANK_LOSS_WEIGHT}
#   P2 mcmc_num_steps=${MCMC_NUM_STEPS} randomize=${RANDOMIZE_MCMC_NUM_STEPS}
#   P3+P4 sudoku_ratio_schedule=${SUDOKU_RATIO_SCHEDULE}
#          phase cutoffs=${SUDOKU_PHASE1_STEPS},${SUDOKU_PHASE2_STEPS}
#          phase ratios=${SUDOKU_WARMUP_RATIO},${SUDOKU_FOCUS_RATIO},${SUDOKU_CONSOLIDATE_RATIO}
#   P5 difficulty_schedule=${SUDOKU_DIFFICULTY_SCHEDULE}
#      bucket weights warm=[${SUDOKU_WARMUP_BUCKET_WEIGHTS}] focus=[${SUDOKU_FOCUS_BUCKET_WEIGHTS}] consolidate=[${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS}]
#   P8 prompt 20, response 8, format grid+flat
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
echo "[Sudoku SFT V3 训练配置]"
echo "================================================================================"
echo "Pretrain Checkpoint:        ${PRETRAIN_CKPT}"
echo "Dataset:                    sudoku_mixed (v3)"
echo "sudoku_ratio (initial):     ${SUDOKU_RATIO}"
echo "sudoku_ratio_schedule:      ${SUDOKU_RATIO_SCHEDULE}"
echo "phase cutoffs:              ${SUDOKU_PHASE1_STEPS}, ${SUDOKU_PHASE2_STEPS}"
echo "phase ratios:               ${SUDOKU_WARMUP_RATIO}, ${SUDOKU_FOCUS_RATIO}, ${SUDOKU_CONSOLIDATE_RATIO}"
echo "sudoku_difficulty_schedule: ${SUDOKU_DIFFICULTY_SCHEDULE}"
echo "bucket weights:             warm=[${SUDOKU_WARMUP_BUCKET_WEIGHTS}] focus=[${SUDOKU_FOCUS_BUCKET_WEIGHTS}] consolidate=[${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS}]"
echo "dataset_seed:               ${DATASET_SEED}"
echo "sudoku_blank_loss_weight:   ${SUDOKU_BLANK_LOSS_WEIGHT}"
echo "mcmc_num_steps:             ${MCMC_NUM_STEPS} (randomize=${RANDOMIZE_MCMC_NUM_STEPS})"
echo "Peak LR:                    ${PEAK_LR}"
echo "LR schedule:                warmup=${WARMUP_RATIO}, warmdown=${WARMDOWN_RATIO}, final_lr_frac=${FINAL_LR_FRAC}"
echo "Weight Decay:               ${WEIGHT_DECAY}"
echo "Max Steps:                  ${MAX_STEPS}"
echo "Max Scheduling Steps:       ${MAX_SCHEDULING_STEPS}"
echo "Val Check Interval:         ${VAL_CHECK_INTERVAL}"
echo "Val Steps:                  ${VAL_STEPS}"
echo "Limit Val Batches:          ${LIMIT_VAL_BATCHES}"
echo "Post-train ckpt sweep:      ${RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN} (max=${SUDOKU_CKPT_SWEEP_MAX_CKPTS}, samples=${SUDOKU_CKPT_SWEEP_NUM_SAMPLES}, prompt=${SUDOKU_CKPT_SWEEP_PROMPT_STYLE})"
echo ""
echo "================================================================================"
echo "[开始训练]"
echo "================================================================================"
echo ""

set -e

# v3: 注意以下额外 flag (相对 v2):
#   --mcmc_num_steps ${MCMC_NUM_STEPS}                  [P2]
#   --randomize_mcmc_num_steps ${RANDOMIZE_MCMC_NUM_STEPS}        [P2]
#   --sudoku_ratio_schedule …           [P3+P4]
#   --sudoku_difficulty_schedule …      [P5]
#   --sudoku_blank_loss_weight …        [P1]
torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/train.py \
--run_name ${RUN_NAME}_${current_time} \
--checkpoint_dir "${EXP_CKPT_DIR}" \
--wandb_save_dir "${EXP_WANDB_DIR}" \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
--pretokenize_dataset \
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
--denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps ${MCMC_NUM_STEPS} \
--randomize_mcmc_num_steps ${RANDOMIZE_MCMC_NUM_STEPS} \
--context_length ${CONTEXT_LENGTH} \
--gpus "-1" \
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
--weight_decay ${WEIGHT_DECAY} \
--beta1 ${BETA1} \
--beta2 ${BETA2} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--dataset_name "sudoku_mixed" \
--sudoku_ratio ${SUDOKU_RATIO} \
--sudoku_ratio_schedule ${SUDOKU_RATIO_SCHEDULE} \
--sudoku_phase1_steps ${SUDOKU_PHASE1_STEPS} \
--sudoku_phase2_steps ${SUDOKU_PHASE2_STEPS} \
--sudoku_warmup_ratio ${SUDOKU_WARMUP_RATIO} \
--sudoku_focus_ratio ${SUDOKU_FOCUS_RATIO} \
--sudoku_consolidate_ratio ${SUDOKU_CONSOLIDATE_RATIO} \
--sudoku_guard_sft_baseline ${SUDOKU_GUARD_SFT_BASELINE} \
--sudoku_guard_tolerance ${SUDOKU_GUARD_TOLERANCE} \
--sudoku_ratio_floor ${SUDOKU_RATIO_FLOOR} \
--sudoku_ratio_ceil ${SUDOKU_RATIO_CEIL} \
--sudoku_ratio_step ${SUDOKU_RATIO_STEP} \
--sudoku_difficulty_schedule ${SUDOKU_DIFFICULTY_SCHEDULE} \
--sudoku_warmup_bucket_weights ${SUDOKU_WARMUP_BUCKET_WEIGHTS} \
--sudoku_focus_bucket_weights ${SUDOKU_FOCUS_BUCKET_WEIGHTS} \
--sudoku_consolidate_bucket_weights ${SUDOKU_CONSOLIDATE_BUCKET_WEIGHTS} \
--dataset_seed ${DATASET_SEED} \
--sudoku_blank_loss_weight ${SUDOKU_BLANK_LOSS_WEIGHT} \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--val_steps ${VAL_STEPS} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.1 \
--wandb_project 'nlp_sudoku_sft_v3' \
--log_model_archi \
--checkpoint_monitor_string valid_loss_balanced \
--set_matmul_precision "medium" \
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps ${SAVE_PERIODIC_STEPS} \
--float_precision "bf16-mixed" \
--finetuning_model_ckpt ${PRETRAIN_CKPT} \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_DIR}/sft_train" "sudoku_sft_v3_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ Sudoku SFT V3 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ Sudoku SFT V3 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"

if [ "${RUN_SUDOKU_CKPT_SWEEP_AFTER_TRAIN}" = "1" ]; then
    echo ""
    echo "开始按 Sudoku 生成指标重排 checkpoint 候选..."
    python /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/select_sudoku_sft_ckpt.py \
        "${EXP_DIR}/sft_train" \
        --ckpt_dir "${EXP_CKPT_DIR}" \
        --max_ckpts "${SUDOKU_CKPT_SWEEP_MAX_CKPTS}" \
        --num_samples "${SUDOKU_CKPT_SWEEP_NUM_SAMPLES}" \
        --prompt_style "${SUDOKU_CKPT_SWEEP_PROMPT_STYLE}" \
        2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}" || true
fi

BEST_CKPT="$(ls -t "${EXP_CKPT_DIR}"/s=step=*.ckpt 2>/dev/null | head -1 || true)"
if [ "${RUN_SUDOKU_EVAL_AFTER_TRAIN:-0}" = "1" ] && [ -n "${BEST_CKPT}" ]; then
    echo ""
    echo "开始自动 Sudoku sample eval: ${BEST_CKPT}"
    python /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/eval_sudoku_samples.py \
        --checkpoint "${BEST_CKPT}" \
        --num_samples "${SUDOKU_EVAL_NUM_SAMPLES:-512}" \
        --run_dir "${EXP_DIR}/sft_train" \
        2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"
elif [ -n "${BEST_CKPT}" ]; then
    echo ""
    echo "推荐下一步评估 RL 起点候选:"
    echo "python /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/eval_sudoku_samples.py --checkpoint \"${BEST_CKPT}\" --num_samples 1024 --run_dir \"${EXP_DIR}/sft_train\""
fi

################################################################################
# Changelog (v2 → v3)
################################################################################
# - Adaptive `sudoku_ratio` (`three_phase_with_guard`):
#     warm / focus / consolidate cutoffs are configurable and scale with MAX_STEPS;
#     PI guard pulls back when valid_loss_sft drifts > 0.03 over baseline 1.121. (P3+P4)
# - Difficulty-aware resampling of RRN-train: hard bucket defaults to 0.60 in focus. (P5)
# - Blank-position loss weighting K=${SUDOKU_BLANK_LOSS_WEIGHT}. (P1)
# - MCMC stays at 2/randomize=0 by default for OOM safety; override env vars for smoke tests. (P2)
# - KD anchor: deferred, NOT implemented in v3 (retention guard carries SFT preservation).
# - New `valid_loss_balanced = 0.5*sudoku + 0.5*sft` for ratio-fair tracking. (P3)
# - Prompt templates 5 → 20, response 3 → 8, new flat (spaceless) puzzle format
#   with 70/30 grid/flat mix. (P8 + user req, must-have)
# - Eval script: lenient parse, by-givens bucket, SIGINT-safe finalize, error dumps;
#   new parse_eval_log.py.
# - SAVE_TOP_K defaults to 5 and periodic ckpts default to 500 steps for RL-start selection.
################################################################################
