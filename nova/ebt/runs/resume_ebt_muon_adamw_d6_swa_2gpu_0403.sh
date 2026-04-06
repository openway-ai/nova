#!/bin/bash

################################################################################
# EBT d6 SWA 恢复训练脚本 - 2 GPU
#
# 日期: 2026-04-03
# 说明:
#   从 4-GPU 训练的 last.ckpt 恢复, 改用 2 GPU 继续训练
#   已消耗约 418 步 × 262,144 tokens/step ≈ 109.6M tokens
#   目标总量 2B tokens, 剩余 ~1.89B tokens 需用 2 GPU 完成
#
# 恢复点:
#   /root/nova_forked/nova/nova/ebt/logs/checkpoints/
#   ebt-d6-swa-muon-adamw-0403_20260403_154552_2026-04-03_15-54-51_/last.ckpt
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --job-name=ebt-d6-swa-resume
#SBATCH --output=logs/slurm/nlp/ebt-d6-swa-resume_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-d6-swa-muon-adamw-resume-2gpu-0403"
export MODEL_NAME="ebt"
export MODEL_SIZE="d6"

### 恢复检查点路径 ###
RESUME_CKPT="/root/nova_forked/nova/nova/ebt/logs/checkpoints/ebt-d6-swa-muon-adamw-0403_20260403_154552_2026-04-03_15-54-51_/last.ckpt"

### 环境变量 ###
export NANOCHAT_BASE_DIR="/root/.cache/nanochat"
source /root/nova/.venv/ebt/bin/activate

export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# EBT 核心超参数 (与原脚本完全一致)
################################################################################

MCMC_STEP_SIZE=150.0
MCMC_STEP_SIZE_LR_MULTIPLIER=450
MCMC_NUM_STEPS=2
EBT_TYPE="nanochat_time_embed"
WINDOW_PATTERN="SSSL"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置与步数计算
################################################################################

NUM_GPUS=2
DEVICE_BATCH_SIZE=16
GRAD_ACCUM=4
CONTEXT_LENGTH=1024

# 原始 4-GPU 配置
ORIG_NUM_GPUS=4
ORIG_EFFECTIVE_BATCH=$((ORIG_NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# 4 × 16 × 4 × 1024 = 262,144 tokens/step

# 已完成的步数和消耗的 tokens
STEPS_DONE=418
TOKENS_CONSUMED=$((STEPS_DONE * ORIG_EFFECTIVE_BATCH))
# 418 × 262,144 = 109,576,192 tokens

# 目标总量
TARGET_TOTAL_TOKENS=2000000000  # 2B

# 2-GPU 有效 batch size
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# 2 × 16 × 4 × 1024 = 131,072 tokens/step

# 剩余 tokens 和需要的步数
TOKENS_REMAINING=$((TARGET_TOTAL_TOKENS - TOKENS_CONSUMED))
REMAINING_STEPS=$((TOKENS_REMAINING / EFFECTIVE_BATCH_SIZE))

# Lightning 从 checkpoint 恢复后会从 STEPS_DONE 继续计数
# 因此 MAX_STEPS = 已完成步数 + 剩余步数
MAX_STEPS=$((STEPS_DONE + REMAINING_STEPS))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "断点续训步数信息:"
echo "  - 原始有效 Batch:    ${ORIG_EFFECTIVE_BATCH} tokens/step (4 GPU)"
echo "  - 新有效 Batch:      ${EFFECTIVE_BATCH_SIZE} tokens/step (2 GPU)"
echo "  - 已消耗 tokens:     ${TOKENS_CONSUMED}"
echo "  - 剩余 tokens:       ${TOKENS_REMAINING}"
echo "  - 已完成步数:        ${STEPS_DONE}"
echo "  - 剩余步数:          ${REMAINING_STEPS}"
echo "  - MAX_STEPS:         ${MAX_STEPS}"

################################################################################
# 学习率配置 (与原脚本一致)
################################################################################

PEAK_LR=0.0012
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置
################################################################################

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

VAL_CHECK_INTERVAL=500
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8

################################################################################
# 优化选项 (与原脚本一致)
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"

COMPILE_FLAGS="--compile_model --compile_mode full"

WANDB_FLAGS=""

################################################################################
# 日志配置
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_${MODEL_NAME}_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_swa-${WINDOW_PATTERN}_resume-2gpu.log"
mkdir -p logs

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

print_separator() {
    local char=${1:-"═"}
    local width=${2:-80}
    printf "${BLUE}"
    printf '%*s' "$width" | tr ' ' "$char"
    printf "${NC}\n"
}

print_header() {
    local title=$1
    echo ""
    print_separator "═"
    echo -e "${BOLD}${GREEN}  $title${NC}"
    print_separator "═"
}

print_kv() {
    local key=$1
    local value=$2
    local width=${3:-30}
    printf "  ${DIM}%-${width}s${NC} %s\n" "$key:" "$value"
}

################################################################################
# 显示配置
################################################################################

print_header "EBT d6 SWA 断点续训 (2 GPU)"

echo ""
echo -e "${CYAN}▶ 恢复配置${NC}"
print_separator "─" 60
print_kv "Resume Checkpoint" "${RESUME_CKPT}"
print_kv "Steps Done" "${STEPS_DONE}"
print_kv "Tokens Consumed" "~$(awk "BEGIN {printf \"%.1f\", ${TOKENS_CONSUMED}/1e6}")M"
print_kv "Tokens Remaining" "~$(awk "BEGIN {printf \"%.2f\", ${TOKENS_REMAINING}/1e9}")B"

echo ""
echo -e "${CYAN}▶ Batch 配置 (2 GPU)${NC}"
print_separator "─" 60
print_kv "Num GPUs" "${NUM_GPUS} (原 4 GPU)"
print_kv "Effective Batch" "${EFFECTIVE_BATCH_SIZE} tokens/step (原 ${ORIG_EFFECTIVE_BATCH})"
print_kv "Remaining Steps" "${REMAINING_STEPS}"
print_kv "MAX_STEPS" "${MAX_STEPS}"

echo ""
echo -e "${YELLOW}⚠ 注意事项${NC}"
echo "  1. Batch size 减半 (262,144 → 131,072 tokens/step)"
echo "  2. LR 调度从 checkpoint 恢复, 基于新 MAX_STEPS=${MAX_STEPS} 重新规划"
echo "  3. 优化器状态 (Muon/AdamW moments) 从 checkpoint 完整恢复"
echo "  4. 模型权重完整恢复, 训练无缝继续"

echo ""
read -p "按 Enter 开始恢复训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#              EBT d6 SWA 断点续训日志 (2 GPU)
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Resume From:     ${RESUME_CKPT}
# Steps Done:      ${STEPS_DONE}
# Tokens Done:     ${TOKENS_CONSUMED} (~109.6M)
# Tokens Remaining:${TOKENS_REMAINING} (~1.89B)
# MAX_STEPS:       ${MAX_STEPS}
#
################################################################################

LOG_HEADER

exec > >(tee -a "${LOG_FILE}") 2>&1

################################################################################
# 启动恢复训练
################################################################################

print_header "开始恢复训练"
echo ""

set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} /root/nova_forked/nova/nova/ebt/train.py \
--run_name ${RUN_NAME}_${current_time} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
--window_pattern ${WINDOW_PATTERN} \
--denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps ${MCMC_NUM_STEPS} \
\
--context_length ${CONTEXT_LENGTH} \
\
--gpus "-1" \
\
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
\
--weight_decay ${WEIGHT_DECAY} \
--beta1 ${BETA1} \
--beta2 ${BETA2} \
--min_lr_scale ${MIN_LR_SCALE} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--warm_up_steps ${WARM_UP_STEPS} \
--warm_up_base_lr_divider ${WARM_UP_BASE_LR_DIVIDER} \
\
--dataset_name "nanochat" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
\
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS} \
--resume_training_ckpt ${RESUME_CKPT}

TRAIN_EXIT_CODE=$?
set -e

################################################################################
# 训练结束处理
################################################################################

echo ""
print_header "训练结束"

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ 恢复训练成功完成${NC}"
else
    echo -e "${RED}✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)${NC}"

    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
        echo -e "${YELLOW}⚠ 检测到 OOM - 建议减小 DEVICE_BATCH_SIZE${NC}"
    fi
fi

echo ""
echo "日志文件: ${LOG_FILE}"
echo ""
