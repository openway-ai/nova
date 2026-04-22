#!/bin/bash

################################################################################
# EBT d26 训练脚本 - Muon+AdamW 混合优化器 (对齐 NanoChat)
#
# 策略:
#   1. 优化器: Muon+AdamW (复用 nanochat/optim.py)
#      - Transformer 矩阵参数 → Muon (LR=0.02)
#      - Embedding/Scalar/Alpha → AdamW (beta1=0.8, beta2=0.95)
#   2. Weight Decay: 0.2 + 动态衰减到 0
#   3. LR 调度: Linear Warmdown (后50%衰减到0)
#   4. EBT 特有参数保持官方推荐 (MCMC step_size, alpha LR 等)
#
################################################################################

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"

### 基础配置 ###
export MODEL_NAME="ebt-ctx2048-bf16mixed"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="/mnt/shared-storage-user/luyudong/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

# EBT 特定参数
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"               # 官方建议: can almost always stay
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置与训练步数自动计算
################################################################################

NUM_GPUS=8
DEVICE_BATCH_SIZE=2
GRAD_ACCUM=32
CONTEXT_LENGTH=1024

# 每步有效 Token 数
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# = 8 × 2 × 32 × 1024 = 524,288 tokens/step

# 目标总 Token 数 (对齐 NanoChat speedrun.sh --depth=26: 7.34B tokens)
TARGET_TOTAL_TOKENS=7340032000

# 自动计算总 Steps 数 (向下取整)
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置
################################################################################

# d26 (~1031M) 使用 LR=0.00025 (来源: nova/ebt/utils.py 模型规模推荐)
# Alpha 实际 LR = 0.00025 × 1500 = 0.375 (官方推荐范围 0.3-0.9)
PEAK_LR=0.00025

# Warmup/Warmdown: 由 --linear_warmdown 的 --warmup_ratio / --warmdown_ratio 控制
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50  # fallback: 仅在不使用 --linear_warmdown 时生效

################################################################################
# 优化器配置 (Muon+AdamW)
################################################################################

WEIGHT_DECAY=0.2              # 配合 --dynamic_wd 线性衰减到 0
BETA1=0.8                     # AdamW 部分 (embedding/scalar/alpha)
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8
SAVE_TOP_K=2

################################################################################
# 优化选项配置
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################

COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

################################################################################
# WandB 配置
################################################################################

#   不传 --wandb_watch              → 只记录 scalar metrics (默认)
#   --wandb_watch                    → 记录 parameters only
#   --wandb_watch --wandb_watch_level all → 记录 gradients+parameters+activations (debug 用)
#   --disable_wandb                  → 完全关闭 wandb
WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level parameters"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level all"

################################################################################
# 日志配置
################################################################################

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
CONFIG_TAG="${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}"

RUN_PREFIX="${MODEL_NAME}"
export RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}"

LOG_DIR="logs_base_train/${DATE_DIR}"
current_time=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# 日志函数
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

print_header "EBT d26 Muon+AdamW 训练配置"

echo ""
echo -e "${CYAN}▶ 优化器策略${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} Muon+AdamW 混合优化器"
echo "  ${GREEN}✓${NC} Transformer 矩阵参数 → Muon (LR=0.02)"
echo "  ${GREEN}✓${NC} Embedding/Scalar/Alpha → AdamW"
echo "  ${GREEN}✓${NC} Adam Beta: (${BOLD}${BETA1}, ${BETA2}${NC})"
echo "  ${GREEN}✓${NC} Weight Decay: ${BOLD}${WEIGHT_DECAY}${NC} + 动态衰减到 0"
echo "  ${GREEN}✓${NC} LR 调度: Linear Warmdown (后50%衰减到0)"

echo ""
echo -e "${CYAN}▶ EBT 核心参数${NC}"
print_separator "─" 60
print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)"
print_kv "MCMC Num Steps" "${MCMC_NUM_STEPS}"
print_kv "EBT Type" "${EBT_TYPE}"
print_kv "Normalize Init Condition" "${NORMALIZE_INITIAL_CONDITION}"
print_kv "Denoising Init Condition" "${DENOISING_INITIAL_CONDITION}"

echo ""
echo -e "${CYAN}▶ 模型配置${NC}"
print_separator "─" 60
print_kv "Model Size" "${MODEL_SIZE}"
print_kv "Context Length" "${CONTEXT_LENGTH}"
print_kv "参数量" "~1031M (1.03B)"

echo ""
echo -e "${CYAN}▶ Batch 配置${NC}"
print_separator "─" 60
print_kv "Device Batch Size" "${DEVICE_BATCH_SIZE}"
print_kv "Gradient Accumulation" "${GRAD_ACCUM}"
print_kv "Num GPUs" "${NUM_GPUS}"
print_kv "Effective Batch Size" "${EFFECTIVE_BATCH_SIZE} tokens/step"

echo ""
echo -e "${CYAN}▶ 学习率配置${NC}"
print_separator "─" 60
print_kv "Peak LR (Base)" "${PEAK_LR}"
alpha_lr=$(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
print_kv "Alpha Effective LR" "${BOLD}${alpha_lr}${NC} (=${PEAK_LR}×${MCMC_STEP_SIZE_LR_MULTIPLIER})"
print_kv "Warmup Ratio" "0.0"
print_kv "Warmdown Ratio" "0.5 (后50%线性衰减到0)"

echo ""
echo -e "${CYAN}▶ 优化器配置${NC}"
print_separator "─" 60
print_kv "Weight Decay" "${WEIGHT_DECAY}"
print_kv "Adam Beta1" "${BETA1}"
print_kv "Adam Beta2" "${BETA2}"
print_kv "Gradient Clip" "${GRADIENT_CLIP_VAL}"

echo ""
echo -e "${CYAN}▶ 训练进度${NC}"
print_separator "─" 60
print_kv "Target Tokens" "7.34B"
print_kv "Max Steps" "${MAX_STEPS} (自适应计算)"
print_kv "Max Sched Steps" "${MAX_SCHEDULING_STEPS}"
total_tokens_b=$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")
print_kv "Actual Total Tokens" "~${total_tokens_b}B"
print_kv "Val Check Interval" "${VAL_CHECK_INTERVAL}"

echo ""
echo -e "${CYAN}▶ 输出配置${NC}"
print_separator "─" 60
print_kv "Run Name" "${RUN_NAME}_${current_time}"
print_kv "Log File" "${LOG_FILE}"
print_kv "WandB Mode" "${WANDB_MODE}"

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 Muon+AdamW 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 优化器策略:
#   1. Muon+AdamW 混合优化器
#   2. Adam Beta: (${BETA1}, ${BETA2})
#   3. Weight Decay: ${WEIGHT_DECAY} + 动态衰减到 0
#   4. LR 调度: Linear Warmdown (后50%衰减到0)
#   5. Alpha LR = ${PEAK_LR} × ${MCMC_STEP_SIZE_LR_MULTIPLIER} = ${alpha_lr}
#
################################################################################

================================================================================
[SYSTEM INFO]
================================================================================
Hostname:         $(hostname)
User:             $(whoami)
Python:           $(python3 --version 2>&1)
PyTorch:          $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA Available:   $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
GPU Count:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "N/A")
GPU Model:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

================================================================================
[EBT 核心参数]
================================================================================
MCMC Step Size:           ${MCMC_STEP_SIZE}
MCMC LR Multiplier:       ${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)
Alpha Effective LR:       ${alpha_lr}
MCMC Num Steps:           ${MCMC_NUM_STEPS}
EBT Type:                 ${EBT_TYPE}

================================================================================
[训练配置]
================================================================================
Model Size:               ${MODEL_SIZE}
Context Length:            ${CONTEXT_LENGTH}
Device Batch Size:        ${DEVICE_BATCH_SIZE}
Gradient Accumulation:    ${GRAD_ACCUM}
Effective Batch Size:     ${EFFECTIVE_BATCH_SIZE} tokens/step

Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Max Steps:                ${MAX_STEPS}
Total Tokens:             ${total_tokens_b}B

Option Flags:             ${OPTION_FLAGS:-"None (Baseline)"}
Compile Flags:            ${COMPILE_FLAGS:-"None"}

================================================================================
[训练输出]
================================================================================

LOG_HEADER

################################################################################
# 自动重定向所有输出到日志文件
################################################################################

exec > >(tee -a "${LOG_FILE}") 2>&1

################################################################################
# 启动训练
################################################################################

echo ""
print_header "开始训练"
echo ""

set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/luyudong/nova/nova/ebt/train.py \
--run_name ${RUN_NAME}_${current_time} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
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
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps 1000 \
--float_precision bf16-mixed \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

################################################################################
# 训练结束处理
################################################################################

echo ""
print_header "训练结束"

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "${RED}✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)${NC}"

    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
        echo -e "${YELLOW}⚠ 检测到 OOM - 建议减小 DEVICE_BATCH_SIZE${NC}"
    fi

    if grep -q "nan\|NaN\|inf\|Inf" "${LOG_FILE}" 2>/dev/null | head -5; then
        echo -e "${YELLOW}⚠ 检测到 NaN/Inf - 可能需要进一步降低学习率${NC}"
    fi
fi

echo ""
echo "日志文件: ${LOG_FILE}"
echo ""

################################################################################
# 监控建议
################################################################################

print_header "训练监控建议"
cat << MONITOR_HELP

实时监控命令:
  tail -f ${LOG_FILE} | grep -E "(train_loss|Alpha_MCMC|Global_LR)"

MONITOR_HELP

echo ""
