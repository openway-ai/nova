#!/bin/bash

################################################################################
# EBT d6 训练脚本 - Muon+AdamW 混合优化器 + 滑动窗口注意力 (SWA)
#
# 日期: 2026-04-03
# 策略:
#   1. 优化器: Muon+AdamW (复用 nanochat/optim.py)
#      - Transformer 矩阵参数 → Muon (LR=0.02)
#      - Embedding/Scalar/Alpha → AdamW (beta1=0.8, beta2=0.95)
#   2. Weight Decay: 0.2 + 动态衰减到 0 (对齐 base_train.py)
#   3. LR 调度: Linear Warmdown (后50%衰减到0, 对齐 base_train.py)
#   4. EBT 类型: nanochat_time_embed (支持 SWA)
#   5. SWA 模式: SSSL (每4层一个全上下文层, 其余为半上下文滑动窗口)
#
# 模型规格 (nanochat_depth_scaling(6)):
#   - 6 layers, 3 heads, dim=384
#   - 参数量: ~33M
#
# 参考:
#   - 基于: nova/ebt/runs/run_ebt_muon_adamw_0324.sh (d26 版本)
#   - SWA 设计: nova/ebt/SLIDING_WINDOW_ATTENTION.md
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-d6-swa
#SBATCH --output=logs/slurm/nlp/ebt-d6-swa_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-d6-swa-muon-adamw-0403"
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="d6"

### 环境变量 ###
export NANOCHAT_BASE_DIR="/root/.cache/nanochat"
source /root/nova/.venv/ebt/bin/activate

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# WandB 配置
export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# EBT 核心超参数 (严格遵循官方建议)
################################################################################
# 来源: https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py
#
# 官方说明:
# "keeping mcmc_step_size_lr_multiplier = 3x mcmc_step_size is safe and
#  what works well so the most important and arguably only really necessary
#  to tune hparam is mcmc_step_size"
#
# d6 参数推导:
#   PEAK_LR = 0.0012 (d6 小模型推荐)
#   官方要求: Alpha 实际 LR = PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER ∈ [0.3, 0.9]
#   → 0.3 ≤ 0.0012 × 3 × MCMC_STEP_SIZE ≤ 0.9
#   → 83 ≤ MCMC_STEP_SIZE ≤ 250
#   → 选 MCMC_STEP_SIZE=150, LR_MULT=450, alpha_lr=0.54 ✓
################################################################################

# EBT 特定参数
MCMC_STEP_SIZE=150.0
MCMC_STEP_SIZE_LR_MULTIPLIER=450     # 3 × 150 (官方推荐比例)
MCMC_NUM_STEPS=2                      # 官方建议: 2 is very safe
EBT_TYPE="nanochat_time_embed"        # 支持 SWA 的 EBT 类型
WINDOW_PATTERN="SSSL"                 # SWA 模式: 3层半上下文 + 1层全上下文, 循环铺开
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置与训练步数自动计算 - 针对 d6 模型优化
################################################################################

# 1. 硬件与 Batch 基础配置
# d6 模型参数量 (~33M) 远小于 d26 (~1031M), 可用更大 batch size
# 同时使用更长上下文 (1024) 以发挥 SWA 优势 (SWA 降低长上下文注意力开销)
NUM_GPUS=4
DEVICE_BATCH_SIZE=16
GRAD_ACCUM=4
CONTEXT_LENGTH=1024

# 2. 计算每步的有效 Token 数 (Tokens per step)
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# 当前配置: 4 × 16 × 4 × 1024 = 262,144 tokens/step

# 3. 设置目标总 Token 数
# d6 小模型, 用于 SWA 功能验证和消融实验
# 目标 ~2B tokens (约 3814 steps)
TARGET_TOTAL_TOKENS=2000000000  # 2B tokens

# 4. 自动计算总 Steps 数 (向下取整)
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置 (基于模型规模推荐)
################################################################################
# d6 模型规模分析:
#   - 6 layers, 3 heads, dim=384 (nanochat_depth_scaling(6))
#   - 参数量: ~33M
#
# 对比 model_sizes 中最接近的规格:
#   xxs (6层, 384dim):   LR = 0.0012  ← d6 使用相同 dim, 推荐相同 LR
#
# Alpha 实际 LR = PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER
#               = 0.0012 × 450 = 0.54 ✓ (官方推荐范围 0.3-0.9)
#
# adamw_dmodel_lr_scaling 对 d6 (dim=384):
#   scale = (384/768)^-0.5 = √2 ≈ 1.414
#   实际 embedding LR = 0.3 × 1.414 = 0.424
#   实际 scalar LR    = 0.04 × 1.414 = 0.057
################################################################################

PEAK_LR=0.0012

WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置 (Muon+AdamW 模式, 对齐 NanoChat base_train.py)
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
# 优化选项配置 (Muon+AdamW 全套优化, 对齐 NanoChat)
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################

COMPILE_FLAGS="--compile_model --compile_mode full"

################################################################################
# WandB 配置 (训练参数)
################################################################################

WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level parameters"
# WANDB_FLAGS="--wandb_watch --wandb_watch_level all"

################################################################################
# 日志配置
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_${MODEL_NAME}_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_swa-${WINDOW_PATTERN}_muon-adamw_gpu${NUM_GPUS}.log"
mkdir -p logs

# 定义颜色
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

print_header "EBT d6 SWA Muon+AdamW 训练配置"

echo ""
echo -e "${CYAN}▶ 优化器策略 (对齐 NanoChat)${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} Muon+AdamW 混合优化器 (nanochat/optim.py)"
echo "  ${GREEN}✓${NC} Transformer 矩阵参数 → Muon (LR=0.02)"
echo "  ${GREEN}✓${NC} Embedding/Scalar/Alpha → AdamW"
echo "  ${GREEN}✓${NC} Adam Beta: (${BOLD}${BETA1}, ${BETA2}${NC}) (对齐 NanoChat)"
echo "  ${GREEN}✓${NC} Weight Decay: ${BOLD}${WEIGHT_DECAY}${NC} + 动态衰减到 0"
echo "  ${GREEN}✓${NC} LR 调度: Linear Warmdown (后50%衰减到0)"

echo ""
echo -e "${CYAN}▶ SWA 配置${NC}"
print_separator "─" 60
print_kv "EBT Type" "${EBT_TYPE}"
print_kv "Window Pattern" "${WINDOW_PATTERN} (每4层1个全上下文层)"
print_kv "Context Length" "${CONTEXT_LENGTH} (长上下文以发挥SWA优势)"

echo ""
echo -e "${CYAN}▶ EBT 核心参数 (官方推荐)${NC}"
print_separator "─" 60
print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)"
print_kv "MCMC Num Steps" "${MCMC_NUM_STEPS}"
alpha_lr=$(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
print_kv "Alpha Effective LR" "${alpha_lr} (官方推荐范围 0.3-0.9)"

echo ""
echo -e "${CYAN}▶ 模型配置${NC}"
print_separator "─" 60
print_kv "Model Size" "${MODEL_SIZE}"
print_kv "架构" "6 layers, dim=384, 3 heads (~33M params)"

echo ""
echo -e "${CYAN}▶ Batch 配置${NC}"
print_separator "─" 60
print_kv "Device Batch Size" "${DEVICE_BATCH_SIZE}"
print_kv "Gradient Accumulation" "${GRAD_ACCUM}"
print_kv "Num GPUs" "${NUM_GPUS}"
print_kv "Effective Batch Size" "${EFFECTIVE_BATCH_SIZE} tokens/step"

echo ""
echo -e "${CYAN}▶ 训练进度${NC}"
print_separator "─" 60
print_kv "Target Tokens" "2B"
print_kv "Max Steps" "${MAX_STEPS} (自适应计算)"
total_tokens_b=$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")
print_kv "Actual Total Tokens" "~${total_tokens_b}B"
print_kv "Val Check Interval" "${VAL_CHECK_INTERVAL}"

echo ""
echo -e "${YELLOW}⚠ 重要说明${NC}"
echo "  1. EBT 类型改为 nanochat_time_embed (支持 SWA, 替代 d26 的 time_embed)"
echo "  2. SWA 模式 SSSL: 3层滑动窗口(半上下文) + 1层全上下文, 循环铺满6层 → SSSL SS"
echo "  3. MCMC 参数按 d6 LR 重新推导 (alpha_lr=0.54, 在官方推荐范围内)"
echo "  4. 监控关键指标: train_loss, Alpha_MCMC, Global_LR"

echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d6 SWA Muon+AdamW 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 模型: d6 (6 layers, dim=384, 3 heads, ~33M params)
# SWA:  nanochat_time_embed + window_pattern=${WINDOW_PATTERN}
#
# 优化器策略 (对齐 NanoChat base_train.py):
#   1. Muon+AdamW 混合优化器 (nanochat/optim.py)
#   2. Adam Beta: (${BETA1}, ${BETA2})
#   3. Weight Decay: ${WEIGHT_DECAY} + 动态衰减到 0
#   4. LR 调度: Linear Warmdown (后50%衰减到0)
#   5. Alpha LR = PEAK_LR × MCMC_LR_MULT = ${PEAK_LR} × ${MCMC_STEP_SIZE_LR_MULTIPLIER} = $(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
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
EBT Type:                 ${EBT_TYPE}
Window Pattern:           ${WINDOW_PATTERN}
MCMC Step Size:           ${MCMC_STEP_SIZE}
MCMC LR Multiplier:       ${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)
Alpha Effective LR:       $(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
MCMC Num Steps:           ${MCMC_NUM_STEPS}

================================================================================
[训练配置]
================================================================================
Model Size:               ${MODEL_SIZE} (6L, d=384, h=3)
Context Length:           ${CONTEXT_LENGTH}
Device Batch Size:        ${DEVICE_BATCH_SIZE}
Gradient Accumulation:    ${GRAD_ACCUM}
Effective Batch Size:     ${EFFECTIVE_BATCH_SIZE} tokens/step

Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Max Steps:                ${MAX_STEPS}

Option Flags:             ${OPTION_FLAGS}
Compile Flags:            ${COMPILE_FLAGS}

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
        echo -e "${YELLOW}⚠ 检测到 NaN/Inf - 可能需要进一步降低 MCMC_STEP_SIZE${NC}"
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

实时监控命令 (在另一个终端运行):
  tail -f ${LOG_FILE} | grep -E "(train_loss|Alpha_MCMC|Global_LR)"

关键指标检查:
  1. train_loss: 应该平稳下降,避免突然飙升
  2. Alpha_MCMC: MCMC步长应该稳定 (目标范围 100-200)
  3. Global_LR: 学习率应该平滑衰减

SWA 相关调优:
  - 如果效果不佳,尝试调整 WINDOW_PATTERN (如 "SSL" 或 "L" 作为 baseline 对比)
  - SWA 层分配: 6层模型 + "SSSL" → SSSL SS (前4层SSSL, 后2层SS)
  - 对比实验: 使用 --window_pattern L 运行 baseline 对比 SWA 效果

异常诊断:
  - loss 突然飙升: 降低 MCMC_STEP_SIZE (如 100 或 80)
  - Alpha_MCMC 震荡: 降低 MCMC_STEP_SIZE_LR_MULTIPLIER
  - NaN 出现: 降低 PEAK_LR 或 MCMC_STEP_SIZE

MONITOR_HELP

echo ""
