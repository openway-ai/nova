#!/bin/bash

################################################################################
# EBT d26 训练脚本 - Muon+AdamW 混合优化器 - 本地单节点运行版
#
# 策略 (对齐 run_ebt_2node_8gpu.sh):
#   1. 优化器: Muon+AdamW (复用 nanochat/optim.py)
#      - Transformer 矩阵参数 → Muon (LR=0.02)
#      - Embedding/Scalar/Alpha → AdamW (beta1=0.8, beta2=0.95)
#   2. Weight Decay: 0.2 + 动态衰减到 0
#   3. LR 调度: Linear Warmdown (后50%衰减到0)
#   4. EBT 特有参数保持官方推荐 (MCMC step_size, alpha LR 等)
#
# 用法:
#   bash run_ebt_muon_adamw_c2048.sh
#   NUM_GPUS=4 bash run_ebt_muon_adamw_c2048.sh  # 指定 GPU 数量
#
# 注意:
#   本脚本为本地单节点运行版本，使用 torchrun --standalone 模式
#   无需 SLURM 或 rjob 平台支持
#
################################################################################

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"

### 路径配置 ###
NOVA_HOME="/mnt/shared-storage-user/luyudong/nova"
NANOCHAT_HOME="/mnt/shared-storage-user/luyudong/nanochat"
TRAIN_SCRIPT="${NOVA_HOME}/nova/ebt/train.py"

# 进入工作目录
cd "${NOVA_HOME}"

### 基础配置 ###
RUN_PREFIX="local-1node-bf16mixed"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="${NANOCHAT_HOME}"
export NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"

################################################################################
# NCCL 配置 (本地单节点)
# 本地单节点通常无 InfiniBand，禁用 IB 避免警告
################################################################################

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

################################################################################
# GPU 配置
# 本地单节点运行，默认使用本机所有可见 GPU
################################################################################

NUM_GPUS=${NUM_GPUS:-8}

echo "=== 本地单节点配置 ==="
echo "NUM_GPUS: ${NUM_GPUS}"

################################################################################
# EBT 核心超参数 (严格遵循官方建议, 对齐 run_ebt_2node_8gpu.sh)
################################################################################

MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"               # 官方建议: can almost always stay
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false
USE_SDPA_ATTENTION=false             # 是否启用 SDPA 注意力，设为 true 启用，目前不可启用

################################################################################
# Batch 配置与训练步数自动计算
################################################################################

DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=2048

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

TARGET_TOTAL_TOKENS=7340032000
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置
################################################################################

PEAK_LR=0.00025
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置 (Muon+AdamW 模式)
################################################################################

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
SAVE_TOP_K=2

################################################################################
# 优化选项
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.05 \
--optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling \
--muon_momentum_warmup_steps 300"

################################################################################
# torch.compile 配置
# 对齐 run_ebt_2node_8gpu.sh: 默认不启用 compile
################################################################################

# COMPILE_FLAGS=""
COMPILE_FLAGS="--compile_model --compile_mode transformer_only"
# COMPILE_FLAGS="--compile_model --compile_mode full"

################################################################################
# SDPA 注意力配置
################################################################################

SDPA_FLAGS=""
if [[ "${USE_SDPA_ATTENTION}" == "true" ]]; then
    SDPA_FLAGS="--use_sdpa_attention"
fi

################################################################################
# WandB 训练标志
################################################################################

WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"

################################################################################
# 日志配置 - 自动命名 + 按日期分层
################################################################################

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
CONFIG_TAG="${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}_${NUM_GPUS}gpus"
export RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}"

LOG_DIR="${NOVA_HOME}/logs_base_train/${DATE_DIR}"
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

print_header "EBT d26 Muon+AdamW 训练配置 (本地单节点版)"

echo ""
echo -e "${CYAN}▶ 运行模式${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} 本地单节点运行 (torchrun --standalone)"
echo "  ${GREEN}✓${NC} GPU 数量: ${BOLD}${NUM_GPUS}${NC}"

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
echo -e "${CYAN}▶ EBT 核心参数 (官方推荐)${NC}"
print_separator "─" 60
print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)"
print_kv "MCMC Num Steps" "${MCMC_NUM_STEPS}"
print_kv "EBT Type" "${EBT_TYPE}"
print_kv "Normalize Init Condition" "${NORMALIZE_INITIAL_CONDITION}"
print_kv "Denoising Init Condition" "${DENOISING_INITIAL_CONDITION}"
print_kv "Use SDPA Attention" "${USE_SDPA_ATTENTION}"

echo ""
echo -e "${CYAN}▶ 模型配置${NC}"
print_separator "─" 60
print_kv "Model Size" "${MODEL_SIZE}"
print_kv "Context Length" "${CONTEXT_LENGTH}"
print_kv "估计参数量" "~1031M (1.03B) - 介于large和xl之间"

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
print_kv "Warmup Steps" "${WARM_UP_STEPS}"
print_kv "Min LR Scale" "${MIN_LR_SCALE}"
final_lr=$(awk "BEGIN {printf \"%.8f\", ${PEAK_LR}/${MIN_LR_SCALE}}")
print_kv "Final LR" "${final_lr}"

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
print_kv "Run Name" "${RUN_NAME}"
print_kv "Log File" "${LOG_FILE}"
print_kv "WandB Mode" "${WANDB_MODE}"
print_kv "Compile" "${COMPILE_FLAGS:-disabled}"

echo ""
echo -e "${YELLOW}⚠ 重要说明${NC}"
echo "  1. EBT 核心参数 (MCMC) 保持官方推荐值不变"
echo "  2. 优化器/LR调度/Weight Decay 对齐 NanoChat base_train.py"
echo "  3. Alpha LR 仍由 MCMC_STEP_SIZE_LR_MULTIPLIER × PEAK_LR 控制 (EBT 特有)"
echo "  4. 监控关键指标: train_loss, Alpha_MCMC, Global_LR"
echo "  5. 参数设定对齐 run_ebt_2node_8gpu.sh"

echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 Muon+AdamW 训练日志 (本地单节点版)
################################################################################
#
# Run Name:        ${RUN_NAME}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 优化器策略 (对齐 NanoChat base_train.py):
#   1. Muon+AdamW 混合优化器 (nanochat/optim.py)
#   2. Adam Beta: (${BETA1}, ${BETA2})
#   3. Weight Decay: ${WEIGHT_DECAY} + 动态衰减到 0
#   4. LR 调度: Linear Warmdown (后50%衰减到0)
#   5. EBT 特有: Alpha LR = PEAK_LR × MCMC_LR_MULT (保持不变)
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
[EBT 核心参数 - 官方推荐]
================================================================================
MCMC Step Size:           ${MCMC_STEP_SIZE}
MCMC LR Multiplier:       ${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)
Alpha Effective LR:       $(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
MCMC Num Steps:           ${MCMC_NUM_STEPS}
EBT Type:                 ${EBT_TYPE}

================================================================================
[训练配置]
================================================================================
Model Size:               ${MODEL_SIZE}
Context Length:            ${CONTEXT_LENGTH}
Device Batch Size:         ${DEVICE_BATCH_SIZE}
Gradient Accumulation:     ${GRAD_ACCUM}
Effective Batch Size:      ${EFFECTIVE_BATCH_SIZE} tokens/step

Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Warmup Steps:             ${WARM_UP_STEPS}
Max Steps:                ${MAX_STEPS}
Total Tokens:             ${total_tokens_b}

Option Flags:             ${OPTION_FLAGS:-"None (Baseline)"}
Compile Flags:            ${COMPILE_FLAGS:-"None"}
SDPA Attention:           ${USE_SDPA_ATTENTION}

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

torchrun --standalone --nproc_per_node=${NUM_GPUS} "${TRAIN_SCRIPT}" \
  --run_name "${RUN_NAME}" \
  --modality "NLP" \
  --model_name "${MODEL_NAME}" \
  --model_size "${MODEL_SIZE}" \
  \
  --pretokenize_dataset \
  \
  --normalize_initial_condition \
  --ebt_type "${EBT_TYPE}" \
  --denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
  --mcmc_step_size_learnable \
  --mcmc_step_size "${MCMC_STEP_SIZE}" \
  --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
  --mcmc_num_steps "${MCMC_NUM_STEPS}" \
  \
  --context_length "${CONTEXT_LENGTH}" \
  \
  --gpus "-1" \
  \
  --peak_learning_rate "${PEAK_LR}" \
  --batch_size_per_device "${DEVICE_BATCH_SIZE}" \
  --accumulate_grad_batches "${GRAD_ACCUM}" \
  --gradient_clip_val "${GRADIENT_CLIP_VAL}" \
  \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --min_lr_scale "${MIN_LR_SCALE}" \
  --max_steps "${MAX_STEPS}" \
  --max_scheduling_steps "${MAX_SCHEDULING_STEPS}" \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --warm_up_base_lr_divider "${WARM_UP_BASE_LR_DIVIDER}" \
  \
  --dataset_name "nanochat" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_sanity 1 \
  --validation_split_pct 0.0027 \
  \
  --wandb_project 'nlp_pretrain' \
  --log_model_archi \
  --set_matmul_precision "medium" \
  --float_precision "bf16-true" \
  --manual_gc_collect_every_n_steps 100 \
  --save_top_k_ckpts "${SAVE_TOP_K}" \
  --save_periodic_steps 1000 \
  ${WANDB_FLAGS} \
  ${OPTION_FLAGS} \
  ${COMPILE_FLAGS} \
  ${SDPA_FLAGS}

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

    # 检查常见问题
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

实时监控命令 (在另一个终端运行):
  tail -f ${LOG_FILE} | grep -E "(train_loss|Alpha_MCMC|Global_LR)"

关键指标检查:
  1. train_loss: 应该平稳下降,避免突然飙升
  2. Alpha_MCMC: MCMC步长应该稳定在合理范围 (如 400-600)
  3. Global_LR: 学习率应该平滑衰减

异常诊断:
  - 如果 loss 突然飙升: 检查 Alpha_MCMC 是否异常
  - 如果 Alpha_MCMC 震荡剧烈: 考虑进一步降低 MCMC_STEP_SIZE_LR_MULTIPLIER
  - 如果 NaN 出现: 降低 PEAK_LR 或启用 gradient checkpointing
  - 如果 OOM: 减小 DEVICE_BATCH_SIZE 或启用 --use_sdpa_attention

对比基准 (之前失败的配置):
  - 之前: Alpha LR = 1.2, 多次 loss 暴涨
  - 现在: Alpha LR = 0.375, 预期稳定

MONITOR_HELP

echo ""