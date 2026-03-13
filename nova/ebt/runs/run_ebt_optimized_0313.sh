#!/bin/bash

################################################################################
# EBT d26 稳定训练脚本 - 基于官方推荐参数修复版
#
# 修复日期: 2026-03-13
# 修复内容:
#   1. 降低 MCMC LR Multiplier (2000 → 1500, 遵循官方3x建议)
#   2. 禁用所有不稳定的优化选项
#   3. 恢复保守的 Adam Beta 参数
#   4. 降低 Weight Decay 避免过度正则化
#   5. 使用标准 Cosine Annealing LR 调度
#
# 参考: https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-d26-stable
#SBATCH --output=logs/slurm/nlp/ebt-d26-stable_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-d26-stable"
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
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
# 关键要点:
# 1. mcmc_step_size 是唯一需要调优的超参数
# 2. mcmc_step_size_lr_multiplier 应该是 step_size 的 3 倍
# 3. 其他参数保持默认即可
################################################################################

# EBT 特定参数
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"               # 官方建议: can almost always stay
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

# 不使用的高级参数 (官方: not recommended for getting started)
# ebt_norm, ebt_act_func, dyt_alpha_init, mcmc_replay_buffer 等
# 均保持代码默认值,不在命令行覆盖

################################################################################
# Batch 配置 - 针对 d26 模型优化
################################################################################

DEVICE_BATCH_SIZE=8
GRAD_ACCUM=8
CONTEXT_LENGTH=256
NUM_GPUS=8

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
# 131,072 tokens/step

################################################################################
# 训练步数计算
################################################################################

# d26 模型规模: 26层, 13头, 1664维 ≈ 400M 参数
# 目标训练量: ~10B tokens
# 131k tokens/step × 75,400 steps ≈ 9.88B tokens

MAX_STEPS=75400
MAX_SCHEDULING_STEPS=75400

################################################################################
# 学习率配置 (基于模型规模推荐)
################################################################################
# 来源: https://github.com/openway-ai/nova/blob/main/nova/ebt/utils.py
#
# d26 模型规模分析:
#   - 26 layers, 13 heads, 1664 dim
#   - 参数量: ~1031M (1.03B)
#
# 模型规模对比:
#   small (162M):   LR = 0.0006
#   medium (405M):  LR = 0.0003
#   large (833M):   LR = 0.00025  ← 最接近
#   d26 (1031M):    LR = 0.00025  ← 推荐 (介于large和xl之间,更接近large)
#   xl (1413M):     LR = 0.0002
#
# 重要: d26 比 large 大 24%, 但远小于 xl (仅为 xl 的 73%)
#       因此应该使用 large 的 LR, 而不是 small 的 LR!
#
# 注意: Alpha 的实际 LR = PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER
#       = 0.00025 × 1500 = 0.375 ✓ (官方推荐范围 0.3-0.9)
################################################################################

PEAK_LR=0.00025  # 与 large 一致 (d26 参数量更接近 large 而非 small)

# Warmup 配置: 5% (较保守,有助于稳定 EBT 初期)
WARM_UP_STEPS=3770  # 5% of 75400
WARM_UP_BASE_LR_DIVIDER=10

# LR 衰减: 使用标准 Cosine Annealing
# 避免 linear_warmdown 的突变
MIN_LR_SCALE=50  # peak_lr / 50 作为最终 LR (0.00025/50 = 0.000005)
                 # 适中的衰减,不过于激进

################################################################################
# 优化器配置 (保守稳定)
################################################################################

# Weight Decay: 适度正则化
# 原 0.05 可能对 400M 模型过强,降低到 0.02
WEIGHT_DECAY=0.02

# Adam Beta 参数: 使用 PyTorch 标准值
# 不采用 NanoChat 的激进设置 (那是针对 Muon 优化器的)
BETA1=0.9      # 标准动量
BETA2=0.999    # 标准二阶矩估计

# 梯度裁剪: 标准设置
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

# VAL_CHECK_INTERVAL=1000
VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8

################################################################################
# 优化选项配置
################################################################################
#
# 策略: 先使用 Baseline 验证稳定性
#
# 可选优化 (逐个测试):
#   1. --layered_lr: 分层学习率 (embedding, transformer, scalar)
#   2. --dynamic_wd: 动态 Weight Decay (线性衰减到0)
#   3. --linear_warmdown: NanoChat 风格 LR 调度
#
# 注意:
#   - 这些优化可能与 EBT 的 MCMC 机制冲突
#   - 建议在 baseline 稳定后,单独测试每个选项
#   - 避免多个优化同时启用造成干扰
################################################################################

# === Baseline: 不启用任何高级优化 (推荐) ===
OPTION_FLAGS=""

# === 如果 Baseline 稳定,可尝试逐个测试以下选项 ===
# OPTION_FLAGS="--layered_lr"
# OPTION_FLAGS="--dynamic_wd"
# OPTION_FLAGS="--linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0"

################################################################################
# torch.compile 配置
################################################################################
# EBT 使用 autograd.grad 进行 MCMC 更新,与 fullgraph 模式不兼容
# 推荐: transformer_only 模式,仅编译 transformer 部分

# 暂时禁用,确保稳定性优先
COMPILE_FLAGS=""

# 稳定后可尝试启用
# COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

################################################################################
# 日志配置
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_ebt_d26_stable.log"
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

print_header "EBT d26 稳定训练配置 (官方推荐参数)"

echo ""
echo -e "${CYAN}▶ 关键修复${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} MCMC LR Multiplier: 2000 → ${BOLD}1500${NC} (遵循官方3x建议)"
echo "  ${GREEN}✓${NC} Peak LR: 0.0006 → ${BOLD}0.00025${NC} (d26应使用large的LR,非small)"
echo "  ${GREEN}✓${NC} Alpha 实际 LR: 1.2 → ${BOLD}0.375${NC} (降低69%!)"
echo "  ${GREEN}✓${NC} 禁用所有高级优化选项 (避免干扰)"
echo "  ${GREEN}✓${NC} Adam Beta: 恢复标准值 (beta1=0.9, beta2=0.999)"
echo "  ${GREEN}✓${NC} Weight Decay: 0.05 → ${BOLD}0.02${NC} (适度正则化)"
echo "  ${GREEN}✓${NC} LR 调度: 使用 Cosine Annealing (避免突变)"

echo ""
echo -e "${CYAN}▶ EBT 核心参数 (官方推荐)${NC}"
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
local alpha_lr=$(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
print_kv "Alpha Effective LR" "${BOLD}${alpha_lr}${NC} (=${PEAK_LR}×${MCMC_STEP_SIZE_LR_MULTIPLIER})"
print_kv "Warmup Steps" "${WARM_UP_STEPS} (5%)"
print_kv "Min LR Scale" "${MIN_LR_SCALE}"
local final_lr=$(awk "BEGIN {printf \"%.8f\", ${PEAK_LR}/${MIN_LR_SCALE}}")
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
print_kv "Max Steps" "${MAX_STEPS}"
local total_tokens=$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")
print_kv "Total Tokens" "~${total_tokens}B"
print_kv "Val Check Interval" "${VAL_CHECK_INTERVAL}"

echo ""
echo -e "${CYAN}▶ 输出配置${NC}"
print_separator "─" 60
print_kv "Run Name" "${RUN_NAME}_${current_time}"
print_kv "Log File" "${LOG_FILE}"
print_kv "WandB Mode" "${WANDB_MODE}"

echo ""
echo -e "${YELLOW}⚠ 重要说明${NC}"
echo "  1. 本配置严格遵循 EBT 官方推荐参数"
echo "  2. 使用 Baseline 模式,不启用任何高级优化"
echo "  3. 如需启用优化,请在验证 Baseline 稳定后逐个测试"
echo "  4. 监控关键指标: train_loss, Alpha_MCMC, Global_LR"

echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT d26 稳定训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 关键修复:
#   1. MCMC LR Multiplier: 2000 → 1500 (官方3x建议)
#   2. Peak LR: 0.0006 → 0.00025 (d26应使用large的LR)
#   3. Alpha Effective LR: 1.2 → 0.375 (降低69%!)
#   4. 禁用所有高级优化选项
#   5. 恢复保守 Adam Beta 参数
#   6. Weight Decay: 0.05 → 0.02
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
Context Length:           ${CONTEXT_LENGTH}
Device Batch Size:        ${DEVICE_BATCH_SIZE}
Gradient Accumulation:    ${GRAD_ACCUM}
Effective Batch Size:     ${EFFECTIVE_BATCH_SIZE} tokens/step

Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Warmup Steps:             ${WARM_UP_STEPS}
Max Steps:                ${MAX_STEPS}
Total Tokens:             ~${total_tokens}B

Option Flags:             ${OPTION_FLAGS:-"None (Baseline)"}
Compile Flags:            ${COMPILE_FLAGS:-"None"}

================================================================================
[训练输出]
================================================================================

LOG_HEADER

################################################################################
# 启动训练
################################################################################

echo ""
print_header "开始训练"
echo ""

set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/train.py \
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
--wandb_watch \
${OPTION_FLAGS} \
${COMPILE_FLAGS} \
2>&1 | tee -a "${LOG_FILE}"

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

对比基准 (之前失败的配置):
  - 之前: Alpha LR = 1.2, 多次 loss 暴涨
  - 现在: Alpha LR = 0.375, 预期稳定

MONITOR_HELP

echo ""
