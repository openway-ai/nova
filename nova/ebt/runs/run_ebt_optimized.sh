#!/bin/bash

################################################################################
# EBT Large 鲁棒训练脚本 - 内存优化版
#
# 针对 140GB GPU 优化的最鲁棒配置
# 基于实际 OOM 错误分析
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-large-robust
#SBATCH --output=logs/slurm/nlp/ebt-large-robust_%A-%a.log

### 基础配置 ###
# export RUN_NAME="ebt-large-robust"
# export MODEL_NAME="${RUN_NAME%%-*}"
# export MODEL_SIZE="large"

export RUN_NAME="ebt-d26-robust"
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
# 关键内存优化策略
################################################################################

# 策略 1: 减小 Batch Size（最重要）
# 问题: Large 模型 + EBT MCMC 双倍开销 + attention O(n^2)
# 解决: 使用更小的 micro-batch，更多的 gradient accumulation
#
# 原配置: 32 batch × 1 accum = 32 effective (OOM)
# 优化 1: 16 batch × 2 accum = 32 effective (可能还会 OOM)
# 优化 2: 8 batch × 4 accum = 32 effective (稳定)
# 最鲁棒: 4 batch × 8 accum = 32 effective (极度稳定)

DEVICE_BATCH_SIZE=8
GRAD_ACCUM=8

# 策略 2: 减小序列长度（如果可接受）
# EBT 的 attention 是 O(n^2)，序列长度减半 → 内存降 4 倍
# 256 → 128: 内存降 75%，但可能影响性能
#
# 推荐: 先尝试 256，如果还 OOM 再降到 128
CONTEXT_LENGTH=256

# 策略 3: 调整总 batch size 以保持训练效果
# 目标: 保持与原配置相同的 effective batch size
#
# 8 GPUs × 8 batch × 8 accum × 256 seq = 131,072 tokens/step
# 这比原计划的 262k 小一半，但比 Small 的 98k 大
NUM_GPUS=8

# 计算新的训练步数
# 原计划: 262k tokens/step, 37,700 steps = 9.9B tokens
# 新配置: 131k tokens/step, 需要 75,400 steps 达到相同 token 量
#
# 为了保持训练时间合理，我们可以:
# 选项 A: 保持 9.9B tokens (75,400 steps) - 完整训练
# 选项 B: 降到 5B tokens (38,200 steps) - 快速训练
#
# 推荐选项 A
MAX_STEPS=75400
MAX_SCHEDULING_STEPS=75400

################################################################################
# 其他配置
################################################################################

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE tokens/step"

# 学习率保持不变
# PEAK_LR=0.00025
# MCMC_STEP_SIZE=500
# MCMC_STEP_SIZE_LR_MULTIPLIER=300

################################################################################
# 优化器配置 - 参考 base_train.py 并适配 EBT 特性
################################################################################

# 1. 学习率配置
# base_train.py 使用分层学习率:
#   - embedding_lr: 0.3 (Adam)
#   - unembedding_lr: 0.004 (Adam)
#   - matrix_lr: 0.02 (Muon)
#   - scalar_lr: 0.5 (Adam)
#
# EBT 的特殊性:
#   - alpha (MCMC step size) 需要更大的学习率以快速适应
#   - 其他参数使用标准 AdamW
#
# 策略: 使用适中的 peak_lr，并为 alpha 参数使用更高的 multiplier
PEAK_LR=0.0006  # 降低基础学习率，从 0.0012 -> 0.0006 (更稳定)
MCMC_STEP_SIZE=500
MCMC_STEP_SIZE_LR_MULTIPLIER=2000  # alpha 的 LR = 0.0006 * 2000 = 1.2 (保持较高学习率以快速适应)

# 2. AdamW Beta 参数
# base_train.py: beta1=0.8, beta2=0.95 (更激进的动量设置)
# EBT 当前默认: beta1=0.9, beta2=0.999 (PyTorch 标准)
#
# 推荐: 采用 base_train.py 的设置，可以:
#   - 减少 momentum 累积，更快适应新梯度
#   - beta2=0.95 对于大 batch 训练更稳定
BETA1=0.8   # 从默认 0.9 降低到 0.8
BETA2=0.95  # 从默认 0.999 降低到 0.95

# 3. Weight Decay 策略
# base_train.py: 使用动态 weight decay
#   - 基础值: 0.2
#   - 随训练线性衰减到 0
#   - 根据 batch size 和 token 数量缩放
#   - 计算公式: λ = λ_ref · √(B/B_ref) · (D_ref/D)
#
# EBT 当前: weight_decay=0.01 (固定)
#
# 策略: 提高 weight decay 基础值，并建议实现动态衰减
# 131k tokens/step, 75400 steps = 9.9B tokens
# 参考 d12: B_REF=524288, D_REF≈5.5B
# batch_ratio = 131072 / 524288 ≈ 0.25
# data_ratio = 5.5B / 9.9B ≈ 0.56
# weight_decay_scaled = 0.2 * sqrt(0.25) * 0.56 = 0.2 * 0.5 * 0.56 ≈ 0.056
WEIGHT_DECAY=0.05  # 从 0.01 提高到 0.05 (更强正则化)

# 4. 学习率调度策略
# base_train.py:
#   - warmup_ratio: 0.0 (无 warmup，Muon 优化器不需要)
#   - warmdown_ratio: 0.5 (后 50% 步数线性衰减)
#   - final_lr_frac: 0.0 (衰减到 0)
#
# EBT 当前:
#   - warm_up_steps: 1508 (2% of 75400)
#   - cosine annealing 到 peak_lr / min_lr_scale
#
# 策略: 保持 warmup (EBT 训练初期需要稳定)，但调整比例和调度方式
# 参考 Chinchilla 和 base_train.py 的经验:
#   - 更长的 warmup 可以稳定 EBT 的 MCMC 过程
#   - 更激进的 cooldown 可以提升最终性能
WARM_UP_STEPS=3770      # 5% of 75400 (从 2% 增加到 5%)
MIN_LR_SCALE=100        # peak_lr / 100 作为最终 LR (从 20 增加到 100，更激进的衰减)
WARM_UP_BASE_LR_DIVIDER=10  # warmup 从 peak_lr/10 开始 (新增参数)

# 5. 梯度裁剪
# base_train.py: 无显式梯度裁剪 (Muon 优化器内置处理)
# EBT 当前: gradient_clip_val=1.0
#
# 策略: 保持梯度裁剪，但调整阈值
# EBT 的 MCMC 梯度可能更嘈杂，需要裁剪
GRADIENT_CLIP_VAL=1.0  # 保持 1.0 (标准设置)
GRADIENT_CLIP_ALGORITHM="norm"  # 使用 L2 norm 裁剪 (默认)

# 6. 其他训练超参数
VAL_CHECK_INTERVAL=1000  # 保持不变
LIMIT_VAL_BATCHES=50     # 保持不变
NUM_WORKERS=8            # 保持不变

################################################################################
# 优化建议总结
################################################################################
#
# 主要改进:
# 1. 降低基础学习率 (0.0012 -> 0.0006) 以提高训练稳定性
# 2. 采用更激进的 Adam beta 参数 (beta1=0.8, beta2=0.95) 以加快收敛
# 3. 提高 weight decay (0.01 -> 0.05) 以增强正则化
# 4. 延长 warmup 期 (2% -> 5%) 以稳定 EBT 初期训练
# 5. 更激进的学习率衰减 (min_lr_scale: 20 -> 100)
#
# 进一步优化方向 (需要修改 train.py):
# 1. 实现动态 weight decay (线性衰减到 0)
# 2. 实现分层学习率 (embedding vs. transformer vs. alpha)
# 3. 考虑使用 Muon 优化器替代 AdamW (针对矩阵参数)
# 4. 实现 batch size 和 token 数量的自动缩放
################################################################################

################################################################################
# 配置总结
################################################################################

################################################################################
# 选择要启用的 Option (取消注释对应行)
################################################################################

# === Baseline: 不启用任何优化 ===
OPTION_FLAGS=""

# === Option 1: 分层学习率 ===
# OPTION_FLAGS="--layered_lr"
# OPTION_FLAGS="--layered_lr --embedding_lr_mult 0.3 --vocab_to_embed_lr_mult 0.1 --scalar_lr_mult 0.5"

# === Option 2: 动态 Weight Decay ===
# OPTION_FLAGS="--dynamic_wd"

# === Option 3: Linear Warmdown LR 调度 (NanoChat 风格) ===
# OPTION_FLAGS="--linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0"

# === Option 1 + 2: 分层学习率 + 动态 Weight Decay ===
# OPTION_FLAGS="--layered_lr --dynamic_wd"

# === Option 1 + 2 + 3: 全部优化 ===
OPTION_FLAGS="--layered_lr --dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0"

# === Option 2 + 3: 动态 WD + Linear Warmdown ===
# OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0"

################################################################################
# torch.compile 配置
################################################################################
# 注意: EBT 使用 autograd.grad 进行 MCMC 更新，与 fullgraph 模式不兼容
# 推荐使用 transformer_only 模式，仅编译 transformer 部分

# 启用 torch.compile (推荐)
# COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

# 或者禁用 torch.compile
COMPILE_FLAGS=""

# 或者尝试编译整个模型 (可能失败)
# COMPILE_FLAGS="--compile_model --compile_mode full --compile_dynamic"


current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_ebt_d26.log"

# 创建日志目录
mkdir -p logs

################################################################################
# 日志系统配置
################################################################################

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# 日志级别: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
LOG_LEVEL=${LOG_LEVEL:-1}

# 获取带时间戳的日志前缀
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# 通用日志函数
log() {
    local level=$1
    local level_name=$2
    local color=$3
    local message=$4
    local timestamp=$(get_timestamp)

    if [[ $level -ge $LOG_LEVEL ]]; then
        echo -e "${DIM}[${timestamp}]${NC} ${color}[${level_name}]${NC} ${message}"
    fi
}

# 各级别日志函数
log_debug() { log 0 "DEBUG" "${CYAN}" "$1"; }
log_info()  { log 1 "INFO " "${GREEN}" "$1"; }
log_warn()  { log 2 "WARN " "${YELLOW}" "$1"; }
log_error() { log 3 "ERROR" "${RED}" "$1"; }

# 打印分隔线
print_separator() {
    local char=${1:-"═"}
    local width=${2:-80}
    printf "${BLUE}"
    printf '%*s' "$width" | tr ' ' "$char"
    printf "${NC}\n"
}

# 打印标题
print_header() {
    local title=$1
    echo ""
    print_separator "═"
    echo -e "${BOLD}${GREEN}  $title${NC}"
    print_separator "═"
}

# 打印子标题
print_subheader() {
    local title=$1
    echo ""
    echo -e "${BOLD}${CYAN}▶ $title${NC}"
    print_separator "─" 60
}

# 打印键值对（对齐格式）
print_kv() {
    local key=$1
    local value=$2
    local width=${3:-28}
    printf "  ${DIM}%-${width}s${NC} %s\n" "$key:" "$value"
}

# 打印带状态的键值对
print_kv_status() {
    local key=$1
    local value=$2
    local status=$3  # ok, warn, error
    local width=${4:-28}
    local status_icon=""

    case $status in
        ok)    status_icon="${GREEN}✓${NC}" ;;
        warn)  status_icon="${YELLOW}⚠${NC}" ;;
        error) status_icon="${RED}✗${NC}" ;;
        *)     status_icon=" " ;;
    esac

    printf "  ${DIM}%-${width}s${NC} %s %s\n" "$key:" "$value" "$status_icon"
}

################################################################################
# 环境检查函数
################################################################################

check_environment() {
    print_header "环境检查"
    local all_ok=true

    # 检查 GPU
    print_subheader "GPU 状态"
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        local gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

        if [[ $gpu_count -ge $NUM_GPUS ]]; then
            print_kv_status "可用 GPU 数量" "$gpu_count (需要 $NUM_GPUS)" "ok"
        else
            print_kv_status "可用 GPU 数量" "$gpu_count (需要 $NUM_GPUS)" "error"
            all_ok=false
        fi
        print_kv "GPU 型号" "$gpu_name"
        print_kv "GPU 显存" "$gpu_memory"
        print_kv "驱动版本" "$gpu_driver"

        # 检查 GPU 使用情况
        echo ""
        log_info "当前 GPU 使用情况:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader | while read line; do
            echo "    GPU $line"
        done
    else
        print_kv_status "nvidia-smi" "未找到" "error"
        all_ok=false
    fi

    # 检查磁盘空间
    print_subheader "磁盘空间"
    local disk_avail=$(df -h . | awk 'NR==2 {print $4}')
    local disk_avail_gb=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ $disk_avail_gb -gt 50 ]]; then
        print_kv_status "可用空间" "$disk_avail" "ok"
    elif [[ $disk_avail_gb -gt 20 ]]; then
        print_kv_status "可用空间" "$disk_avail (建议 >50GB)" "warn"
    else
        print_kv_status "可用空间" "$disk_avail (不足!)" "error"
        all_ok=false
    fi

    # 检查内存
    print_subheader "系统内存"
    local mem_total=$(free -h | awk '/^Mem:/ {print $2}')
    local mem_avail=$(free -h | awk '/^Mem:/ {print $7}')
    local mem_avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
    print_kv "总内存" "$mem_total"
    if [[ $mem_avail_gb -gt 32 ]]; then
        print_kv_status "可用内存" "$mem_avail" "ok"
    elif [[ $mem_avail_gb -gt 16 ]]; then
        print_kv_status "可用内存" "$mem_avail" "warn"
    else
        print_kv_status "可用内存" "$mem_avail (可能不足)" "warn"
    fi

    # 检查关键目录和文件
    print_subheader "路径检查"
    if [[ -d "$NANOCHAT_BASE_DIR" ]]; then
        print_kv_status "NANOCHAT_BASE_DIR" "$NANOCHAT_BASE_DIR" "ok"
    else
        print_kv_status "NANOCHAT_BASE_DIR" "$NANOCHAT_BASE_DIR (不存在)" "error"
        all_ok=false
    fi

    local train_script="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/train.py"
    if [[ -f "$train_script" ]]; then
        print_kv_status "训练脚本" "train.py 存在" "ok"
    else
        print_kv_status "训练脚本" "train.py 不存在" "error"
        all_ok=false
    fi

    # 检查 Python 环境
    print_subheader "Python 环境"
    local python_version=$(python3 --version 2>&1)
    print_kv "Python 版本" "$python_version"

    if python3 -c "import torch" 2>/dev/null; then
        local torch_version=$(python3 -c "import torch; print(torch.__version__)")
        local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")
        print_kv_status "PyTorch 版本" "$torch_version" "ok"
        if [[ "$cuda_available" == "True" ]]; then
            print_kv_status "CUDA 可用" "是" "ok"
        else
            print_kv_status "CUDA 可用" "否" "error"
            all_ok=false
        fi
    else
        print_kv_status "PyTorch" "未安装" "error"
        all_ok=false
    fi

    # 检查 WandB 配置
    print_subheader "WandB 配置"
    print_kv "WANDB_MODE" "${WANDB_MODE:-online}"
    if [[ -n "$WANDB_API_KEY" ]]; then
        print_kv_status "WANDB_API_KEY" "已设置 (${WANDB_API_KEY:0:8}...)" "ok"
    else
        print_kv_status "WANDB_API_KEY" "未设置" "warn"
    fi

    echo ""
    if $all_ok; then
        log_info "环境检查通过 ✓"
    else
        log_error "环境检查发现问题，请检查上述错误项"
        echo ""
        read -p "是否继续? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "用户取消训练"
            exit 1
        fi
    fi
}

################################################################################
# 显示训练配置
################################################################################

show_training_config() {
    print_header "EBT 训练配置 - ${MODEL_SIZE} 模型"

    print_subheader "模型配置"
    print_kv "模型名称" "${MODEL_NAME}"
    print_kv "模型规模" "${MODEL_SIZE}"
    print_kv "上下文长度" "${CONTEXT_LENGTH}"

    print_subheader "Batch 配置"
    print_kv "Device Batch Size" "${DEVICE_BATCH_SIZE}"
    print_kv "Gradient Accumulation" "${GRAD_ACCUM}"
    print_kv "GPU 数量" "${NUM_GPUS}"
    print_kv "Effective Batch" "${EFFECTIVE_BATCH_SIZE} tokens/step"

    print_subheader "优化器配置"
    print_kv "Peak Learning Rate" "${PEAK_LR}"
    print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
    print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER}"
    local alpha_lr=$(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
    print_kv "Alpha 实际 LR" "${alpha_lr}"
    print_kv "Weight Decay" "${WEIGHT_DECAY}"
    print_kv "Adam Beta1" "${BETA1}"
    print_kv "Adam Beta2" "${BETA2}"
    print_kv "Gradient Clip" "${GRADIENT_CLIP_VAL}"

    print_subheader "学习率调度"
    print_kv "Warmup Steps" "${WARM_UP_STEPS} ($(awk "BEGIN {printf \"%.1f\", ${WARM_UP_STEPS}/${MAX_STEPS}*100}")%)"
    print_kv "Warmup 起始 LR" "peak_lr / ${WARM_UP_BASE_LR_DIVIDER}"
    print_kv "Min LR Scale" "${MIN_LR_SCALE}"
    local final_lr=$(awk "BEGIN {printf \"%.8f\", ${PEAK_LR}/${MIN_LR_SCALE}}")
    print_kv "最终 LR" "${final_lr}"

    print_subheader "训练进度"
    print_kv "最大步数" "${MAX_STEPS}"
    print_kv "调度步数" "${MAX_SCHEDULING_STEPS}"
    local total_tokens=$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")
    print_kv "预计总 Tokens" "~${total_tokens}B"
    print_kv "验证间隔" "每 ${VAL_CHECK_INTERVAL} 步"

    print_subheader "优化选项"
    if [[ -n "$OPTION_FLAGS" ]]; then
        echo "  ${GREEN}已启用:${NC}"
        echo "$OPTION_FLAGS" | tr ' ' '\n' | grep -v '^$' | while read flag; do
            echo "    • $flag"
        done
    else
        echo "  ${DIM}未启用任何优化选项 (Baseline)${NC}"
    fi

    print_subheader "编译配置"
    if [[ -n "$COMPILE_FLAGS" ]]; then
        print_kv_status "torch.compile" "已启用" "ok"
        echo "  ${DIM}Flags:${NC} $COMPILE_FLAGS"
    else
        print_kv_status "torch.compile" "已禁用" "warn"
    fi

    print_subheader "输出配置"
    print_kv "日志文件" "${LOG_FILE}"
    print_kv "WandB 项目" "nlp_pretrain"
    print_kv "Run Name" "${RUN_NAME}_${current_time}"

    echo ""
}

################################################################################
# 执行环境检查和显示配置
################################################################################

check_environment
show_training_config

echo ""
log_info "配置检查完成，准备开始训练"
echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 启动训练
################################################################################

# 记录训练开始时间
TRAIN_START_TIME=$(date +%s)
TRAIN_START_DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
print_header "开始训练"
log_info "训练启动时间: ${TRAIN_START_DATE}"
log_info "Run Name: ${RUN_NAME}_${current_time}"
log_info "日志文件: ${LOG_FILE}"
echo ""

# 写入结构化日志头
write_log_header() {
    cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                         EBT Training Log
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      ${TRAIN_START_DATE}
# Log File:        ${LOG_FILE}
#
################################################################################

================================================================================
[SYSTEM INFO]
================================================================================
Hostname:         $(hostname)
User:             $(whoami)
Working Dir:      $(pwd)
Python:           $(python3 --version 2>&1)
PyTorch:          $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA Available:   $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
GPU Count:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "N/A")
GPU Model:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

================================================================================
[TRAINING CONFIGURATION]
================================================================================

[Model]
  Model Name:           ${MODEL_NAME}
  Model Size:           ${MODEL_SIZE}
  Context Length:       ${CONTEXT_LENGTH}

[Batch]
  Device Batch Size:    ${DEVICE_BATCH_SIZE}
  Gradient Accumulation: ${GRAD_ACCUM}
  Num GPUs:             ${NUM_GPUS}
  Effective Batch:      ${EFFECTIVE_BATCH_SIZE} tokens/step

[Optimizer]
  Peak LR:              ${PEAK_LR}
  MCMC Step Size:       ${MCMC_STEP_SIZE}
  MCMC LR Multiplier:   ${MCMC_STEP_SIZE_LR_MULTIPLIER}
  Alpha Effective LR:   $(awk "BEGIN {printf \"%.4f\", ${PEAK_LR} * ${MCMC_STEP_SIZE_LR_MULTIPLIER}}")
  Weight Decay:         ${WEIGHT_DECAY}
  Beta1:                ${BETA1}
  Beta2:                ${BETA2}
  Gradient Clip:        ${GRADIENT_CLIP_VAL}

[LR Schedule]
  Warmup Steps:         ${WARM_UP_STEPS} ($(awk "BEGIN {printf \"%.1f\", ${WARM_UP_STEPS}/${MAX_STEPS}*100}")%)
  Min LR Scale:         ${MIN_LR_SCALE}
  Final LR:             $(awk "BEGIN {printf \"%.8f\", ${PEAK_LR}/${MIN_LR_SCALE}}")

[Training]
  Max Steps:            ${MAX_STEPS}
  Scheduling Steps:     ${MAX_SCHEDULING_STEPS}
  Total Tokens:         ~$(awk "BEGIN {printf \"%.2f\", ${MAX_STEPS} * ${EFFECTIVE_BATCH_SIZE} / 1000000000}")B
  Val Check Interval:   ${VAL_CHECK_INTERVAL}

[Options]
  Option Flags:         ${OPTION_FLAGS:-"None (Baseline)"}
  Compile Flags:        ${COMPILE_FLAGS:-"None (Disabled)"}

================================================================================
[TRAINING OUTPUT]
================================================================================

LOG_HEADER
}

write_log_header

# 显示训练命令
log_info "执行训练命令..."
log_debug "torchrun --standalone --nproc_per_node=${NUM_GPUS} train.py ..."

# 捕获训练退出状态
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
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps 2 \
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

# 记录训练结束
TRAIN_END_TIME=$(date +%s)
TRAIN_END_DATE=$(date '+%Y-%m-%d %H:%M:%S')
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
TRAIN_HOURS=$((TRAIN_DURATION / 3600))
TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))
TRAIN_SECONDS=$((TRAIN_DURATION % 60))

echo ""
print_header "训练结束"

# 根据退出码显示状态
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    log_info "训练状态: ${GREEN}成功完成${NC}"
else
    log_error "训练状态: ${RED}异常退出 (exit code: $TRAIN_EXIT_CODE)${NC}"
fi

log_info "结束时间: ${TRAIN_END_DATE}"
log_info "总耗时: ${BOLD}${TRAIN_HOURS}h ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s${NC}"
echo ""

# 追加训练摘要到日志
cat << TRAIN_SUMMARY >> "${LOG_FILE}"

================================================================================
[TRAINING SUMMARY]
================================================================================
Run Name:         ${RUN_NAME}_${current_time}
Start Time:       ${TRAIN_START_DATE}
End Time:         ${TRAIN_END_DATE}
Duration:         ${TRAIN_HOURS}h ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s
Exit Code:        ${TRAIN_EXIT_CODE}
Status:           $([ $TRAIN_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")
================================================================================

TRAIN_SUMMARY

# 如果训练失败，提供诊断信息
if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo ""
    log_warn "训练异常退出，以下是可能的诊断信息:"
    echo ""

    # 检查是否是 OOM
    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
        log_error "检测到 GPU 内存不足 (OOM)"
        echo "  建议: 减小 DEVICE_BATCH_SIZE 或 CONTEXT_LENGTH"
    fi

    # 检查是否是 NaN
    if grep -q "nan\|NaN\|inf\|Inf" "${LOG_FILE}" 2>/dev/null | head -5; then
        log_error "检测到 NaN/Inf 值"
        echo "  建议: 降低学习率或检查数据"
    fi

    # 显示最后几行错误
    echo ""
    log_info "日志最后 20 行:"
    tail -20 "${LOG_FILE}" | sed 's/^/  /'
fi

################################################################################
# 训练后信息
################################################################################

# 只在训练成功时显示后续建议
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    print_subheader "训练完成后续步骤"
    echo "  1. 查看完整日志: less ${LOG_FILE}"
    echo "  2. 查看 WandB 面板 (如果在线模式)"
    echo "  3. 检查模型 checkpoint 目录"
    echo ""
fi

################################################################################
# OOM 降级方案（仅在 OOM 时显示）
################################################################################

if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
    print_header "OOM 降级方案"
    cat << 'OOM_HELP'

如果仍然 OOM，可以尝试以下降级方案：

方案 1: 进一步减小 batch size
  DEVICE_BATCH_SIZE=4
  GRAD_ACCUM=16
  # 这将内存降到 ~3 GB/GPU

方案 2: 减小序列长度
  CONTEXT_LENGTH=128
  # Attention 内存降 75%

方案 3: 使用梯度检查点 (Gradient Checkpointing)
  在 train.py 中添加:
  --gradient_checkpointing
  # 牺牲 20% 速度换取 50% 内存节省

方案 4: 训练 Medium 模型而非 Large
  MODEL_SIZE="medium"  # 250M params
  DEVICE_BATCH_SIZE=16
  GRAD_ACCUM=4
  # 参数量降 37%，内存需求显著降低

OOM_HELP
fi

################################################################################
# 日志分析辅助命令
################################################################################

print_header "日志分析命令"
cat << ANALYSIS_HELP

实时监控 (在另一个终端运行):
  tail -f ${LOG_FILE} | grep -E "(step|loss|lr|Alpha|bpb)"

查看训练指标:
  grep -E "train_loss" ${LOG_FILE} | tail -20      # Loss 变化
  grep -E "Global_LR|Alpha_LR" ${LOG_FILE} | tail -20  # 学习率
  grep -E "Alpha_MCMC" ${LOG_FILE} | tail -20      # MCMC step size
  grep -E "bpb" ${LOG_FILE} | tail -20             # BPB 指标
  grep -E "val_" ${LOG_FILE} | tail -20            # 验证集指标

快速状态检查:
  watch -n 10 'tail -5 ${LOG_FILE} | grep -E "(loss|lr|Alpha)"'

提取指标生成 CSV:
  grep -oP "step=\K[0-9]+" ${LOG_FILE} > /tmp/steps.txt
  grep -oP "train_loss=\K[0-9.]+" ${LOG_FILE} > /tmp/loss.txt
  paste -d',' /tmp/steps.txt /tmp/loss.txt > train_loss.csv

ANALYSIS_HELP

echo ""
log_info "脚本执行完毕"
echo ""
