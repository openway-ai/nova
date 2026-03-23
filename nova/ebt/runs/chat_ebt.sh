#!/bin/bash
################################################################################
# EBT 交互式对话脚本
#
# 与训练好的 EBT 模型进行交互式对话，支持展示 MCMC 迭代过程
#
# 使用方法:
#   bash runs/chat_ebt.sh                    # 默认模式
#   bash runs/chat_ebt.sh --show-mcmc        # 展示 MCMC 步骤
#   bash runs/chat_ebt.sh --verbose          # 详细模式
#   bash runs/chat_ebt.sh --help             # 显示帮助
#
# 环境变量:
#   CKPT_PATH      - Checkpoint 路径 (可选)
#   TEMPERATURE    - 生成温度 (默认: 0.8)
#   TOP_P          - Top-P 采样 (默认: 0.9)
#   MAX_TOKENS     - 最大生成 tokens (默认: 256)
#   SHOW_MCMC      - 是否展示 MCMC (true/false, 默认: false)
#   VERBOSE        - 详细模式 (true/false, 默认: false)
################################################################################

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 默认配置
DEFAULT_CKPT="/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt"
DEFAULT_TOKENIZER="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer"

# 从环境变量读取配置
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

# 解析命令行参数
SHOW_MCMC_FLAG=""
VERBOSE_FLAG=""
SHOW_ENERGY_FLAG=""
SHOW_DIST_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --show-mcmc|-m)
            SHOW_MCMC_FLAG="--show-mcmc"
            shift
            ;;
        --verbose|-v)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        --show-energy|-e)
            SHOW_ENERGY_FLAG="--show-energy"
            shift
            ;;
        --show-distribution|-d)
            SHOW_DIST_FLAG="--show-distribution"
            shift
            ;;
        --checkpoint|-c)
            CKPT_PATH="$2"
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p|-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "EBT 交互式对话脚本"
            echo ""
            echo "使用方法:"
            echo "  bash runs/chat_ebt.sh [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --checkpoint PATH    Checkpoint 路径"
            echo "  -t, --temperature VAL    生成温度 (默认: 0.8)"
            echo "  -p, --top-p VAL          Top-P 采样 (默认: 0.9)"
            echo "  --max-tokens VAL         最大生成 tokens (默认: 256)"
            echo "  -m, --show-mcmc          展示 MCMC 步骤过程"
            echo "  -v, --verbose            详细模式"
            echo "  -e, --show-energy        展示能量值变化"
            echo "  -d, --show-distribution  展示概率分布变化"
            echo "  --dtype TYPE             数据类型 (float32/bfloat16, 默认: bfloat16)"
            echo "  --device DEVICE          设备 (cuda/cpu, 默认: cuda)"
            echo "  -h, --help               显示此帮助"
            echo ""
            echo "环境变量:"
            echo "  CKPT_PATH      - Checkpoint 路径"
            echo "  TEMPERATURE    - 生成温度"
            echo "  TOP_P          - Top-P 采样"
            echo "  MAX_TOKENS     - 最大生成 tokens"
            echo ""
            echo "示例:"
            echo "  # 基本对话"
            echo "  bash runs/chat_ebt.sh"
            echo ""
            echo "  # 展示 MCMC 迭代过程"
            echo "  bash runs/chat_ebt.sh --show-mcmc"
            echo ""
            echo "  # 详细模式 + MCMC 显示"
            echo "  bash runs/chat_ebt.sh --show-mcmc --verbose"
            echo ""
            echo "  # 使用自定义 checkpoint"
            echo "  bash runs/chat_ebt.sh -c /path/to/checkpoint.ckpt"
            echo ""
            echo "  # 调整生成参数"
            echo "  TEMPERATURE=0.6 TOP_P=0.95 bash runs/chat_ebt.sh"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 从环境变量设置标志
if [ "${SHOW_MCMC:-false}" = "true" ]; then
    SHOW_MCMC_FLAG="--show-mcmc"
fi

if [ "${VERBOSE:-false}" = "true" ]; then
    VERBOSE_FLAG="--verbose"
fi

# 验证 checkpoint 存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: 找不到 checkpoint: $CKPT_PATH"
    echo ""
    echo "请设置 CKPT_PATH 环境变量或使用 --checkpoint 参数指定正确的路径"
    exit 1
fi

# 设置环境变量
export NANOCHAT_OFFLINE_MODE=1
export HF_HUB_OFFLINE=1
export NANOCHAT_BASE_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"

# 清除分布式训练环境变量
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# 激活 conda 环境 (如果需要)
if command -v conda &> /dev/null; then
    CONDA_ENV_PATH="/mnt/shared-storage-user/puyuan/conda_envs/nanochat"
    if [ -d "$CONDA_ENV_PATH" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV_PATH" 2>/dev/null || true
    fi
fi

# 显示配置
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EBT 交互式对话"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Checkpoint: $CKPT_PATH"
echo "温度: $TEMPERATURE | Top-P: $TOP_P | 最大 Tokens: $MAX_TOKENS"
echo "MCMC 显示: ${SHOW_MCMC_FLAG:-关闭} | 详细模式: ${VERBOSE_FLAG:-关闭}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 切换到 EBT 目录
cd "$EBT_DIR"

# 运行 Python 脚本
python -m scripts.chat_ebt \
    --checkpoint "$CKPT_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --override-mcmc-steps 10 \
    $SHOW_MCMC_FLAG \
    $VERBOSE_FLAG \
    $SHOW_ENERGY_FLAG \
    $SHOW_DIST_FLAG
