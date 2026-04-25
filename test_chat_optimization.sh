#!/bin/bash
################################################################################
# EBT Chat 优化测试脚本
#
# 测试优化后的 chat_ebt.py 输出质量
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
CHECKPOINT="/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt"
TOKENIZER="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer"

# 测试提示词
TEST_PROMPT="Hello! How are you?"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  EBT Chat 优化测试${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 设置环境
export NANOCHAT_OFFLINE_MODE=1
export HF_HUB_OFFLINE=1
export NANOCHAT_BASE_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"

# 清除分布式环境变量
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# 激活 conda 环境
if command -v conda &> /dev/null; then
    CONDA_ENV_PATH="/mnt/shared-storage-user/puyuan/conda_envs/nanochat"
    if [ -d "$CONDA_ENV_PATH" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate "$CONDA_ENV_PATH" 2>/dev/null || true
    fi
fi

cd /mnt/shared-storage-user/puyuan/code/nova/nova/ebt

echo -e "${YELLOW}测试 1: 使用训练参数 (默认,推荐)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "提示词: $TEST_PROMPT"
echo ""
echo "$TEST_PROMPT" | python -m scripts.chat_ebt \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --show-mcmc \
    --verbose \
    --max-tokens 50 || true

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}测试完成!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "观察指标:"
echo "  1. 熵值 (H): 应该 < 6.0 (越低越好,理想 < 3.0)"
echo "  2. 最大概率 (P_max): 应该 > 0.1 (越高越好,理想 > 0.5)"
echo "  3. Top tokens 概率: 应该有明显差异 (不应该都是 0.001)"
echo "  4. 能量变化: 应该逐步降低 (负变化)"
echo ""
echo "如果输出质量仍然差,可以尝试:"
echo "  1. 检查 checkpoint 是否训练充分"
echo "  2. 尝试降低温度: --temperature 0.5"
echo "  3. 尝试调整 MCMC 步数: --override-mcmc-steps 5"
echo ""
