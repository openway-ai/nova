#!/bin/bash
################################################################################
# EBT 评估入口脚本
#
# 所有参数在此统一配置，自动传递给子脚本
# 运行: bash runs/eval_ebt_nanochat_gsm8k.sh
################################################################################

set -e  # 遇到错误立即退出

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

################################################################################
# 通用配置 (所有子脚本共享)
################################################################################

# Checkpoint 路径
export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt}"

# 验证 checkpoint 存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 未找到 checkpoint: $CKPT_PATH"
    exit 1
fi

# Tokenizer 路径
export TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"

# GPU 和批量配置
export GPUS="${GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"

# WandB 配置
export USE_WANDB="${USE_WANDB:-false}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

################################################################################
# NanoChat 分片评估配置
################################################################################

export EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,15}"
export MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-5}"
export ENABLE_GENERATION="${ENABLE_GENERATION:-true}"
export GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"
export MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"

################################################################################
# GSM8K 评估配置
################################################################################

export EVAL_TASK="${EVAL_TASK:-gsm8k}"

################################################################################
# 初始化
################################################################################

echo "📦 使用 checkpoint: $CKPT_PATH"
echo ""

# 创建汇总日志目录
EVAL_BASE_DIR="$EBT_DIR/logs/eval"
SUMMARY_DIR="${EVAL_BASE_DIR}/summary"
mkdir -p "$SUMMARY_DIR"
SUMMARY_FILE="$SUMMARY_DIR/eval_summary_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "======================================================================"
    echo "EBT 模型评估汇总报告"
    echo "时间: $(date)"
    echo "Checkpoint: $CKPT_PATH"
    echo "======================================================================"
    echo ""
} > "$SUMMARY_FILE"

################################################################################
# 示例 1: NanoChat 分片评估
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "示例 1: NanoChat 分片评估 (shard $EVAL_SHARD_INDICES)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

{
    echo "----------------------------------------"
    echo "任务 1: NanoChat 分片 PPL (shard $EVAL_SHARD_INDICES)"
    echo "----------------------------------------"
} >> "$SUMMARY_FILE"

bash "$SCRIPT_DIR/eval_nanochat_shards.sh"

# 提取结果到汇总文件
RESULT_FILE=$(ls -t $EBT_DIR/logs/eval_nanochat_shards_*.log 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    echo "结果摘要:" >> "$SUMMARY_FILE"
    grep -A 15 "TEST EPOCH SUMMARY" "$RESULT_FILE" | tail -16 >> "$SUMMARY_FILE" 2>/dev/null || echo "  未找到汇总指标" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

echo ""
echo ""

################################################################################
# 示例 2: GSM8K 数学推理
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "示例 2: GSM8K 数学推理任务"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

{
    echo "----------------------------------------"
    echo "任务 2: GSM8K"
    echo "----------------------------------------"
} >> "$SUMMARY_FILE"

bash "$SCRIPT_DIR/eval_ebt.sh"

RESULT_FILE=$(ls -t ${EVAL_BASE_DIR}/logs/gsm8k_*_result.txt 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    cat "$RESULT_FILE" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

echo ""
echo ""

################################################################################
# 汇总报告
################################################################################

echo "======================================================================"
echo "✅ 所有评估完成！"
echo "======================================================================"
echo ""
echo "📊 汇总报告已保存到: $SUMMARY_FILE"
echo ""
echo "查看报告:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "======================================================================"
