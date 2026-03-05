#!/bin/bash
################################################################################
# EBT 评估使用示例（改进版）
# 使用改进的评估脚本，输出结构化日志
################################################################################

# 设置 checkpoint 路径
CHECKPOINT="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/logs/checkpoints/ebt-small-bs_256_s1_lr_0.0012_2026-03-05_01-01-55_/last.ckpt"

LATEST_CKPT=$(ls -t $CHECKPOINT 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "❌ 未找到 checkpoint"
    exit 1
fi

echo "📦 使用 checkpoint: $LATEST_CKPT"
echo ""

# 创建汇总日志目录（在统一的评估日志目录下）
EVAL_BASE_DIR="../../nova/ebt/logs/eval"
SUMMARY_DIR="${EVAL_BASE_DIR}/summary"
mkdir -p "$SUMMARY_DIR"
SUMMARY_FILE="$SUMMARY_DIR/eval_summary_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "======================================================================"
    echo "EBT 模型评估汇总报告"
    echo "时间: $(date)"
    echo "Checkpoint: $LATEST_CKPT"
    echo "======================================================================"
    echo ""
} > "$SUMMARY_FILE"

################################################################################
# 示例 1: NanoChat PPL 测试
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "示例 1: NanoChat 数据集评估 (计算 PPL)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

{
    echo "----------------------------------------"
    echo "任务 1: NanoChat PPL"
    echo "----------------------------------------"
} >> "$SUMMARY_FILE"

CKPT_PATH="$LATEST_CKPT" \
EVAL_TASK="nanochat" \
LIMIT_TEST_BATCHES=50 \
bash runs/eval_ebt.sh

# 提取结果到汇总文件
RESULT_FILE=$(ls -t ${EVAL_BASE_DIR}/logs/nanochat_*_result.txt 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    cat "$RESULT_FILE" >> "$SUMMARY_FILE"
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

CKPT_PATH="$LATEST_CKPT" \
EVAL_TASK="gsm8k" \
LIMIT_TEST_BATCHES=100 \
BATCH_SIZE=4 \
bash runs/eval_ebt.sh

RESULT_FILE=$(ls -t ${EVAL_BASE_DIR}/logs/gsm8k_*_result.txt 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    cat "$RESULT_FILE" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

echo ""
echo ""

