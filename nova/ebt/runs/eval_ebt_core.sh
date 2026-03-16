#!/bin/bash
################################################################################
# EBT 模型 CORE 评估脚本
# 基于 nanochat 的 base_eval 逻辑，适配 EBT 模型
################################################################################

set -e  # 遇到错误立即退出

# =============================================================================
# 配置参数
# =============================================================================

# 环境配置
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_OFFLINE_MODE=1
export OMP_NUM_THREADS=1

# CKPT_PATH="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/logs/checkpoints/ebt-small-bs_256_s1_lr_0.0012_2026-03-05_01-01-55_/last.ckpt"
export CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt}"

# EBT checkpoint 路径（必须指定）
CKPT_PATH="${CKPT_PATH:-}"
if [ -z "$CKPT_PATH" ]; then
    echo "❌ 错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt bash runs/eval_ebt_core.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

# 评估配置
EVAL_MODES="${EVAL_MODES:-core}"  # 默认只评估 CORE，可选 core,bpb,sample

# 样本数配置
# MAX_PER_TASK: 全局默认值，-1 表示评估所有样本（获得完整 CORE score）
# 可以通过环境变量覆盖，例如: MAX_PER_TASK=100 bash runs/eval_ebt_core.sh
MAX_PER_TASK="${MAX_PER_TASK:--1}"

# 每个任务的样本数配置（可选，覆盖全局设置）
# 格式: "task_name:num_samples,task_name:num_samples,..."
# 例如: TASK_SAMPLES="hellaswag_zeroshot:100,arc_easy:50"
TASK_SAMPLES="${TASK_SAMPLES:-}"

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"

# GPU 配置
# NUM_GPUS: 使用的 GPU 数量
# -1 表示使用所有检测到的 GPU（默认）
# 可以指定具体数量，例如: NUM_GPUS=2
NUM_GPUS="${NUM_GPUS:--1}"

# 自动检测可用 GPU 数量
if [ "$NUM_GPUS" = "-1" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        NUM_GPUS=$DETECTED_GPUS
        echo "🔍 自动检测到 $NUM_GPUS 个 GPU"
    else
        NUM_GPUS=1
        echo "⚠️  无法检测 GPU，默认使用 1 个 GPU"
    fi
fi

# Tokenizer 路径
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"

# 输出配置
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
OUTPUT_DIR="../../nova/ebt/logs/core_eval/${RUN_NAME}/${CKPT_FILENAME}"
LOG_FILE="../../nova/ebt/logs/core_eval_${TIMESTAMP}.log"

# 创建输出目录
cd /mnt/shared-storage-user/puyuan/code/nova/nova/ebt
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir -p "logs_core"
cd ../..

# =============================================================================
# 检查环境准备
# =============================================================================

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  EBT 模型 CORE 评估"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📦 Checkpoint: $CKPT_PATH"
echo "📦 Tokenizer: $TOKENIZER_PATH"
echo "📊 评估模式: $EVAL_MODES"
echo "📊 全局最大样本数: $MAX_PER_TASK (-1 = 全部样本)"
if [ -n "$TASK_SAMPLES" ]; then
    echo "📊 任务特定样本数: $TASK_SAMPLES"
fi
echo "📊 Batch Size: $DEVICE_BATCH_SIZE"
echo "📊 GPU 数量: $NUM_GPUS"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📁 日志文件: $LOG_FILE"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 检查 eval_bundle 是否存在
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    echo "❌ 错误: 评估数据集 (eval_bundle) 不存在"
    echo "请先下载评估数据集:"
    echo "  cd /mnt/shared-storage-user/puyuan/code/nanochat"
    echo "  bash runs/download_eval_bundle.sh"
    exit 1
fi

echo "✅ 评估数据集已就绪: $NANOCHAT_BASE_DIR/eval_bundle"
echo ""

# 检查 tokenizer 是否存在
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "❌ 错误: Tokenizer 不存在: $TOKENIZER_PATH"
    exit 1
fi

echo "✅ Tokenizer 已就绪"
echo ""

# =============================================================================
# 激活环境
# =============================================================================

# echo "🔧 激活 conda 环境..."
# # 使用 conda 环境中的 Python
# conda activate /mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/activate

# echo "✅ 环境已激活"
# echo ""

# =============================================================================
# 运行 CORE 评估
# =============================================================================

echo "🚀 开始 CORE 评估..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# 构建评估命令
cd nova/ebt

# 构建任务样本数参数
TASK_SAMPLES_ARG=""
if [ -n "$TASK_SAMPLES" ]; then
    TASK_SAMPLES_ARG="--task-samples '$TASK_SAMPLES'"
fi

# 使用 conda 环境的 Python
EVAL_CMD="python -m scripts.ebt_core_eval \
    --ckpt-path '$CKPT_PATH' \
    --tokenizer-path '$TOKENIZER_PATH' \
    --eval-bundle-dir '$NANOCHAT_BASE_DIR/eval_bundle' \
    --eval-modes '$EVAL_MODES' \
    --max-per-task $MAX_PER_TASK \
    $TASK_SAMPLES_ARG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --output-dir '$OUTPUT_DIR' \
    --gpus $NUM_GPUS"

echo "执行命令:"
echo "$EVAL_CMD"
echo ""

# 执行评估（将输出同时保存到日志和显示在屏幕）
{
    eval $EVAL_CMD
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

cd ../..

# =============================================================================
# 显示结果摘要
# =============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  CORE 评估结果摘要"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 状态: 成功"
    echo ""

    # 提取 CORE 分数
    CORE_SCORE=$(grep "CORE Metric:" "$LOG_FILE" | tail -1 | awk '{print $3}')
    if [ ! -z "$CORE_SCORE" ]; then
        echo "📊 CORE 分数: $CORE_SCORE"
    fi
    echo ""

    # 显示各任务结果
    echo "📋 各任务详细结果:"
    grep -A 100 "Task Results:" "$LOG_FILE" | grep -E "^\s+[a-z_]+:" | head -20 || echo "  未找到详细结果"
    echo ""

    # 输出文件
    echo "📁 输出文件:"
    if [ -f "$OUTPUT_DIR/core_results.json" ]; then
        echo "  - CORE 结果: $OUTPUT_DIR/core_results.json"
    fi
    if [ -f "$OUTPUT_DIR/core_results.csv" ]; then
        echo "  - CSV 报告: $OUTPUT_DIR/core_results.csv"
    fi
    echo "  - 完整日志: $LOG_FILE"

else
    echo "❌ 状态: 失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "📋 错误信息:"
    tail -30 "$LOG_FILE"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

exit $EXIT_CODE
