#!/bin/bash
################################################################################
# EBT 模型评估脚本
#
# 接收来自父脚本的环境变量，无需重复配置
# 可独立运行：CKPT_PATH=/path/to/ckpt EVAL_TASK=gsm8k bash runs/eval_ebt.sh
################################################################################

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 激活虚拟环境并设置环境
source "$EBT_DIR/../../.venv/bin/activate" 2>/dev/null || true
cd "$EBT_DIR"

export NANOCHAT_BASE_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"
export OMP_NUM_THREADS=1
export NANOCHAT_OFFLINE_MODE=1

################################################################################
# 参数 (优先使用环境变量，否则使用默认值)
################################################################################

CKPT_PATH="${CKPT_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
EVAL_TASK="${EVAL_TASK:-nanochat}"

################################################################################
# 参数验证
################################################################################

if [ -z "$CKPT_PATH" ]; then
    echo "❌ 错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt EVAL_TASK=gsm8k bash runs/eval_ebt.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

################################################################################
# 任务配置
################################################################################

case $EVAL_TASK in
    "nanochat")
        DATASET_NAME="nanochat"
        EXECUTION_MODE="inference"
        TASK_DESC="NanoChat 数据集 (PPL 测试)"
        ;;
    "gsm8k")
        DATASET_NAME="gsm8k"
        EXECUTION_MODE="inference"
        TASK_DESC="GSM8K (数学推理)"
        ;;
    "arc")
        DATASET_NAME="arc"
        EXECUTION_MODE="inference"
        TASK_DESC="ARC (科学问答)"
        ;;
    "humaneval")
        DATASET_NAME="humaneval"
        EXECUTION_MODE="inference"
        TASK_DESC="HumanEval (代码生成)"
        ;;
    "mmlu")
        DATASET_NAME="mmlu"
        EXECUTION_MODE="inference"
        TASK_DESC="MMLU (多任务理解)"
        ;;
    "smoltalk")
        DATASET_NAME="smoltalk"
        EXECUTION_MODE="inference"
        TASK_DESC="SmolTalk (对话)"
        ;;
    "spellingbee")
        DATASET_NAME="spellingbee"
        EXECUTION_MODE="inference"
        TASK_DESC="SpellingBee"
        ;;
    *)
        echo "❌ 错误: 未知任务类型: $EVAL_TASK"
        exit 1
        ;;
esac

################################################################################
# 输出目录设置
################################################################################

RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
OUTPUT_DIR="$EBT_DIR/logs/eval/inference/${DATASET_NAME}/${RUN_NAME}/${CKPT_FILENAME}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$EBT_DIR/logs/eval/inference/${DATASET_NAME}/${RUN_NAME}/eval_${EVAL_TASK}_${TIMESTAMP}.log"

################################################################################
# 打印评估信息
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 评估任务: $TASK_DESC"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Checkpoint: $CKPT_PATH"
echo "📁 Tokenizer: $TOKENIZER_PATH"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📁 日志文件: $LOG_FILE"

################################################################################
# WandB 配置
################################################################################

WANDB_FLAGS="--no_wandb"
if [ "$USE_WANDB" = "true" ] && [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY="$WANDB_API_KEY"
    WANDB_FLAGS="--wandb_project ebt_eval --wandb_entity '' --wandb_offline"
    echo "📊 WandB: 启用 (离线模式)"
else
    echo "📊 WandB: 禁用"
fi

echo ""
echo "🚀 开始评估..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# 运行评估
################################################################################

{
python train.py \
    --only_test \
    --only_test_model_ckpt "$CKPT_PATH" \
    --execution_mode "$EXECUTION_MODE" \
    --dataset_name "$DATASET_NAME" \
    --context_length 256 \
    --tokenizer "$TOKENIZER_PATH" \
    --infer_max_gen_len 256 \
    --infer_temp 0.6 \
    --infer_topp 0.9 \
    --gpus "$GPUS" \
    --batch_size_per_device "$BATCH_SIZE" \
    --limit_test_batches "$LIMIT_TEST_BATCHES" \
    --num_workers 4 \
    --infer_output_dir "$EBT_DIR/logs/eval/inference" \
    $WANDB_FLAGS \
    --val_sanity 0 \
    --val_every_n_step 1000
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

################################################################################
# 显示结果摘要
################################################################################

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 评估结果摘要"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 状态: 成功"
    echo ""
    if [ "$EVAL_TASK" = "nanochat" ]; then
        echo "📈 PPL 指标:"
        grep -A 5 "TEST EPOCH SUMMARY" "$LOG_FILE" | tail -6 || echo "  未找到汇总指标"
    else
        echo "📈 推理结果:"
        grep -E "Accuracy|acc|score|result" "$LOG_FILE" | tail -5 || echo "  未找到评估指标"
    fi
    echo ""
    echo "📁 输出文件:"
    if [ -f "${OUTPUT_DIR}/results.jsonl" ]; then
        RESULT_COUNT=$(wc -l < "${OUTPUT_DIR}/results.jsonl")
        echo "  - 生成轨迹: ${OUTPUT_DIR}/results.jsonl ($RESULT_COUNT 条)"
    else
        echo "  - 生成轨迹: 未生成"
    fi
    echo "  - 完整日志: $LOG_FILE"
else
    echo "❌ 状态: 失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "📋 最后20行错误:"
    tail -20 "$LOG_FILE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
