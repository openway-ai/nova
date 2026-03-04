#!/bin/bash
################################################################################
# EBT 模型评估脚本
# 支持在 nanochat 数据集和 OOD 任务上评估训练好的 checkpoint
################################################################################

# 激活虚拟环境
source .venv/bin/activate
cd nova/ebt

################################################################################
# 配置区域 - 请根据实际情况修改
################################################################################

# 1. Checkpoint 路径（必填）
# 示例: "../../logs/checkpoints/ebt-xxs-bs_256_s1_lr_0.0012_2024-03-04_22-30-15_/epoch=10-step=50000-valid_loss=2.3456.ckpt"
CKPT_PATH="${CKPT_PATH:-}"

# 2. 评估任务类型（必填）
# 选项: "nanochat" | "gsm8k" | "arc" | "humaneval" | "mmlu" | "smoltalk" | "spellingbee"
EVAL_TASK="${EVAL_TASK:-nanochat}"

# 3. WandB 配置（可选）
WANDB_API_KEY="${WANDB_API_KEY:-}"
USE_WANDB="${USE_WANDB:-false}"  # true 或 false

# 4. GPU 配置
GPUS="${GPUS:-1}"  # 使用的 GPU 数量或 ID，例如 "1" 或 "[0,1]"

# 5. Batch size（根据 GPU 内存调整）
BATCH_SIZE="${BATCH_SIZE:-1}"

# 6. 测试样本数量（-1 表示全部）
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"

################################################################################
# 参数验证
################################################################################

if [ -z "$CKPT_PATH" ]; then
    echo "❌ 错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt bash runs/eval_ebt.sh"
    echo "或者: CKPT_PATH=/path/to/checkpoint.ckpt EVAL_TASK=gsm8k bash runs/eval_ebt.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

################################################################################
# 任务特定配置
################################################################################

case $EVAL_TASK in
    "nanochat")
        DATASET_NAME="nanochat"
        EXECUTION_MODE="pretrain"  # nanochat 不是 inference 任务
        echo "📊 评估任务: NanoChat 数据集 (PPL 测试)"
        ;;
    "gsm8k")
        DATASET_NAME="gsm8k"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: GSM8K (数学推理)"
        ;;
    "arc")
        DATASET_NAME="arc"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: ARC (科学问答)"
        ;;
    "humaneval")
        DATASET_NAME="humaneval"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: HumanEval (代码生成)"
        ;;
    "mmlu")
        DATASET_NAME="mmlu"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: MMLU (多任务理解)"
        ;;
    "smoltalk")
        DATASET_NAME="smoltalk"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: SmolTalk (对话)"
        ;;
    "spellingbee")
        DATASET_NAME="spellingbee"
        EXECUTION_MODE="inference"
        echo "📊 评估任务: SpellingBee"
        ;;
    *)
        echo "❌ 错误: 未知任务类型: $EVAL_TASK"
        echo "支持的任务: nanochat | gsm8k | arc | humaneval | mmlu | smoltalk | spellingbee"
        exit 1
        ;;
esac

################################################################################
# 输出目录
################################################################################

RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)
OUTPUT_DIR="../../logs/inference/NLP/${DATASET_NAME}/${RUN_NAME}/${CKPT_FILENAME}"

echo "📁 Checkpoint: $CKPT_PATH"
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

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

################################################################################
# 运行评估
################################################################################

echo ""
echo "🚀 开始评估..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python train.py \
--only_test \
--only_test_model_ckpt "$CKPT_PATH" \
--execution_mode "$EXECUTION_MODE" \
\
--dataset_name "$DATASET_NAME" \
--context_length 256 \
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--infer_max_gen_len 256 \
--infer_temp 0.6 \
--infer_topp 0.9 \
--infer_echo false \
\
--gpus "$GPUS" \
--batch_size_per_device "$BATCH_SIZE" \
--limit_test_batches "$LIMIT_TEST_BATCHES" \
--num_workers 4 \
\
--infer_output_dir "../../logs/inference" \
$WANDB_FLAGS \
\
--val_sanity 0

################################################################################
# 评估完成
################################################################################

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 评估完成！"
    echo ""
    echo "📊 结果文件:"
    if [ "$EXECUTION_MODE" = "inference" ]; then
        echo "   - 生成文本: ${OUTPUT_DIR}/results.jsonl"
        echo "   - 评估指标: 查看终端输出或 WandB"
    else
        echo "   - PPL 指标: 查看终端输出或 WandB"
    fi
    echo ""
    echo "📁 输出目录: $OUTPUT_DIR"
else
    echo "❌ 评估失败，退出码: $EXIT_CODE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
