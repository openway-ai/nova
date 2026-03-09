#!/bin/bash
################################################################################
# NanoChat 分片评估脚本
# 评估训练数据的第一个和最后一个分片 (shard_00000 和 shard_00015)
# 使用与训练相同的方式: 完整序列计算 PPL
#
# 运行位置: /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/runs/
################################################################################

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 激活虚拟环境
source "$EBT_DIR/../../.venv/bin/activate" 2>/dev/null || true
cd "$EBT_DIR"

CONDA_ENV_NAME="nanochat"
CONDA_ENV_PATH="/mnt/shared-storage-user/puyuan/conda_envs/nanochat"
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export OMP_NUM_THREADS=1
export NANOCHAT_OFFLINE_MODE=1

################################################################################
# 配置区域
################################################################################

# CKPT_PATH="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/logs/checkpoints/ebt-small-bs_256_s1_lr_0.0012_2026-03-05_01-01-55_/last.ckpt"
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 设置 checkpoint 路径
CKPT_PATH="${CKPT_PATH:-$EBT_DIR/logs/checkpoints/ebt-small-bs_256_s1_lr_0.0012_2026-03-05_01-01-55_/last.ckpt}"


WANDB_API_KEY="${WANDB_API_KEY:-}"
USE_WANDB="${USE_WANDB:-false}"
GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer}"

# 分片评估特定参数
EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,370}"  # 第一个和最后一个分片
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-50}"  # 每个分片最多评估50个样本

# 文本生成评估参数
ENABLE_GENERATION="${ENABLE_GENERATION:-true}"  # 启用文本生成评估
GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"  # 前半部分作为prompt,后半部分作为target
MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"  # 最小生成长度

################################################################################
# 参数验证
################################################################################


if [ -z "$CKPT_PATH" ]; then
    echo "❌ 错误: 必须指定 checkpoint 路径"
    echo "用法: CKPT_PATH=/path/to/checkpoint.ckpt bash runs/eval_nanochat_shards.sh"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint 文件不存在: $CKPT_PATH"
    exit 1
fi

################################################################################
# 输出目录设置
################################################################################

RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
CKPT_FILENAME=$(basename "$CKPT_PATH" .ckpt)

OUTPUT_DIR="$EBT_DIR/logs/eval/inference/nanochat_shards/${RUN_NAME}/${CKPT_FILENAME}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$EBT_DIR/logs/eval_nanochat_shards_${TIMESTAMP}.log"

################################################################################
# 打印评估信息
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 评估任务: NanoChat 分片评估 (第一个和最后一个分片)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Checkpoint: $CKPT_PATH"
echo "📁 Tokenizer: $TOKENIZER_PATH"
echo "📁 输出目录: $OUTPUT_DIR"
echo "📁 日志文件: $LOG_FILE"
echo "📊 评估分片: $EVAL_SHARD_INDICES"
echo "📊 每分片样本数: $MAX_SAMPLES_PER_SHARD"
echo "🎯 文本生成: $ENABLE_GENERATION"
if [ "$ENABLE_GENERATION" = "true" ]; then
    echo "   - 分割比例: $GENERATION_SPLIT_RATIO (前${GENERATION_SPLIT_RATIO}作为prompt)"
    echo "   - 最小长度: $MIN_GENERATION_LENGTH tokens"
fi
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

echo ""
echo "🚀 开始评估..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

################################################################################
# 运行评估
################################################################################

# 设置生成评估标志
if [ "$ENABLE_GENERATION" = "true" ]; then
    GENERATION_FLAGS="--enable_nanochat_generation --generation_split_ratio $GENERATION_SPLIT_RATIO --min_generation_length $MIN_GENERATION_LENGTH"
else
    GENERATION_FLAGS=""
fi

{
python train.py \
--only_test \
--only_test_model_ckpt "$CKPT_PATH" \
--execution_mode "inference" \
\
--dataset_name "nanochat_shard_eval" \
--context_length 256 \
--tokenizer "$TOKENIZER_PATH" \
--eval_shard_indices "$EVAL_SHARD_INDICES" \
--max_samples_per_shard "$MAX_SAMPLES_PER_SHARD" \
$GENERATION_FLAGS \
\
--infer_max_gen_len 256 \
--infer_temp 0.6 \
--infer_topp 0.9 \
\
--gpus "$GPUS" \
--batch_size_per_device "$BATCH_SIZE" \
--limit_test_batches "$LIMIT_TEST_BATCHES" \
--num_workers 0 \
\
--infer_output_dir "$EBT_DIR/logs/eval/inference" \
$WANDB_FLAGS \
\
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

    echo "📈 PPL 指标:"
    grep -A 10 "TEST EPOCH SUMMARY" "$LOG_FILE" | tail -11 || echo "  未找到汇总指标"

    echo ""
    echo "📁 输出文件:"
    if [ -f "${OUTPUT_DIR}/results.jsonl" ]; then
        RESULT_COUNT=$(wc -l < "${OUTPUT_DIR}/results.jsonl")
        echo "  - 评估结果: ${OUTPUT_DIR}/results.jsonl ($RESULT_COUNT 条)"
    else
        echo "  - 评估结果: 未生成"
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
