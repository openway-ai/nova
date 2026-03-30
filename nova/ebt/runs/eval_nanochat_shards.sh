#!/bin/bash
################################################################################
# NanoChat 分片评估脚本
#
# 接收来自父脚本的环境变量，无需重复配置
# 可独立运行：CKPT_PATH=/path/to/ckpt bash runs/eval_nanochat_shards.sh
################################################################################

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到 EBT 目录
cd "$EBT_DIR"

# 确定 Python 路径：优先使用 conda 环境的 python，避免误用系统 python
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [ -x "$(command -v python3)" ]; then
    PYTHON="python3"
else
    PYTHON="python"
fi
echo "🐍 Python: $PYTHON ($($PYTHON --version 2>&1))"

export NANOCHAT_BASE_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"
export OMP_NUM_THREADS=1
export NANOCHAT_OFFLINE_MODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

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

# 分片评估特定参数
EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,15}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-50}"
ENABLE_GENERATION="${ENABLE_GENERATION:-true}"
GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"
MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"

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
# 输出目录与日志
################################################################################

# 推理输出目录: 优先使用父脚本传入的 EVAL_RUN_DIR，否则自行生成
if [ -n "$EVAL_RUN_DIR" ]; then
    INFER_OUTPUT_DIR="$EVAL_RUN_DIR"
else
    RUN_NAME=$(basename $(dirname "$CKPT_PATH"))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_SHORT=$(echo "$RUN_NAME" | sed 's/_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}_\?$//')
    INFER_OUTPUT_DIR="$EBT_DIR/logs/eval/${RUN_SHORT}_${TIMESTAMP}"
fi
mkdir -p "$INFER_OUTPUT_DIR"

# 日志路径: 优先使用父脚本传入的 NANOCHAT_EVAL_LOG，否则自行生成
if [ -n "$NANOCHAT_EVAL_LOG" ]; then
    LOG_FILE="$NANOCHAT_EVAL_LOG"
    mkdir -p "$(dirname "$LOG_FILE")"
else
    LOG_FILE="$INFER_OUTPUT_DIR/nanochat_shards.log"
fi

################################################################################
# 打印评估信息
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 评估任务: NanoChat 分片评估"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Checkpoint: $CKPT_PATH"
echo "📁 Tokenizer: $TOKENIZER_PATH"
echo "📁 输出目录: $INFER_OUTPUT_DIR"
echo "📁 日志文件: $LOG_FILE"
echo "📊 评估分片: $EVAL_SHARD_INDICES"
echo "📊 每分片样本数: $MAX_SAMPLES_PER_SHARD"
echo "🎯 文本生成: $ENABLE_GENERATION"
if [ "$ENABLE_GENERATION" = "true" ]; then
    echo "   - 分割比例: $GENERATION_SPLIT_RATIO"
    echo "   - 最小长度: $MIN_GENERATION_LENGTH tokens"
fi

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

GENERATION_FLAGS=""
if [ "$ENABLE_GENERATION" = "true" ]; then
    GENERATION_FLAGS="--enable_nanochat_generation --generation_split_ratio $GENERATION_SPLIT_RATIO --min_generation_length $MIN_GENERATION_LENGTH"
fi

{
$PYTHON train.py \
    --only_test \
    --only_test_model_ckpt "$CKPT_PATH" \
    --execution_mode "inference" \
    --dataset_name "nanochat_shard_eval" \
    --context_length 256 \
    --tokenizer "$TOKENIZER_PATH" \
    --eval_shard_indices "$EVAL_SHARD_INDICES" \
    --max_samples_per_shard "$MAX_SAMPLES_PER_SHARD" \
    $GENERATION_FLAGS \
    --infer_max_gen_len 256 \
    --infer_temp 0.6 \
    --infer_topp 0.9 \
    --gpus "$GPUS" \
    --batch_size_per_device "$BATCH_SIZE" \
    --limit_test_batches "$LIMIT_TEST_BATCHES" \
    --num_workers 0 \
    --infer_output_dir "$INFER_OUTPUT_DIR" \
    --set_matmul_precision "medium" \
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
echo "📊 NanoChat 评估结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 状态: 成功"
    echo ""
    echo "📈 PPL 指标:"
    grep -A 20 "EVALUATION RESULTS SUMMARY" "$LOG_FILE" | tail -21 || echo "  未找到日志汇总指标"
    echo ""
    echo "📁 输出文件:"
    # Python 实际输出路径: $INFER_OUTPUT_DIR/NLP/nanochat_shard_eval/<run_name>/<ckpt_name>/
    RESULT_FILE=$(find "$INFER_OUTPUT_DIR" -name "results.jsonl" -path "*/nanochat_shard_eval/*" 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        RESULT_COUNT=$(wc -l < "$RESULT_FILE")
        echo "  - 评估结果: $RESULT_FILE ($RESULT_COUNT 条)"
        echo "  - 统计信息 (from results.jsonl):"
        python3 - <<'PY' "$RESULT_FILE" | sed 's/^/    /'
import json
import statistics
import sys

path = sys.argv[1]
losses = []
ppls = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "loss" in rec:
            losses.append(float(rec["loss"]))
        if "ppl" in rec:
            ppls.append(float(rec["ppl"]))

def fmt_stats(name, vals):
    if not vals:
        print(f"{name}: N/A")
        return
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"{name}: mean={mean:.4f} std={std:.4f} min={min(vals):.4f} max={max(vals):.4f} n={len(vals)}")

fmt_stats("loss", losses)
fmt_stats("ppl", ppls)
PY
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
