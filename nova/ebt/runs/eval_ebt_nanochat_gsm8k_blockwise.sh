#!/bin/bash
################################################################################
# EBT 评估入口脚本 (Blockwise 版本)
#
# 用法:
#   bash runs/eval_ebt_nanochat_gsm8k_blockwise.sh
#
# 可覆盖参数:
#   INFER_BLOCK_SIZE
#   INFER_BLOCK_USE_REFINE
#   INFER_BLOCK_REFINE_STEPS
#   INFER_BLOCK_INIT_LOGIT_SCALE
#
# 其他参数与 eval_ebt_nanochat_gsm8k.sh 一致，仍可通过环境变量覆盖。
################################################################################

set -e
set -o pipefail

# 默认开启 blockwise 推理；可由外部环境变量覆盖
export INFER_BLOCK_SIZE="${INFER_BLOCK_SIZE:-4}"
export INFER_BLOCK_USE_REFINE="${INFER_BLOCK_USE_REFINE:-true}"
export INFER_BLOCK_REFINE_STEPS="${INFER_BLOCK_REFINE_STEPS:-4}"
export INFER_BLOCK_INIT_LOGIT_SCALE="${INFER_BLOCK_INIT_LOGIT_SCALE:-8.0}"
export PYTHONNOUSERSITE=1
# For NanoChat shard eval, generation in test_step is very expensive with blockwise refine.
# Default to PPL-only there; GSM8K task still runs generation as expected.
export ENABLE_GENERATION="${ENABLE_GENERATION:-false}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🧱 Blockwise 推理配置:"
echo "   INFER_BLOCK_SIZE=$INFER_BLOCK_SIZE"
echo "   INFER_BLOCK_USE_REFINE=$INFER_BLOCK_USE_REFINE"
echo "   INFER_BLOCK_REFINE_STEPS=$INFER_BLOCK_REFINE_STEPS"
echo "   INFER_BLOCK_INIT_LOGIT_SCALE=$INFER_BLOCK_INIT_LOGIT_SCALE"
echo ""

# 复用现有统一评估入口（nanochat shards + gsm8k）
bash "$SCRIPT_DIR/eval_ebt_nanochat_gsm8k.sh"
