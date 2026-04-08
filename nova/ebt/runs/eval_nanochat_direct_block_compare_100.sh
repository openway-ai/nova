#!/bin/bash
################################################################################
# NanoChat 100样本 block-size 对比评测（单卡）
# 对比六种模式：
#   1) sequential (token-by-token, k=1)
#   2) direct_block, K=2
#   3) direct_block, K=4
#   4) direct_block, K=8
#   5) direct_block, K=128
#   6) direct_block, K=256
################################################################################

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$EBT_DIR"

CKPT_PATH="${CKPT_PATH:-/mnt/shared-storage-user/lixueyan/nar/ckpt/last.ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/lixueyan/nar/tokenizer}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,15}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-50}"   # 2 shards * 50 = 100 samples
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"
ENABLE_GENERATION="${ENABLE_GENERATION:-true}"
INFER_MAX_GEN_LEN="${INFER_MAX_GEN_LEN:-256}"
INFER_BLOCK_REFINE_STEPS="${INFER_BLOCK_REFINE_STEPS:-1}"
INFER_BLOCK_INIT_LOGIT_SCALE="${INFER_BLOCK_INIT_LOGIT_SCALE:-8.0}"
GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"
MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${EVAL_RUN_DIR:-$EBT_DIR/logs/eval/direct_block_compare_100_${RUN_TS}}"
mkdir -p "$BASE_OUT"

if [ ! -f "$CKPT_PATH" ]; then
  echo "❌ Checkpoint 不存在: $CKPT_PATH"
  exit 1
fi

CKPT_EBT_TYPE=$(python3 - "$CKPT_PATH" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
hparams = ckpt.get("hyper_parameters", {})
print(hparams.get("ebt_type", "default"))
PY
)

if [ "$CKPT_EBT_TYPE" != "default" ] && [ "$CKPT_EBT_TYPE" != "time_embed" ]; then
  echo "❌ 当前checkpoint不支持 direct_block 对比：ebt_type=$CKPT_EBT_TYPE"
  echo "   direct_block 目前仅支持 ebt_type=default 或 time_embed。"
  echo "   你可以："
  echo "   1) 换一个 ebt_type=default 的 checkpoint；或"
  echo "   2) 继续只跑 sequential / draft-then-refine 对比（不是 direct_block）。"
  exit 2
fi

run_one_mode() {
  local name="$1"
  local mode="$2"
  local block_size="$3"
  local use_refine="$4"

  local mode_out="$BASE_OUT/$name"
  local mode_log="$BASE_OUT/${name}.log"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 运行模式: $name"
  echo "   mode=$mode block_size=$block_size refine=$use_refine"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  (
    export CKPT_PATH TOKENIZER_PATH NANOCHAT_BASE_DIR
    export GPUS BATCH_SIZE CONTEXT_LENGTH EVAL_SHARD_INDICES MAX_SAMPLES_PER_SHARD LIMIT_TEST_BATCHES
    export ENABLE_GENERATION INFER_MAX_GEN_LEN GENERATION_SPLIT_RATIO MIN_GENERATION_LENGTH
    export INFER_BLOCK_MODE="$mode"
    export INFER_BLOCK_SIZE="$block_size"
    export INFER_BLOCK_USE_REFINE="$use_refine"
    export INFER_BLOCK_REFINE_STEPS
    export INFER_BLOCK_INIT_LOGIT_SCALE
    export INFER_BLOCK_DIAGNOSE=false
    export EVAL_RUN_DIR="$mode_out"
    export NANOCHAT_EVAL_LOG="$mode_log"

    bash "$SCRIPT_DIR/eval_nanochat_shards.sh"
  )
}

run_one_mode "sequential_k1" "sequential" "1" "false"
run_one_mode "direct_block_k2" "direct_block" "2" "true"
run_one_mode "direct_block_k4" "direct_block" "4" "true"
run_one_mode "direct_block_k8" "direct_block" "8" "true"
run_one_mode "direct_block_k128" "direct_block" "128" "true"
run_one_mode "direct_block_k256" "direct_block" "256" "true"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 汇总对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 - "$BASE_OUT" <<'PY'
import json
import statistics
import sys
from pathlib import Path
from collections import Counter

base = Path(sys.argv[1])
modes = [
    "sequential_k1",
    "direct_block_k2",
    "direct_block_k4",
    "direct_block_k8",
    "direct_block_k128",
    "direct_block_k256",
]

def load_records(mode_name):
    mode_dir = base / mode_name
    files = sorted(mode_dir.rglob("results.jsonl"))
    if not files:
        return []
    recs = []
    with files[0].open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs

def mean(vals):
    return statistics.fmean(vals) if vals else None

def safefmt(x):
    return "N/A" if x is None else f"{x:.4f}"

all_data = {m: load_records(m) for m in modes}
baseline = all_data["sequential_k1"]

print(f"Base output dir: {base}")
print()

print("[teacher_forced_ppl]")
print("  NOTE: teacher-forced 指标来自 get_ppl()，与 free-running 生成质量分开看。")
for mode in modes:
    recs = all_data[mode]
    losses = [float(r["teacher_forced_loss"]) for r in recs if "teacher_forced_loss" in r]
    ppls = [float(r["teacher_forced_ppl"]) for r in recs if "teacher_forced_ppl" in r]
    bpbs = [float(r["teacher_forced_bpb"]) for r in recs if "teacher_forced_bpb" in r]
    print(
        f"  [{mode}] samples={len(recs)} "
        f"teacher_forced_loss={safefmt(mean(losses))} "
        f"teacher_forced_ppl={safefmt(mean(ppls))} "
        f"teacher_forced_bpb={safefmt(mean(bpbs))}"
    )
print()

print("[free_running_generation_metrics]")
def char_overlap(a, b):
    if not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    return inter / max(1, len(b))

for mode in modes:
    recs = all_data[mode]
    dts = [float(r["decode_time_sec"]) for r in recs if "decode_time_sec" in r]
    eb = [float(r["block_energy_before_refine"]) for r in recs if "block_energy_before_refine" in r]
    ea = [float(r["block_energy_after_refine"]) for r in recs if "block_energy_after_refine" in r]
    lens = [len(r.get("generation", "")) for r in recs if "generation" in r]
    target_overlaps = []
    for r in recs:
        if "generation" in r and "target" in r:
            target_overlaps.append(char_overlap(r.get("generation", ""), r.get("target", "")))
    print(f"[{mode}]")
    print(f"  samples: {len(recs)}")
    print(f"  decode_time_sec_mean: {safefmt(mean(dts))}")
    print(f"  generation_char_len_mean: {safefmt(mean(lens))}")
    print(f"  generation_char_overlap_vs_target: {safefmt(mean(target_overlaps))}")
    print(f"  block_energy_before_mean: {safefmt(mean(eb))}")
    print(f"  block_energy_after_mean: {safefmt(mean(ea))}")
    print()

if baseline:
    print("[vs_sequential_generation_metrics]")
    base_gens = [r.get("generation", "") for r in baseline]
    for mode in modes[1:]:
        recs = all_data[mode]
        n = min(len(base_gens), len(recs))
        if n == 0:
            print(f"  {mode}: N/A")
            continue
        exact = 0
        prefix = []
        for i in range(n):
            g0 = base_gens[i]
            g1 = recs[i].get("generation", "")
            if g0 == g1:
                exact += 1
            match = 0
            for a, b in zip(g0, g1):
                if a != b:
                    break
                match += 1
            denom = max(1, len(g0))
            prefix.append(match / denom)
        print(
            f"  {mode}: "
            f"generation_exact_match_vs_sequential={exact/n:.4f}, "
            f"generation_prefix_char_match_vs_sequential={statistics.fmean(prefix):.4f}, "
            f"paired_n={n}"
        )
PY

echo ""
echo "✅ 完成。结果目录: $BASE_OUT"
