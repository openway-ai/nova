#!/bin/bash
################################################################################
# Spec-decoding-style eval for the (TF-)blockwise checkpoints.
#
# Runs the SAME checkpoint twice on the SAME prompts:
#   1) sequential_k1   — AR baseline (acts as the "verifier" oracle)
#   2) direct_block_k2 — blockwise drafter, K=2 parallel offsets
#
# Then computes (in the post-processing python at the bottom):
#   * teacher_forced_loss / ppl / bpb        (quality, temperature-independent)
#   * decode_time_sec_mean                   (wall-clock end-to-end)
#   * speedup = sequential_time / block_time (how much faster the K=2 path is)
#   * token-level prefix match vs sequential (overall + per-offset)
#       this is the "accept rate" — fraction of positions where block_k2's
#       greedy argmax token equals sequential's greedy argmax token.
#
# Temperature is forced to 0 so accept-rate is well-defined. Refine is OFF
# by default (no MCMC on top of the head; we want to measure the head's
# own quality, not extra refinement).
#
# Usage:
#   CKPT_PATH=.../ebt-xxs-blockwise-k2-tfmtphead/last.ckpt \
#   BLOCK_MODE=blockwise \
#   bash runs/eval_nanochat_tf_spec.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$EBT_DIR"

CKPT_PATH="${CKPT_PATH:?must set CKPT_PATH=/path/to/last.ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/lixueyan/nar/tokenizer}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"
BLOCK_MODE="${BLOCK_MODE:-blockwise}"

GPUS="${GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
EVAL_SHARD_INDICES="${EVAL_SHARD_INDICES:-0,15}"
MAX_SAMPLES_PER_SHARD="${MAX_SAMPLES_PER_SHARD:-50}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-100}"
ENABLE_GENERATION="${ENABLE_GENERATION:-true}"
INFER_MAX_GEN_LEN="${INFER_MAX_GEN_LEN:-256}"
INFER_BLOCK_REFINE_STEPS="${INFER_BLOCK_REFINE_STEPS:-0}"   # 0 = no MCMC refine
INFER_BLOCK_INIT_LOGIT_SCALE="${INFER_BLOCK_INIT_LOGIT_SCALE:-8.0}"
GENERATION_SPLIT_RATIO="${GENERATION_SPLIT_RATIO:-0.5}"
MIN_GENERATION_LENGTH="${MIN_GENERATION_LENGTH:-64}"
# K of the drafter — must match the ckpt's train_block_size.
BLOCK_K="${BLOCK_K:-2}"
# Temperature: 0 makes accept-rate well-defined (deterministic per prompt).
INFER_TEMP="${INFER_TEMP:-0.0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(basename "$(dirname "$CKPT_PATH")")"
BASE_OUT="${EVAL_RUN_DIR:-$EBT_DIR/logs/eval/tf_spec_${CKPT_TAG}_${RUN_TS}}"
mkdir -p "$BASE_OUT"

run_one_mode() {
  local name="$1"
  local mode="$2"
  local block_size="$3"
  local use_refine="$4"

  local mode_out="$BASE_OUT/$name"
  local mode_log="$BASE_OUT/${name}.log"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "🚀 mode=$name | $mode | block_size=$block_size | refine=$use_refine"
  echo "   ckpt=$CKPT_PATH"
  echo "   block_mode=$BLOCK_MODE  temp=$INFER_TEMP"
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
    export BLOCK_MODE
    # eval_nanochat_shards.sh now honors INFER_TEMP / INFER_TOPP (defaults
    # are 0.6 / 0.9; we set INFER_TEMP=0 for accept-rate determinism).
    export INFER_TEMP
    export INFER_TOPP="${INFER_TOPP:-0.9}"

    bash "$SCRIPT_DIR/eval_nanochat_shards.sh"
  )
}

run_one_mode "sequential_k1"    "sequential"   "1"        "false"
run_one_mode "direct_block_k${BLOCK_K}" "direct_block" "$BLOCK_K" "false"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SPEC-EVAL AGGREGATE — ckpt=$CKPT_TAG  K=$BLOCK_K  temp=$INFER_TEMP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHON_BIN="${PYTHON:-/mnt/shared-storage-user/lixueyan/miniconda3/envs/nanochat/bin/python}"
TOKENIZER_DIR="$TOKENIZER_PATH" BASE_OUT_DIR="$BASE_OUT" BLOCK_K="$BLOCK_K" \
"$PYTHON_BIN" - <<'PY'
import json, os, statistics, sys
from pathlib import Path

base = Path(os.environ["BASE_OUT_DIR"])
K = int(os.environ.get("BLOCK_K", "2"))
tokenizer_dir = os.environ.get("TOKENIZER_DIR", "")

modes = ["sequential_k1", f"direct_block_k{K}"]

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

def fmt(x, p=4):
    return "N/A" if x is None else f"{x:.{p}f}"

data = {m: load_records(m) for m in modes}

# ---- 1) teacher-forced quality (unchanged across temp; cross-checks training) ----
print()
print("[teacher_forced_quality]")
for m in modes:
    recs = data[m]
    tf_loss = [float(r["teacher_forced_loss"]) for r in recs if "teacher_forced_loss" in r]
    tf_ppl  = [float(r["teacher_forced_ppl"])  for r in recs if "teacher_forced_ppl"  in r]
    tf_bpb  = [float(r["teacher_forced_bpb"])  for r in recs if "teacher_forced_bpb"  in r]
    print(
        f"  [{m}] n={len(recs)} loss={fmt(mean(tf_loss))} ppl={fmt(mean(tf_ppl))} bpb={fmt(mean(tf_bpb))}"
    )
    # Per-offset teacher-forced metrics (offset_1, offset_2, ... offset_K)
    # forwarded from get_ppl via results.jsonl. Only printed when present
    # (chunked-loop path on explicit-block-latent ckpts).
    per_offset_keys = sorted({k for r in recs for k in r if k.startswith("offset_") and (
        k.endswith("_loss") or k.endswith("_ppl") or k.endswith("_bpb")
    )})
    if per_offset_keys:
        # Group by offset index for readability: offset_1: loss/ppl/bpb, offset_2: ...
        import re as _re
        offset_idx_to_metrics = {}
        for k in per_offset_keys:
            mobj = _re.match(r"^offset_(\d+)_(loss|ppl|bpb)$", k)
            if not mobj:
                continue
            j = int(mobj.group(1)); name = mobj.group(2)
            offset_idx_to_metrics.setdefault(j, {})[name] = k
        for j in sorted(offset_idx_to_metrics):
            d_ = offset_idx_to_metrics[j]
            vals = {}
            for name in ("loss", "ppl", "bpb"):
                if name in d_:
                    arr = [float(r[d_[name]]) for r in recs if d_[name] in r]
                    vals[name] = mean(arr)
            print(
                f"    offset_{j}: loss={fmt(vals.get('loss'))} ppl={fmt(vals.get('ppl'))} bpb={fmt(vals.get('bpb'))}"
            )

# ---- 2) wall-clock + speedup ----
print()
print("[wallclock]")
seq_t = mean([float(r["decode_time_sec"]) for r in data[modes[0]] if "decode_time_sec" in r])
blk_t = mean([float(r["decode_time_sec"]) for r in data[modes[1]] if "decode_time_sec" in r])
print(f"  sequential_k1     decode_time_sec_mean = {fmt(seq_t)}")
print(f"  direct_block_k{K}  decode_time_sec_mean = {fmt(blk_t)}")
if seq_t and blk_t:
    print(f"  speedup            = {seq_t / blk_t:.3f}x  (>1 means blockwise faster)")

# ---- 3) accept rate (token-level vs sequential, paired by prompt) ----
# Re-tokenize generation strings; align by position and compute:
#   * overall token-prefix match (length-normalized)
#   * per-offset accept rate (position % K)
#   * blockwise per-block accept rate (longest run of leading accepts within each K-block)
try:
    # nanochat tokenizer — same one the eval pipeline uses. Reads from
    # NANOCHAT_BASE_DIR (set in the parent shell) by default.
    from nanochat.tokenizer import get_tokenizer  # type: ignore
    _tok = get_tokenizer()
    def encode(text):
        ids = _tok.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)
except Exception as e:
    print(f"  WARN: tokenizer load failed ({e}); falling back to char-level prefix.")
    encode = None

seq_recs = data[modes[0]]
blk_recs = data[modes[1]]
n = min(len(seq_recs), len(blk_recs))

overall_match = []
per_offset_matches = [[] for _ in range(K)]
per_block_accept_counts = []   # how many leading tokens in each K-block matched
exact_match_blocks = 0
total_blocks = 0

for i in range(n):
    g_seq = seq_recs[i].get("generation", "")
    g_blk = blk_recs[i].get("generation", "")
    if encode is not None:
        toks_seq = encode(g_seq)
        toks_blk = encode(g_blk)
    else:
        # char fallback
        toks_seq = list(g_seq)
        toks_blk = list(g_blk)
    L = min(len(toks_seq), len(toks_blk))
    if L == 0:
        continue
    matches = [int(toks_seq[j] == toks_blk[j]) for j in range(L)]
    overall_match.append(sum(matches) / L)
    # per-offset position % K (after the prompt, the block_k2 emission is
    # block-aligned starting at index 0 of the generation).
    for j, m in enumerate(matches):
        per_offset_matches[j % K].append(m)
    # per-block leading-accept count
    for b_start in range(0, L - K + 1, K):
        run = 0
        for off in range(K):
            if matches[b_start + off]:
                run += 1
            else:
                break
        per_block_accept_counts.append(run)
        if run == K:
            exact_match_blocks += 1
        total_blocks += 1

print()
print("[accept_rate_vs_sequential]")
print(f"  paired_samples = {n}")
print(f"  overall_token_match (paired prefix, mean across samples) = {fmt(mean(overall_match))}")
for j in range(K):
    mj = mean(per_offset_matches[j])
    print(f"  offset_{j+1}_token_match = {fmt(mj)}  (n_tokens={len(per_offset_matches[j])})")
if total_blocks > 0:
    mean_accept = mean([float(x) for x in per_block_accept_counts])
    print(f"  mean_accepted_tokens_per_block (out of {K}) = {fmt(mean_accept, 3)}")
    print(f"  fraction_of_blocks_with_all_K_accepted     = {exact_match_blocks/total_blocks:.4f}")
    eff_speedup = mean_accept
    print(f"  → effective spec-decoding token throughput multiplier ≈ {fmt(eff_speedup, 3)}x")
    print(f"    (this is the speed gain assuming verifier cost == drafter cost;")
    print(f"     real speedup is wall-clock above, this is the *acceptance ceiling*.)")
print()
print(f"results dir: {base}")
PY

echo ""
echo "✅ done."
