#!/usr/bin/env bash
# Runs SDPA attention correctness tests and benchmarks.
#
# Usage (from repo root):
#   bash nova/ebt/runs/test_sdpa.sh [--gpus N]   # default N=1
# Usage (from nova/ebt/):
#   bash runs/test_sdpa.sh [--gpus N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EBT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOVA_DIR="$(cd "${EBT_DIR}/../.." && pwd)"
VENV="${NOVA_DIR}/.venv/ebt/bin/activate"

# Parse --gpus argument (default: 1)
NGPUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) NGPUS="$2"; shift 2 ;;
        *)      echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== EBT SDPA Attention Test ==="
echo "EBT dir : ${EBT_DIR}"
echo "venv    : ${VENV}"
echo "GPUs    : ${NGPUS}"
echo

# shellcheck disable=SC1090
source "${VENV}"
cd "${EBT_DIR}"

# ── Single-GPU: correctness tests + benchmarks ───────────────────────────────
echo "--- Correctness tests + benchmarks (single-GPU) ---"
OMP_NUM_THREADS=1 torchrun \
    --standalone \
    --nproc_per_node=1 \
    test/test_sdpa_attention.py
echo

# ── Distributed tests (only when NGPUS > 1) ──────────────────────────────────
if [[ "${NGPUS}" -gt 1 ]]; then
    echo "--- Distributed correctness + benchmarks (${NGPUS} GPUs) ---"
    OMP_NUM_THREADS=1 torchrun \
        --standalone \
        --nproc_per_node="${NGPUS}" \
        test/test_sdpa_attention.py
fi
