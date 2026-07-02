#!/bin/bash

################################################################################
# EBT d26 free-embedding direct-unembed Sudoku SFT - 1 node x 8 GPUs
#
# Thin wrapper around the 2-node script. The underlying script is fully
# parameterized by NODE_COUNT/PROC_PER_NODE, so keeping one implementation avoids
# config drift between 1-node smoke runs and 2-node production runs.
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NODE_COUNT="${NODE_COUNT:-1}"
export PROC_PER_NODE="${PROC_PER_NODE:-8}"
export RUN_PREFIX="${RUN_PREFIX:-d26-ctx2048-sudoku-sft-freeembed-direct-1node-8gpu}"
export CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat}"

# Required for the source free-embed checkpoint: d26 uses dim=1664 and
# FFN hidden dim=int(1664 * 2.67)=4442. Without this, the model is built with
# FFN hidden dim=1664 and checkpoint loading fails with size mismatches.
export FFN_DIM_MULTIPLIER="${FFN_DIM_MULTIPLIER:-2.67}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.6}"

exec "${SCRIPT_DIR}/run_ebt_freeembed_sudoku_sft_2node_8gpu.sh"
