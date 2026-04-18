#!/usr/bin/env bash
set -euo pipefail

# Optional SLURM hints:
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$EBT_DIR"

# Prefer project venv, then conda, then system python.
if [ -f "$EBT_DIR/../../.venv/ebt/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$EBT_DIR/../../.venv/ebt/bin/activate"
fi

if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
  PYTHON_BIN="$PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -x "/mnt/shared-storage-user/lixueyan/miniconda3/envs/nanochat/bin/python" ]; then
  PYTHON_BIN="/mnt/shared-storage-user/lixueyan/miniconda3/envs/nanochat/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="$(command -v python)"
fi

mkdir -p logs/slurm/nlp/
if command -v module >/dev/null 2>&1; then
  module purge
fi

################################################################################
# Blockwise training config (dense MTP-style supervision; edit these first)
################################################################################
RUN_NAME="${RUN_NAME:-ebt-xxs-blockwise-k2-v3}"
MODEL_NAME="${MODEL_NAME:-ebt}"
MODEL_SIZE="${MODEL_SIZE:-xxs}"
DATASET_NAME="${DATASET_NAME:-nanochat}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"
# Compat-only in dense mode (ignored by current blockwise path).
TRAIN_CONTEXT_LENGTH="${TRAIN_CONTEXT_LENGTH:-}"

GPUS="${GPUS:--1}"
PEAK_LR="${PEAK_LR:-0.0002}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
MAX_STEPS="${MAX_STEPS:-50000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-12}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-100}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"

################################################################################
# Launch
################################################################################
CMD=(
  "$PYTHON_BIN" train.py
  --run_name "$RUN_NAME"
  --modality NLP
  --model_name "$MODEL_NAME"
  --model_size "$MODEL_SIZE"
  --dataset_name "$DATASET_NAME"
  --execution_mode pretrain
  --pretokenize_dataset
  --normalize_initial_condition
  --denoising_initial_condition random_noise
  --ebt_type time_embed
  --training_objective blockwise
  --context_length "$CONTEXT_LENGTH"
  --train_block_size "$TRAIN_BLOCK_SIZE"
  --mcmc_step_size_learnable
  --mcmc_step_size 500
  --mcmc_step_size_lr_multiplier 1500
  --mcmc_num_steps 2
  --gpus "$GPUS"
  --peak_learning_rate "$PEAK_LR"
  --batch_size_per_device "$BATCH_SIZE_PER_DEVICE"
  --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
  --gradient_clip_val 1.0
  --weight_decay 0.01
  --min_lr_scale 10
  --max_steps "$MAX_STEPS"
  --max_scheduling_steps "$MAX_STEPS"
  --warm_up_steps "$WARMUP_STEPS"
  --num_workers "$NUM_WORKERS"
  --val_check_interval "$VAL_CHECK_INTERVAL"
  --limit_val_batches "$LIMIT_VAL_BATCHES"
  --val_sanity 1
  --wandb_project nlp_pretrain
  --log_model_archi
  --log_every_n_steps 100
  --set_matmul_precision medium
)

if [ -n "$TRAIN_CONTEXT_LENGTH" ]; then
  CMD+=(--train_context_length "$TRAIN_CONTEXT_LENGTH")
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  CMD+=(--is_slurm_run)
fi

echo "Running blockwise training:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
