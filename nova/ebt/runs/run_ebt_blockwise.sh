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
# Blockwise training config
#   This script covers all three block_modes that share the `blockwise`
#   training_objective (i.e. the dataloader yields block_targets [B, K, S]).
#   Switch branches by setting the BLOCK_MODE env var; everything else
#   (dataset, optimizer, MCMC settings) is shared.
################################################################################
# block_mode selects the attention/MCMC/loss semantic. Supported here:
#   * mtp_mcmc                 (default; current dev-blockwise behavior)
#   * future_latent_non_causal (K independent latents per source pos, intra-
#                               block non-causal)
#   * blockwise                (K independent latents per source pos, intra-
#                               block causal)
# `dense_token` is intentionally NOT supported by this script: it requires
# --training_objective dense_next_token, which is a different training recipe;
# use a separate run script for it.
BLOCK_MODE="${BLOCK_MODE:-blockwise}"
case "$BLOCK_MODE" in
  mtp_mcmc|future_latent_non_causal|blockwise|future_latent_bidirectional)
    TRAINING_OBJECTIVE="${TRAINING_OBJECTIVE:-blockwise}"
    ;;
  dense_token)
    echo "ERROR: --block_mode dense_token requires --training_objective dense_next_token;" >&2
    echo "       this script is for the blockwise training_objective. Use a dense run script." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown BLOCK_MODE='$BLOCK_MODE'." >&2
    echo "       Allowed: mtp_mcmc | future_latent_non_causal | blockwise | future_latent_bidirectional" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-ebt-xxs-blockwise-k2-${BLOCK_MODE}}"
MODEL_NAME="${MODEL_NAME:-ebt}"
MODEL_SIZE="${MODEL_SIZE:-xxs}"
DATASET_NAME="${DATASET_NAME:-nanochat}"

# S in the [B, K, S] dataloader output: number of source positions per sample.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
# K in the [B, K, S] dataloader output: number of future tokens predicted per
# source position. For mtp_mcmc, K is the number of joint-head offsets.
# For future_latent_non_causal / blockwise, K is the number of independent
# latents allocated per source position.
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"
# Optional override for the dataloader-side context length (kept for
# parity with the dense recipe; the blockwise dataloader currently uses
# CONTEXT_LENGTH directly, so leaving this empty is normal).
TRAIN_CONTEXT_LENGTH="${TRAIN_CONTEXT_LENGTH:-}"

################################################################################
# E1 / E2 experimental knobs
#
# These default to no-ops (i.e. the pre-E1/E2 behavior). Override via env vars
# from the launcher to run ablations. Example:
#   OFFSET_LOSS_WEIGHTS="1.0,0.3" RUN_NAME=ebt-xxs-w10_03 bash run_ebt_blockwise.sh
#   BLOCK_LATENT_HEAD_TYPE=per_offset_transformer BLOCK_LATENT_HEAD_LAYERS=1 \
#     RUN_NAME=ebt-xxs-mtphead bash run_ebt_blockwise.sh
#   # E4: Gemma-style teacher-forced heads (offset j>=1 conditions on
#   #     embed(realized x_{t+j-1})). Two variants:
#   BLOCK_LATENT_HEAD_TYPE=per_offset_tf_linear \
#     RUN_NAME=ebt-xxs-tflinear bash run_ebt_blockwise.sh
#   BLOCK_LATENT_HEAD_TYPE=per_offset_tf_transformer BLOCK_LATENT_HEAD_LAYERS=1 \
#     RUN_NAME=ebt-xxs-tfmtphead bash run_ebt_blockwise.sh
################################################################################
# E1.1 — per-offset CE weights. Comma-separated, length == TRAIN_BLOCK_SIZE.
# Empty (default) -> uniform; equivalent to original mean-over-flattened CE.
OFFSET_LOSS_WEIGHTS="${OFFSET_LOSS_WEIGHTS:-}"

# E2 — readout head. "linear" (default) preserves the original shared head;
# "per_offset_linear" / "per_offset_mlp" / "per_offset_transformer" build a
# ModuleList of K heads (one per offset).
# E4 (teacher-forced, Gemma-drafter style): "per_offset_tf_linear" and
# "per_offset_tf_transformer" — each offset j conditions on embed(x_{t+j-1})
# (TF during training, greedy argmax within block at inference).
BLOCK_LATENT_HEAD_TYPE="${BLOCK_LATENT_HEAD_TYPE:-linear}"
# E2.3 only: number of causal AR blocks per offset transformer head.
BLOCK_LATENT_HEAD_LAYERS="${BLOCK_LATENT_HEAD_LAYERS:-1}"
# E2.2 / E2.3: FFN multiplier inside the per-offset head.
BLOCK_LATENT_HEAD_FFN_MULT="${BLOCK_LATENT_HEAD_FFN_MULT:-4.0}"
# E2.3 only: attention heads inside per_offset_transformer head; 0 = inherit
# the trunk's --multiheaded_attention_heads.
BLOCK_LATENT_HEAD_N_HEADS="${BLOCK_LATENT_HEAD_N_HEADS:-0}"

# E3.1 — causal-detach in MCMC. Empty/0 = off (default, no-op).
# Any non-empty value (e.g. "1" or "true") turns it ON via --mcmc_causal_detach.
MCMC_CAUSAL_DETACH="${MCMC_CAUSAL_DETACH:-}"

# E3.2 — staggered MCMC. Same on/off convention as MCMC_CAUSAL_DETACH.
# Can be combined with E3.1 (both flags can be on simultaneously).
MCMC_STAGGERED="${MCMC_STAGGERED:-}"

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

  # --- Run identity / data ---------------------------------------------------
  --run_name "$RUN_NAME"                        # wandb / checkpoint dir name
  --modality NLP                                # this script is NLP-only
  --model_name "$MODEL_NAME"                    # "ebt" -> EBT_NLP
  --model_size "$MODEL_SIZE"                    # xxs / xs / s / ... preset
  --dataset_name "$DATASET_NAME"                # nanochat tokenizer + corpus
  --execution_mode pretrain                     # not finetune (blockwise FT n/a)
  --pretokenize_dataset                         # cache token IDs to disk

  # --- EBT energy-input pre-processing --------------------------------------
  --normalize_initial_condition                 # L2-normalize the noised input
  --denoising_initial_condition random_noise    # init MCMC from gaussian noise
  --ebt_type time_embed                         # trunk variant: prepend a time
                                                #  token; 'default' is also OK

  # --- Block-mode dispatch (training + attention semantic) ------------------
  # training_objective drives which dataset/loss path is used:
  #   * blockwise -> dataloader yields (input_ids [B,S], block_targets [B,K,S])
  # block_mode drives which attention/MCMC/loss semantic the model uses:
  #   * mtp_mcmc                 -> 1 latent / source pos + joint K-head
  #   * future_latent_non_causal -> K latents / source pos, intra-block
  #                                 latents do NOT see each other
  #   * blockwise                -> K latents / source pos, intra-block
  #                                 latents are causal (z_{t,j} sees z_{t,<=j})
  --training_objective "$TRAINING_OBJECTIVE"
  --block_mode "$BLOCK_MODE"

  # --- Sequence layout ------------------------------------------------------
  --context_length "$CONTEXT_LENGTH"            # S: # source positions
  --train_block_size "$TRAIN_BLOCK_SIZE"        # K: # future tokens / pos

  # --- E1 / E2 / E3 experimental knobs (no-op when left at defaults) --------
  --offset_loss_weights "$OFFSET_LOSS_WEIGHTS"             # E1.1
  --block_latent_head_type "$BLOCK_LATENT_HEAD_TYPE"       # E2.x
  --block_latent_head_layers "$BLOCK_LATENT_HEAD_LAYERS"   # E2.3
  --block_latent_head_ffn_mult "$BLOCK_LATENT_HEAD_FFN_MULT"  # E2.2 / E2.3
  --block_latent_head_n_heads "$BLOCK_LATENT_HEAD_N_HEADS"   # E2.3

  # --- MCMC (used by all 3 block_modes here) --------------------------------
  --mcmc_step_size_learnable                    # learn alpha per layer/step
  --mcmc_step_size 500                          # initial alpha (pre-sigmoid)
  --mcmc_step_size_lr_multiplier 1500           # alpha LR multiplier
  --mcmc_num_steps "${MCMC_NUM_STEPS:-2}"       # # Langevin steps per forward (env-var overridable)

  # --- Distributed / optimizer ----------------------------------------------
  --gpus "$GPUS"                                # -1 = all visible GPUs
  --peak_learning_rate "$PEAK_LR"
  --batch_size_per_device "$BATCH_SIZE_PER_DEVICE"
  --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
  --gradient_clip_val 1.0
  --weight_decay 0.01
  --min_lr_scale 10                             # min_lr = peak_lr / 10
  --max_steps "$MAX_STEPS"
  --max_scheduling_steps "$MAX_STEPS"           # LR scheduler horizon
  --warm_up_steps "$WARMUP_STEPS"
  --num_workers "$NUM_WORKERS"

  # --- Validation / logging -------------------------------------------------
  --val_check_interval "$VAL_CHECK_INTERVAL"
  --limit_val_batches "$LIMIT_VAL_BATCHES"
  --val_sanity 1
  --wandb_project nlp_pretrain
  --log_model_archi                             # dump model summary at start
  --log_every_n_steps 100
  --set_matmul_precision medium                 # tf32 matmul on Ampere+
)

if [ -n "$TRAIN_CONTEXT_LENGTH" ]; then
  CMD+=(--train_context_length "$TRAIN_CONTEXT_LENGTH")
fi

# E3.1 — append --mcmc_causal_detach only when env var is non-empty / truthy.
if [ -n "$MCMC_CAUSAL_DETACH" ] && [ "$MCMC_CAUSAL_DETACH" != "0" ] && [ "$MCMC_CAUSAL_DETACH" != "false" ]; then
  CMD+=(--mcmc_causal_detach)
fi
# E3.2 — append --mcmc_staggered only when env var is non-empty / truthy.
if [ -n "$MCMC_STAGGERED" ] && [ "$MCMC_STAGGERED" != "0" ] && [ "$MCMC_STAGGERED" != "false" ]; then
  CMD+=(--mcmc_staggered)
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  CMD+=(--is_slurm_run)
fi

echo "Running blockwise training:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
