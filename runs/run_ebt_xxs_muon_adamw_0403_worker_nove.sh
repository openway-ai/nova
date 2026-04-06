#!/bin/bash
set -euo pipefail

RUN_NAME_BASE="ebt-xxs-muon-adamw-0403-worker-nove"
MODEL_NAME="${RUN_NAME_BASE%%-*}"
MODEL_SIZE="xxs"

REPO_ROOT="/home/liqinuo/nova-dev-ebt"
EBT_DIR="${REPO_ROOT}/nova/ebt"
TORCHRUN_BIN="/home/liqinuo/nova/.venv/bin/torchrun"
DATA_ROOT="/data/liqinuo/nova-dev-ebt-worker-0403"
TMP_ROOT="${DATA_ROOT}/tmp"
CHECKPOINT_ROOT="${DATA_ROOT}/checkpoints"
LOG_ROOT="${EBT_DIR}/logs"

NUM_GPUS=2
DEVICE_BATCH_SIZE=4
GRAD_ACCUM=32
CONTEXT_LENGTH=512
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
TARGET_TOTAL_TOKENS=7340032000
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=${MAX_STEPS}

MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
DENOISING_INITIAL_CONDITION="random_noise"
USE_VE=false

PEAK_LR=0.0012
WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0
VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"
COMPILE_FLAGS="--compile_model --compile_mode full"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export WANDB_MODE="offline"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMP_ROOT}"

mkdir -p "${TMP_ROOT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"
rm -rf "${LOG_ROOT}/checkpoints"
ln -s "${CHECKPOINT_ROOT}" "${LOG_ROOT}/checkpoints"

cd "${EBT_DIR}"

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_ROOT}/${current_time}_${RUN_NAME_BASE}_ctx${CONTEXT_LENGTH}_gpu${NUM_GPUS}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
echo "hostname=$(hostname)"
echo "run_name=${RUN_NAME_BASE}_${current_time}"
echo "use_ve=${USE_VE}"
echo "num_gpus=${NUM_GPUS}"
echo "device_batch_size=${DEVICE_BATCH_SIZE}"
echo "grad_accum=${GRAD_ACCUM}"
echo "effective_batch_size_tokens=${EFFECTIVE_BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "tmp_root=${TMP_ROOT}"

exec "${TORCHRUN_BIN}" --standalone --nproc_per_node=${NUM_GPUS} "${EBT_DIR}/train.py" \
--run_name "${RUN_NAME_BASE}_${current_time}" \
--modality "NLP" \
--model_name "${MODEL_NAME}" \
--model_size "${MODEL_SIZE}" \
--pretokenize_dataset \
--normalize_initial_condition \
--ebt_type "${EBT_TYPE}" \
--denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
--mcmc_step_size_learnable \
--mcmc_step_size "${MCMC_STEP_SIZE}" \
--mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
--mcmc_num_steps "${MCMC_NUM_STEPS}" \
--context_length "${CONTEXT_LENGTH}" \
--gpus "-1" \
--peak_learning_rate "${PEAK_LR}" \
--batch_size_per_device "${DEVICE_BATCH_SIZE}" \
--accumulate_grad_batches "${GRAD_ACCUM}" \
--gradient_clip_val "${GRADIENT_CLIP_VAL}" \
--weight_decay "${WEIGHT_DECAY}" \
--beta1 "${BETA1}" \
--beta2 "${BETA2}" \
--min_lr_scale 50 \
--max_steps "${MAX_STEPS}" \
--max_scheduling_steps "${MAX_SCHEDULING_STEPS}" \
--warm_up_steps 0 \
--warm_up_base_lr_divider 10 \
--dataset_name "nanochat" \
--num_workers "${NUM_WORKERS}" \
--val_check_interval "${VAL_CHECK_INTERVAL}" \
--limit_val_batches "${LIMIT_VAL_BATCHES}" \
--val_sanity 1 \
--validation_split_pct 0.0027 \
--wandb_project "nlp_pretrain" \
--log_model_archi \
--set_matmul_precision "medium" \
${OPTION_FLAGS} \
${COMPILE_FLAGS}
