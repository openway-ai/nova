#!/bin/bash

################################################################################
# EBT standalone valid_loss 评估脚本 - rjob 1节点 × 4卡
#
# 这个版本走训练时同口径的 validation_step / eval_step / forward_loss_wrapper。
# 通过外层 torchrun 启动 4 个进程，避免本地 Lightning spawn 额外放大内存。
################################################################################

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/lixueyan/conda_envs/ebt
export LD_LIBRARY_PATH="/mnt/shared-storage-user/lixueyan/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
umask 022

NOVA_HOME="/mnt/shared-storage-user/lixueyan/nar/nova"
EBT_DIR="${NOVA_HOME}/nova/ebt"
TRAIN_SCRIPT="${EBT_DIR}/train.py"

cd "${NOVA_HOME}"

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: CKPT_PATH is required" >&2
  exit 2
fi

BLOCK_MODE="${BLOCK_MODE:-blockwise}"
TRAINING_OBJECTIVE="${TRAINING_OBJECTIVE:-blockwise}"
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
VAL_STEPS="${VAL_STEPS:-1000}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/shared-storage-user/lixueyan/nar/tokenizer}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NODE_COUNT=${NODE_COUNT:-1}
PROC_PER_NODE=${PROC_PER_NODE:-4}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
JOB_ID=${JOB_ID:-$$}
MASTER_PORT=$(( 20000 + 0x$(echo "${JOB_ID}" | md5sum | head -c 4) % 10000 ))

NUM_NODES=${NODE_COUNT}
GPUS_PER_NODE=${PROC_PER_NODE}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

RUN_PREFIX="validloss-rjob-1node-4gpu-blockwise"
TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_ctx${CONTEXT_LENGTH}"

LOG_DIR="${NOVA_HOME}/logs_base_eval/${DATE_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

echo "=== 启动 rjob standalone valid_loss 评估 ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "LOG_FILE: ${LOG_FILE}"
echo "CKPT_PATH: ${CKPT_PATH}"
echo "BLOCK_MODE: ${BLOCK_MODE}"
echo "TRAINING_OBJECTIVE: ${TRAINING_OBJECTIVE}"
echo "TRAIN_BLOCK_SIZE: ${TRAIN_BLOCK_SIZE}"
echo "CONTEXT_LENGTH: ${CONTEXT_LENGTH}"
echo "LIMIT_VAL_BATCHES: ${LIMIT_VAL_BATCHES}"
echo "VAL_STEPS: ${VAL_STEPS}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NODE_COUNT: ${NODE_COUNT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "PROC_PER_NODE: ${PROC_PER_NODE}"
echo "WORLD_SIZE: ${WORLD_SIZE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

set +e

TORCHRUN_CMD=(
  torchrun
  --nnodes="${NUM_NODES}"
  --nproc_per_node="${GPUS_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  --rdzv_id="${JOB_ID}"
  "${TRAIN_SCRIPT}"
  --only_validate
  --only_validate_model_ckpt "${CKPT_PATH}"
  --execution_mode "pretrain"
  --dataset_name "nanochat"
  --context_length "${CONTEXT_LENGTH}"
  --tokenizer "${TOKENIZER_PATH}"
  --training_objective "${TRAINING_OBJECTIVE}"
  --block_mode "${BLOCK_MODE}"
  --train_block_size "${TRAIN_BLOCK_SIZE}"
  --gpus "-1"
  --batch_size_per_device "${DEVICE_BATCH_SIZE}"
  --limit_val_batches "${LIMIT_VAL_BATCHES}"
  --val_steps "${VAL_STEPS}"
  --set_matmul_precision "medium"
  --float_precision "bf16-mixed"
  --no_wandb
  --val_sanity 0
  --val_check_interval 1.0
)

printf ' %q' "${TORCHRUN_CMD[@]}"
echo
"${TORCHRUN_CMD[@]}"

EVAL_EXIT_CODE=$?
set -e

echo ""
if [[ ${EVAL_EXIT_CODE} -eq 0 ]]; then
  echo "rjob standalone valid_loss 评估成功结束"
else
  echo "rjob standalone valid_loss 评估失败，退出码: ${EVAL_EXIT_CODE}"
fi

exit "${EVAL_EXIT_CODE}"
