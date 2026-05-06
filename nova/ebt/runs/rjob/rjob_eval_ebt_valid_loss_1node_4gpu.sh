#!/bin/bash

################################################################################
# EBT standalone valid_loss 单机 4 卡 rjob 提交脚本
#
# 用法:
#   CKPT_PATH=/path/to/model.ckpt bash rjob_eval_ebt_valid_loss_1node_4gpu.sh
################################################################################

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

rjob submit \
  --name=ebt-validloss-1node-4gpu \
  --gpu=4 \
  --memory=500000 \
  --cpu=40 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/lixueyan:/mnt/shared-storage-user/lixueyan \
  -e DISTRIBUTED_JOB=true \
  -e CKPT_PATH="${CKPT_PATH}" \
  -e BLOCK_MODE="${BLOCK_MODE}" \
  -e TRAINING_OBJECTIVE="${TRAINING_OBJECTIVE}" \
  -e TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE}" \
  -e CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  -e DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE}" \
  -e LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES}" \
  -e VAL_STEPS="${VAL_STEPS}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=1 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "bash /mnt/shared-storage-user/lixueyan/nar/nova/nova/ebt/runs/rjob/run_eval_ebt_valid_loss_1node_4gpu.sh"
