#!/bin/bash

################################################################################
# EBT d26 blockwise 多机训练 rjob 提交脚本 - 2节点 × 8卡/节点
#
# 用法:
#   bash rjob_ebt_blockwise_2node_8gpu.sh
#
# 可选环境变量:
#   BLOCK_MODE=blockwise|mtp_mcmc|future_latent_non_causal
#   TRAIN_BLOCK_SIZE=2
#
# 平台通过 -e DISTRIBUTED_JOB=true 自动注入:
#   NODE_RANK / NODE_COUNT / MASTER_ADDR / PROC_PER_NODE / JOB_ID
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX / CUDA_VISIBLE_DEVICES
################################################################################

BLOCK_MODE="${BLOCK_MODE:-blockwise}"
TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE:-2}"

rjob submit \
  --name=ebt-d26-blockwise-2node-8gpu \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  -P 2 \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/lixueyan:/mnt/shared-storage-user/lixueyan \
  -e DISTRIBUTED_JOB=true \
  -e BLOCK_MODE="${BLOCK_MODE}" \
  -e TRAIN_BLOCK_SIZE="${TRAIN_BLOCK_SIZE}" \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- bash -exc "bash /mnt/shared-storage-user/lixueyan/nar/nova/nova/ebt/runs/rjob/run_ebt_blockwise_2node_8gpu.sh"
