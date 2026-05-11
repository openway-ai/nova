#!/bin/bash

################################################################################
# 提交 EBT Web 对话服务到 rjob — 1 节点 × 1 卡（推理 / FastAPI）
#
# 用法:
#   bash rjob_chat_ebt_web.sh
#   PORT=8080 CKPT_PATH=/path/to.ckpt bash rjob_chat_ebt_web.sh
#
# 说明:
#   - 与训练任务不同：不要加 DISTRIBUTED_JOB=true。
#   - 容器内必须能访问 NOVA_HOME（本脚本用提交机路径）。若代码在 lixueyan
#     目录下，需挂载对应 GPFS（见下方 --mount）；否则改为挂载你的目录并
#     调整路径，或像训练 rjob 一样改用已挂载卷上的副本路径。
#   - 根据 checkpoint/tokenizer 所在存储，补全或修改下方 --mount。
#   - Web 端口需在平台侧做转发或 ingress；此处仅启动容器内监听。
#   - 已加 --enable-sshd，便于平台 SSH 进 Pod 做端口转发或排查。
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# 传给容器内启动命令的环境（可按需改端口、权重路径等）
EXTRA_ENV=()
[[ -n "${PORT:-}" ]] && EXTRA_ENV+=(-e "PORT=${PORT}")
[[ -n "${CKPT_PATH:-}" ]] && EXTRA_ENV+=(-e "CKPT_PATH=${CKPT_PATH}")
[[ -n "${TOKENIZER_PATH:-}" ]] && EXTRA_ENV+=(-e "TOKENIZER_PATH=${TOKENIZER_PATH}")
[[ -n "${HOST:-}" ]] && EXTRA_ENV+=(-e "HOST=${HOST}")

rjob submit \
  --name=ebt-web-chat \
  --gpu=1 \
  --memory=256000 \
  --cpu=32 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  --host-network=false \
  --enable-sshd \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  --mount=gpfs://gpfs1/lixueyan:/mnt/shared-storage-user/lixueyan \
  "${EXTRA_ENV[@]}" \
  -- bash -exc "bash '${NOVA_HOME}/openebm/elm/runs/rjob/run_chat_ebt_web.sh'"
