#!/bin/bash

################################################################################
# EBT Web 对话服务 — 在 rjob 容器内启动（单机单卡推理，无需 DISTRIBUTED_JOB）
#
# 用法:
#   由 rjob_chat_ebt_web.sh 提交；本地可直接:
#     bash openebm/elm/runs/rjob/run_chat_ebt_web.sh
#     bash .../run_chat_ebt_web.sh --port 8080
#
# 说明:
#   - 实际逻辑在 ../chat_ebt_web.sh；本脚本只固定 NOVA_HOME 并转发参数。
#   - 默认 1 张 GPU；无需 NCCL / 多机环境变量。
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

exec bash "${NOVA_HOME}/openebm/elm/runs/chat_ebt_web.sh" "$@"
