#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_ROOT="${REPO_ROOT}/nova/ebt/logs"
PAIR_LOG="${LOG_ROOT}/worker_pair_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${PAIR_LOG}") 2>&1

echo "pair_start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "pair_log=${PAIR_LOG}"

echo "launching VE run"
set +e
bash "${SCRIPT_DIR}/run_ebt_xxs_muon_adamw_0403_worker_ve.sh"
VE_EXIT=$?
echo "ve_exit=${VE_EXIT}"

echo "launching no-VE run"
bash "${SCRIPT_DIR}/run_ebt_xxs_muon_adamw_0403_worker_nove.sh"
NOVE_EXIT=$?
set -e
echo "nove_exit=${NOVE_EXIT}"

echo "pair_end=$(date '+%Y-%m-%d %H:%M:%S')"
