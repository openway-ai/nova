#!/bin/bash

################################################################################
# Submit and run the Sudoku EBM-vs-LLM comparison for the latest free-embed ckpt.
#
# Usage from the repo or any login node:
#   bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/rjob/run_sudoku_compare_freeembed_eval.sh
#
# The script is self-submitting. Without INSIDE_RJOB=1 it submits itself through
# rjob. Inside the container it initializes the same project environment style as
# the existing EBT rjob scripts, runs run_all_sudoku_compare.sh, and writes a
# final Markdown analysis report.
#
# Model set:
#   The LLM comparison set is intentionally not changed here. It is inherited
#   from run_all_sudoku_compare.sh: qwen3_1p7b, llama3p2_1b, qwen3p6_27b, and
#   deepseek_r1_0528. The only model override is EBM_CKPT below.
#
# Note:
#   The compare module resolves LLM weights under
#   /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub, so this
#   submitter adds a gpfs2 public mount in addition to the gpfs1 mounts used by
#   the training rjobs.
################################################################################

set -euo pipefail

OPENEBM_HOME_DEFAULT="/mnt/shared-storage-user/puyuan/code/OpenEBM"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/run_sudoku_compare_freeembed_eval.sh}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
REPO_ROOT="${REPO_ROOT:-${OPENEBM_HOME_DEFAULT}}"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-${REPO_ROOT}}"
COMPARE_BRANCH="${COMPARE_BRANCH:-dev-openebm-sudoku-compare}"
COMPARE_COMMIT="${COMPARE_COMMIT:-$(git -C "${SOURCE_REPO_ROOT}" rev-parse "${COMPARE_BRANCH}")}"

EBM_CKPT="${EBM_CKPT:-${REPO_ROOT}/logs/ebt_runs/d26-ctx2048-sudoku-sft-freeembed-direct-1node-8gpu-20260625-190601/sft_train/checkpoints/s=step=2999-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4366.ckpt}"
RUN_PREFIX="${RUN_PREFIX:-sudoku-compare-freeembed-eval-fix2}"
RUN_ID="${RUN_ID:-${RUN_PREFIX}-$(date +%Y%m%d-%H%M%S)}"
JOB_WORKTREE="${JOB_WORKTREE:-/tmp/openebm-sudoku-compare-${RUN_ID}}"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/openebm/elm/runs/sudoku_compare}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/openebm/elm/data/sudoku_cache_v2}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
START_INDEX="${START_INDEX:-0}"
STAGE_SIZE="${STAGE_SIZE:-0}"

RUN_EBM="${RUN_EBM:-1}"
RUN_LLMS="${RUN_LLMS:-1}"
RUN_SMALL="${RUN_SMALL:-1}"
RUN_QWEN27="${RUN_QWEN27:-1}"
RUN_R1="${RUN_R1:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
VERIFY_ENV="${VERIFY_ENV:-1}"

FFN_DIM_MULTIPLIER="${FFN_DIM_MULTIPLIER:-2.67}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.6}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

RJOB_NAME="${RJOB_NAME:-sudoku-compare-freeembed-eval-fix2}"
RJOB_GPU="${RJOB_GPU:-8}"
RJOB_MEMORY="${RJOB_MEMORY:-1000000}"
RJOB_CPU="${RJOB_CPU:-100}"
RJOB_PRIORITY="${RJOB_PRIORITY:-1}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119}"
HF_PUBLIC_MOUNT="${HF_PUBLIC_MOUNT:-gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public}"

if [[ "${INSIDE_RJOB:-0}" != "1" && "${SUBMIT_RJOB:-1}" == "1" ]]; then
  if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Run script not found: ${RUN_SCRIPT}" >&2
    exit 1
  fi
  git -C "${SOURCE_REPO_ROOT}" cat-file -e "${COMPARE_COMMIT}^{commit}"

  JOB_COMMAND='
set -euo pipefail
cd "${SOURCE_REPO_ROOT}"
git cat-file -e "${COMPARE_COMMIT}^{commit}"
if [[ -e "${JOB_WORKTREE}" ]]; then
  echo "Job worktree already exists: ${JOB_WORKTREE}" >&2
  exit 1
fi
git worktree add --detach "${JOB_WORKTREE}" "${COMPARE_COMMIT}"
cd "${JOB_WORKTREE}"
echo "[run_sudoku_compare_freeembed_eval] compare_branch=${COMPARE_BRANCH}"
echo "[run_sudoku_compare_freeembed_eval] compare_commit=$(git rev-parse HEAD)"
export REPO_ROOT="${JOB_WORKTREE}"
export NOVA_HOME="${JOB_WORKTREE}"
INSIDE_RJOB=1 SUBMIT_RJOB=0 bash openebm/elm/runs/rjob/run_sudoku_compare_freeembed_eval.sh
'

  rjob submit \
    --name="${RJOB_NAME}" \
    --gpu="${RJOB_GPU}" \
    --memory="${RJOB_MEMORY}" \
    --cpu="${RJOB_CPU}" \
    --charged-group="${RJOB_CHARGED_GROUP}" \
    --private-machine="${RJOB_PRIVATE_MACHINE}" \
    -P "${RJOB_PRIORITY}" \
    --image="${RJOB_IMAGE}" \
    --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
    --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
    --mount="${HF_PUBLIC_MOUNT}" \
    -e DISTRIBUTED_JOB=true \
    -e SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT}" \
    -e REPO_ROOT="${REPO_ROOT}" \
    -e COMPARE_BRANCH="${COMPARE_BRANCH}" \
    -e COMPARE_COMMIT="${COMPARE_COMMIT}" \
    -e JOB_WORKTREE="${JOB_WORKTREE}" \
    -e CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    -e PYTHON_BIN="${PYTHON_BIN}" \
    -e EBM_CKPT="${EBM_CKPT}" \
    -e RUN_PREFIX="${RUN_PREFIX}" \
    -e RUN_ID="${RUN_ID}" \
    -e RESULTS_ROOT="${RESULTS_ROOT}" \
    -e DATA_DIR="${DATA_DIR}" \
    -e NUM_SAMPLES="${NUM_SAMPLES}" \
    -e START_INDEX="${START_INDEX}" \
    -e STAGE_SIZE="${STAGE_SIZE}" \
    -e RUN_EBM="${RUN_EBM}" \
    -e RUN_LLMS="${RUN_LLMS}" \
    -e RUN_SMALL="${RUN_SMALL}" \
    -e RUN_QWEN27="${RUN_QWEN27}" \
    -e RUN_R1="${RUN_R1}" \
    -e RUN_REPORTS="${RUN_REPORTS}" \
    -e DRY_RUN="${DRY_RUN}" \
    -e VERIFY_ENV="${VERIFY_ENV}" \
    -e FFN_DIM_MULTIPLIER="${FFN_DIM_MULTIPLIER}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    -e MPLCONFIGDIR="${MPLCONFIGDIR}" \
    --custom-resources brainpp.cn/fuse=1 \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -- bash -exc "${JOB_COMMAND}"
  exit 0
fi

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  if ! conda activate "${CONDA_ENV_PATH}"; then
    echo "Warning: conda activate failed for ${CONDA_ENV_PATH}; continuing with explicit python path." >&2
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PYTHON="${PYTHON:-${PYTHON_BIN}}"
export EBM_PYTHON="${EBM_PYTHON:-${PYTHON_BIN}}"
export LLM_PYTHON="${LLM_PYTHON:-${PYTHON_BIN}}"
export REPORT_PYTHON="${REPORT_PYTHON:-${PYTHON_BIN}}"

NOVA_HOME="${NOVA_HOME:-${REPO_ROOT}}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${NOVA_HOME}/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh}"
REPORT_MODULE="${REPORT_MODULE:-openebm.elm.scripts.sudoku_compare.generate_sudoku_compare_report}"
HF_MODEL_ROOT="${HF_MODEL_ROOT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub}"

cd "${NOVA_HOME}"
export PYTHONPATH="${NOVA_HOME}/nanochat:${NOVA_HOME}:${PYTHONPATH:-}"
export HOME="${NOVA_HOME}"
export HF_HOME="${HF_HOME:-${NOVA_HOME}/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${NOVA_HOME}/data/hf_datasets_cache}"
export MPLCONFIGDIR
export PYTORCH_CUDA_ALLOC_CONF
export FFN_DIM_MULTIPLIER
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOCHAT_OFFLINE_MODE="${NANOCHAT_OFFLINE_MODE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "${MPLCONFIGDIR}"

export REPO_ROOT
export EBM_CKPT
export RESULTS_ROOT
export DATA_DIR
export NUM_SAMPLES
export START_INDEX
export STAGE_SIZE
export RUN_ID
export RUN_EBM
export RUN_LLMS
export RUN_SMALL
export RUN_QWEN27
export RUN_R1
export RUN_REPORTS
export DRY_RUN
export VERIFY_ENV

RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${RESULTS_ROOT}/outputs/${RUN_ID}}"
REPORT_OUT_DIR="${REPORT_OUT_DIR:-${RUN_OUTPUT_ROOT}/reports}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs/${RUN_ID}}"
export RUN_OUTPUT_ROOT
export REPORT_OUT_DIR
export LOG_DIR

if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
  echo "Compare script not found: ${COMPARE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${EBM_CKPT}" ]]; then
  echo "EBM checkpoint not found: ${EBM_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/satnet_test.pt" ]]; then
  echo "Sudoku test split not found: ${DATA_DIR}/satnet_test.pt" >&2
  exit 1
fi
if [[ "${RUN_LLMS}" == "1" && ! -d "${HF_MODEL_ROOT}" ]]; then
  echo "LLM model root not found: ${HF_MODEL_ROOT}" >&2
  echo "Check the gpfs2 public Hugging Face mount or set HF_MODEL_ROOT." >&2
  exit 1
fi

echo "=== Sudoku compare rjob ==="
echo "REPO_ROOT:       ${REPO_ROOT}"
echo "RUN_ID:          ${RUN_ID}"
echo "EBM_CKPT:        ${EBM_CKPT}"
echo "DATA_DIR:        ${DATA_DIR}"
echo "RESULTS_ROOT:    ${RESULTS_ROOT}"
echo "RUN_OUTPUT_ROOT: ${RUN_OUTPUT_ROOT}"
echo "REPORT_OUT_DIR:  ${REPORT_OUT_DIR}"
echo "NUM_SAMPLES:     ${NUM_SAMPLES}"
echo "START_INDEX:     ${START_INDEX}"
echo "STAGE_SIZE:      ${STAGE_SIZE}"
echo "PYTHON_BIN:      ${PYTHON_BIN}"
echo "HF_MODEL_ROOT:   ${HF_MODEL_ROOT}"

bash "${COMPARE_SCRIPT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[run_sudoku_compare_freeembed_eval] DRY_RUN=1; skip final analysis report."
  exit 0
fi
if [[ "${RUN_REPORTS}" != "1" ]]; then
  echo "[run_sudoku_compare_freeembed_eval] RUN_REPORTS=${RUN_REPORTS}; skip final analysis report."
  exit 0
fi
if ! [[ "${STAGE_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "STAGE_SIZE must be a non-negative integer, got ${STAGE_SIZE}" >&2
  exit 2
fi

if [[ "${STAGE_SIZE}" -gt 0 ]]; then
  COMPARISON_CSV="${REPORT_OUT_DIR}/latest_stage_cumulative/comparison.csv"
  REPORT_LABEL="${RUN_ID} staged cumulative"
else
  COMPARISON_CSV="${REPORT_OUT_DIR}/comparison.csv"
  REPORT_LABEL="${RUN_ID}"
fi
FINAL_REPORT="${REPORT_OUT_DIR}/analysis_report.md"

"${REPORT_PYTHON}" -m "${REPORT_MODULE}" \
  --comparison-csv "${COMPARISON_CSV}" \
  --out-md "${FINAL_REPORT}" \
  --run-env "${LOG_DIR}/run_env.txt" \
  --checkpoint "${EBM_CKPT}" \
  --data-dir "${DATA_DIR}" \
  --results-root "${RUN_OUTPUT_ROOT}" \
  --report-dir "${REPORT_OUT_DIR}" \
  --report-label "${REPORT_LABEL}"

ln -sfn "${FINAL_REPORT}" "${RESULTS_ROOT}/latest_analysis_report.md"
echo "[run_sudoku_compare_freeembed_eval] final report: ${FINAL_REPORT}"
