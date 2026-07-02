#!/bin/bash

################################################################################
# GSM8K RL optimized rjob launcher for the PR30+PR33 fusion branch.
#
# Usage from the fusion worktree:
#   bash openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh
#
# The submit-side script records the current git commit. The rjob then creates a
# detached worktree for that exact commit before launching GSM8K RL, so dirty
# local checkouts do not affect the job.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM}"
FUSION_BRANCH="${FUSION_BRANCH:-$(git -C "${LOCAL_REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
FUSION_COMMIT="${FUSION_COMMIT:-$(git -C "${LOCAL_REPO_ROOT}" rev-parse HEAD)}"
RUN_ID="${RUN_ID:-d26-ctx2048-gsm8k-rl-fsdp2-merge-$(date +%Y%m%d-%H%M%S)}"

EBT_RUNS_ROOT="${EBT_RUNS_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs}"
GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data/gsm8k}"
# No dedicated GSM8K SFT ckpt is configured in this branch; keep the existing
# GSM8K RL script default unless the caller overrides SFT_CKPT.
SFT_CKPT="${SFT_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-0.6-20260508/sft_train/checkpoints/s=step=2990-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4782.ckpt}"
BASELINE_RL_RUN_DIR="${BASELINE_RL_RUN_DIR:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-gsm8k-rl-gspo-20260531/sft_train}"

RJOB_NAME="${RJOB_NAME:-${RUN_ID}}"
RJOB_GPU="${RJOB_GPU:-8}"
RJOB_CPU="${RJOB_CPU:-100}"
RJOB_MEMORY="${RJOB_MEMORY:-1000000}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-narmodel_gpu}"
RJOB_PRIVATE_MACHINE="${RJOB_PRIVATE_MACHINE:-group}"
RJOB_IMAGE="${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119}"
CONDA_ENV="${CONDA_ENV:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat}"
JOB_WORKTREE="${JOB_WORKTREE:-/tmp/openebm-${RUN_ID}}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-192}"
TEMPERATURE="${TEMPERATURE:-0.70}"
TOP_P="${TOP_P:-0.85}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
RL_LOSS_TYPE="${RL_LOSS_TYPE:-energy_gspo}"
MAX_STEPS="${MAX_STEPS:-500}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SAVE_TOP_K="${SAVE_TOP_K:-2}"
LEARNING_RATE="${LEARNING_RATE:-7e-7}"
BETA="${BETA:-0.4}"
ENERGY_KL_MODE="${ENERGY_KL_MODE:-symmetric_huber}"
ENERGY_KL_HUBER_DELTA="${ENERGY_KL_HUBER_DELTA:-0.5}"
SKIP_DEGENERATE_THRESHOLD="${SKIP_DEGENERATE_THRESHOLD:-0.85}"
MIN_REWARD_STD_TO_UPDATE="${MIN_REWARD_STD_TO_UPDATE:-0.01}"
MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE="${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE:-0.0}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adamw}"
ANALYZE_AFTER_TRAIN="${ANALYZE_AFTER_TRAIN:-1}"

run_inside_rjob() {
    cd "${LOCAL_REPO_ROOT}"

    if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
        source /root/miniconda3/etc/profile.d/conda.sh
        conda activate "${CONDA_ENV}"
    elif [[ -x "${CONDA_ENV}/bin/python" ]]; then
        export PATH="${CONDA_ENV}/bin:${PATH}"
    fi
    export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

    set +e
    NCCL_EXPORTS="$(curl -s --max-time 20 http://deploy.i.h.pjlab.org.cn/infra/scripts/nccl_auto_config.py | python3 - --shell-export 2>/dev/null)"
    NCCL_STATUS=$?
    set -e
    if [[ ${NCCL_STATUS} -eq 0 && -n "${NCCL_EXPORTS}" ]]; then
        eval "${NCCL_EXPORTS}"
    else
        echo "[WARN] nccl_auto_config.py was not applied; continuing with platform defaults."
    fi
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-60}"
    export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-20}"

    export PYTHONPATH="${LOCAL_REPO_ROOT}/nanochat:${LOCAL_REPO_ROOT}:${PYTHONPATH:-}"
    export EBT_RUNS_ROOT
    export GSM8K_DATA_PATH
    export SFT_CKPT
    export BASELINE_RL_RUN_DIR
    export EXP_ID="${EXP_ID:-${RUN_ID}}"
    export SKIP_CONFIRM=1
    export ANALYZE_AFTER_TRAIN
    export NUM_GPUS
    export NUM_GENERATIONS
    export MAX_COMPLETION_LENGTH
    export MAX_PROMPT_LENGTH
    export TEMPERATURE
    export TOP_P
    export GENERATION_BATCH_SIZE
    export RL_LOSS_TYPE
    export MAX_STEPS
    export VAL_CHECK_INTERVAL
    export LOG_INTERVAL
    export SAVE_TOP_K
    export LEARNING_RATE
    export BETA
    export ENERGY_KL_MODE
    export ENERGY_KL_HUBER_DELTA
    export SKIP_DEGENERATE_THRESHOLD
    export MIN_REWARD_STD_TO_UPDATE
    export MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE
    export RL_OPTIMIZER
    export WANDB_MODE="${WANDB_MODE:-offline}"

    echo "=== GSM8K RL optimized rjob ==="
    echo "fusion_branch=${FUSION_BRANCH}"
    echo "fusion_commit=${FUSION_COMMIT}"
    echo "exp_id=${EXP_ID}"
    echo "repo=${LOCAL_REPO_ROOT}"
    echo "sft_ckpt=${SFT_CKPT}"
    echo "gsm8k_data=${GSM8K_DATA_PATH}"

    if [[ ! -f "${SFT_CKPT}" ]]; then
        echo "Missing SFT checkpoint: ${SFT_CKPT}" >&2
        exit 1
    fi
    if [[ ! -d "${GSM8K_DATA_PATH}" ]]; then
        echo "Missing GSM8K data directory: ${GSM8K_DATA_PATH}" >&2
        exit 1
    fi

    bash openebm/elm/runs/run_ebt_gsm8k_rl.sh

    RUN_ROOT="${EBT_RUNS_ROOT}/${EXP_ID}"
    REPORT_DIR="$(find "${RUN_ROOT}" -maxdepth 1 -type d -name 'sft_train*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -z "${REPORT_DIR}" ]]; then
        REPORT_DIR="${RUN_ROOT}/sft_train"
    fi

    if [[ "${ANALYZE_AFTER_TRAIN}" = "1" && -f "${REPORT_DIR}/logs/train.log" ]]; then
        python openebm/elm/scripts/analyze_rl_run.py "${REPORT_DIR}" || true
    fi
    if [[ -d "${BASELINE_RL_RUN_DIR}" && ! -f "${BASELINE_RL_RUN_DIR}/analysis_metrics.csv" && -f "${BASELINE_RL_RUN_DIR}/logs/train.log" ]]; then
        python openebm/elm/scripts/analyze_rl_run.py "${BASELINE_RL_RUN_DIR}" || true
    fi

    mkdir -p "${REPORT_DIR}/config"
    cat > "${REPORT_DIR}/config/fusion_rjob_metadata.txt" << EOF_META
fusion_branch=${FUSION_BRANCH}
fusion_commit=${FUSION_COMMIT}
repo_root=${LOCAL_REPO_ROOT}
sft_checkpoint=${SFT_CKPT}
gsm8k_data_path=${GSM8K_DATA_PATH}
exp_id=${EXP_ID}
run_root=${RUN_ROOT}
analysis_report=${REPORT_DIR}/analysis_report.md
analysis_metrics=${REPORT_DIR}/analysis_metrics.csv
baseline_rl_run_dir=${BASELINE_RL_RUN_DIR}
num_gpus=${NUM_GPUS}
num_generations=${NUM_GENERATIONS}
rl_loss_type=${RL_LOSS_TYPE}
max_steps=${MAX_STEPS}
temperature=${TEMPERATURE}
top_p=${TOP_P}
learning_rate=${LEARNING_RATE}
beta=${BETA}
energy_kl_mode=${ENERGY_KL_MODE}
skip_degenerate_threshold=${SKIP_DEGENERATE_THRESHOLD}
min_reward_std_to_update=${MIN_REWARD_STD_TO_UPDATE}
min_unique_completion_ratio_to_update=${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE}
EOF_META

    echo "Analysis report: ${REPORT_DIR}/analysis_report.md"
    echo "Metrics CSV:      ${REPORT_DIR}/analysis_metrics.csv"
    echo "Metadata:         ${REPORT_DIR}/config/fusion_rjob_metadata.txt"
}

submit_rjob() {
    git -C "${LOCAL_REPO_ROOT}" cat-file -e "${FUSION_COMMIT}^{commit}"

    JOB_COMMAND='
set -euo pipefail
if [[ -e "${JOB_WORKTREE}" ]]; then
    echo "Job worktree already exists: ${JOB_WORKTREE}" >&2
    exit 1
fi
cd "${REPO_ROOT}"
git worktree add --detach "${JOB_WORKTREE}" "${FUSION_COMMIT}"
cd "${JOB_WORKTREE}"
INSIDE_RJOB=1 bash openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh
'

    echo "Submitting rjob:"
    echo "  name:          ${RJOB_NAME}"
    echo "  fusion_branch: ${FUSION_BRANCH}"
    echo "  fusion_commit: ${FUSION_COMMIT}"
    echo "  exp_id:        ${RUN_ID}"
    echo "  expected report:"
    echo "    ${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_report.md"

    if [[ "${DRY_RUN:-0}" = "1" ]]; then
        echo "DRY_RUN=1, not submitting."
        return 0
    fi

    rjob submit \
      --name="${RJOB_NAME}" \
      --gpu="${RJOB_GPU}" \
      --memory="${RJOB_MEMORY}" \
      --cpu="${RJOB_CPU}" \
      --charged-group="${RJOB_CHARGED_GROUP}" \
      --private-machine="${RJOB_PRIVATE_MACHINE}" \
      --host-network=false \
      --image="${RJOB_IMAGE}" \
      --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
      --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
      -e DISTRIBUTED_JOB=true \
      -e REPO_ROOT="${REPO_ROOT}" \
      -e JOB_WORKTREE="${JOB_WORKTREE}" \
      -e FUSION_BRANCH="${FUSION_BRANCH}" \
      -e FUSION_COMMIT="${FUSION_COMMIT}" \
      -e RUN_ID="${RUN_ID}" \
      -e EBT_RUNS_ROOT="${EBT_RUNS_ROOT}" \
      -e GSM8K_DATA_PATH="${GSM8K_DATA_PATH}" \
      -e SFT_CKPT="${SFT_CKPT}" \
      -e BASELINE_RL_RUN_DIR="${BASELINE_RL_RUN_DIR}" \
      -e CONDA_ENV="${CONDA_ENV}" \
      -e NUM_GPUS="${NUM_GPUS}" \
      -e NUM_GENERATIONS="${NUM_GENERATIONS}" \
      -e MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
      -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
      -e TEMPERATURE="${TEMPERATURE}" \
      -e TOP_P="${TOP_P}" \
      -e GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE}" \
      -e RL_LOSS_TYPE="${RL_LOSS_TYPE}" \
      -e MAX_STEPS="${MAX_STEPS}" \
      -e VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL}" \
      -e LOG_INTERVAL="${LOG_INTERVAL}" \
      -e SAVE_TOP_K="${SAVE_TOP_K}" \
      -e LEARNING_RATE="${LEARNING_RATE}" \
      -e BETA="${BETA}" \
      -e ENERGY_KL_MODE="${ENERGY_KL_MODE}" \
      -e ENERGY_KL_HUBER_DELTA="${ENERGY_KL_HUBER_DELTA}" \
      -e SKIP_DEGENERATE_THRESHOLD="${SKIP_DEGENERATE_THRESHOLD}" \
      -e MIN_REWARD_STD_TO_UPDATE="${MIN_REWARD_STD_TO_UPDATE}" \
      -e MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE="${MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE}" \
      -e RL_OPTIMIZER="${RL_OPTIMIZER}" \
      -e ANALYZE_AFTER_TRAIN="${ANALYZE_AFTER_TRAIN}" \
      --custom-resources brainpp.cn/fuse=1 \
      --custom-resources rdma/mlnx_shared=8 \
      --custom-resources mellanox.com/mlnx_rdma=1 \
      -- bash -exc "${JOB_COMMAND}"
}

if [[ "${INSIDE_RJOB:-0}" = "1" ]]; then
    run_inside_rjob
else
    submit_rjob
fi
