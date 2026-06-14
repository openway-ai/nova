#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the SATNet Sudoku EBM-vs-LLM comparison.
#
# Common overrides:
#   DRY_RUN=1 NUM_SAMPLES=16 RUN_R1=0 RUN_QWEN27=0 bash run_all_sudoku_compare.sh
#   PYTHON=/path/to/python RESULTS_ROOT=/path/to/out bash run_all_sudoku_compare.sh
#
# NUM_SAMPLES=-1 means full test split. Any non-negative value runs a smoke subset.

REPO_ROOT="${REPO_ROOT:-/mnt/shared-storage-user/puyuan/code/OpenEBM}"
PYTHON_BIN="${PYTHON:-/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/openebm/elm/data/sudoku_cache_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/openebm/elm/runs/sudoku_compare}"
EBM_OUT_DIR="${EBM_OUT_DIR:-${RESULTS_ROOT}/ebm}"
LLM_OUT_DIR="${LLM_OUT_DIR:-${RESULTS_ROOT}/llm}"
LOG_DIR="${LOG_DIR:-${RESULTS_ROOT}/logs}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MPLCONFIGDIR

EBM_CKPT="${EBM_CKPT:-/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-v3p1-20260529/sft_train.v4/checkpoints/s=step=4424-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss_balanced=valid_loss_balanced=0.2667.ckpt}"

RUN_EBM="${RUN_EBM:-1}"
RUN_LLMS="${RUN_LLMS:-1}"
RUN_SMALL="${RUN_SMALL:-1}"
RUN_QWEN27="${RUN_QWEN27:-1}"
RUN_R1="${RUN_R1:-1}"
RUN_REPORTS="${RUN_REPORTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
RESPONSE_LOG="${RESPONSE_LOG:-truncated}"
RESPONSE_LOG_CHARS="${RESPONSE_LOG_CHARS:-12000}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"

SMALL_MODELS="${SMALL_MODELS:-qwen3_1p7b llama3p2_1b}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-8}"
SMALL_TP="${SMALL_TP:-1}"
SMALL_MAX_MODEL_LEN="${SMALL_MAX_MODEL_LEN:-}"

QWEN27_BATCH_SIZE="${QWEN27_BATCH_SIZE:-2}"
QWEN27_TP="${QWEN27_TP:-4}"
QWEN27_MAX_MODEL_LEN="${QWEN27_MAX_MODEL_LEN:-16384}"

R1_BATCH_SIZE="${R1_BATCH_SIZE:-1}"
R1_TP="${R1_TP:-8}"
R1_MAX_MODEL_LEN="${R1_MAX_MODEL_LEN:-32768}"
R1_GPU_MEMORY_UTILIZATION="${R1_GPU_MEMORY_UTILIZATION:-0.95}"

mkdir -p "${LOG_DIR}" "${EBM_OUT_DIR}" "${LLM_OUT_DIR}"
cd "${REPO_ROOT}"

run_cmd_log() {
  local name="$1"
  shift
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

append_num_samples_args() {
  local -n out_ref=$1
  if [[ "${NUM_SAMPLES}" != "-1" ]]; then
    out_ref+=(--num-samples "${NUM_SAMPLES}")
  fi
}

append_common_llm_args() {
  local -n out_ref=$1
  out_ref+=(
    --data-dir "${DATA_DIR}"
    --out-dir "${LLM_OUT_DIR}"
    --backend vllm
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --resume
    --response-log "${RESPONSE_LOG}"
    --response-log-chars "${RESPONSE_LOG_CHARS}"
  )
  append_num_samples_args "$1"
  if [[ "${SAVE_PROMPTS}" == "1" ]]; then
    out_ref+=(--save-prompts)
  fi
}

run_llm_group() {
  local log_name="$1"
  local models_string="$2"
  local batch_size="$3"
  local tp_size="$4"
  local max_model_len="$5"
  local gpu_memory_utilization="${6:-}"

  read -r -a model_array <<< "${models_string}"
  local cmd=("${PYTHON_BIN}" -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku)
  cmd+=(--models "${model_array[@]}")
  append_common_llm_args cmd
  cmd+=(--batch-size "${batch_size}" --tensor-parallel-size "${tp_size}")
  if [[ -n "${max_model_len}" ]]; then
    cmd+=(--max-model-len "${max_model_len}")
  fi
  if [[ -n "${gpu_memory_utilization}" ]]; then
    cmd+=(--gpu-memory-utilization "${gpu_memory_utilization}")
  fi
  run_cmd_log "${log_name}" "${cmd[@]}"
}

if [[ "${RUN_EBM}" == "1" ]]; then
  if [[ "${DRY_RUN}" != "1" && ! -f "${EBM_CKPT}" ]]; then
    echo "[run_all_sudoku_compare] ERROR: EBM checkpoint not found: ${EBM_CKPT}" >&2
    exit 2
  fi
  ebm_sample_args=(--full_test)
  if [[ "${NUM_SAMPLES}" != "-1" ]]; then
    ebm_sample_args=(--num_samples_test "${NUM_SAMPLES}")
  fi
  run_cmd_log ebm_test \
    "${PYTHON_BIN}" -m openebm.elm.scripts.eval_sudoku_samples \
    -c "${EBM_CKPT}" \
    --data_dir "${DATA_DIR}" \
    --splits test \
    --no_per_sample_print \
    --out_dir "${EBM_OUT_DIR}" \
    "${ebm_sample_args[@]}"
fi

if [[ "${RUN_LLMS}" == "1" ]]; then
  if [[ "${RUN_SMALL}" == "1" ]]; then
    run_llm_group llm_small "${SMALL_MODELS}" "${SMALL_BATCH_SIZE}" "${SMALL_TP}" "${SMALL_MAX_MODEL_LEN}"
  fi
  if [[ "${RUN_QWEN27}" == "1" ]]; then
    run_llm_group llm_qwen27 "qwen3p6_27b" "${QWEN27_BATCH_SIZE}" "${QWEN27_TP}" "${QWEN27_MAX_MODEL_LEN}"
  fi
  if [[ "${RUN_R1}" == "1" ]]; then
    run_llm_group llm_r1 "deepseek_r1_0528" "${R1_BATCH_SIZE}" "${R1_TP}" "${R1_MAX_MODEL_LEN}" "${R1_GPU_MEMORY_UTILIZATION}"
  fi
fi

if [[ "${RUN_REPORTS}" == "1" ]]; then
  run_cmd_log aggregate \
    "${PYTHON_BIN}" -m openebm.elm.scripts.sudoku_compare.aggregate_results \
    --results-root "${RESULTS_ROOT}"

  run_cmd_log case_study \
    "${PYTHON_BIN}" -m openebm.elm.scripts.sudoku_compare.case_study \
    --results-root "${RESULTS_ROOT}" \
    --data-dir "${DATA_DIR}"
fi

echo "[run_all_sudoku_compare] done. Outputs: ${RESULTS_ROOT}"
