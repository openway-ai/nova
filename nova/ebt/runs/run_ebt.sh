### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EBT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$EBT_DIR"

# 优先使用项目 venv；没有则回退到当前 conda/python 环境
if [ -f "$EBT_DIR/../../.venv/ebt/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$EBT_DIR/../../.venv/ebt/bin/activate"
fi

if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [ -x "/mnt/shared-storage-user/lixueyan/miniconda3/envs/nanochat/bin/python" ]; then
    PYTHON_BIN="/mnt/shared-storage-user/lixueyan/miniconda3/envs/nanochat/bin/python"
elif [ -x "$(command -v python3)" ]; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

### LOG INFO ###
# SBATCH --job-name=ebt-xxs-bs_256_s1_lr_
# SBATCH --output=logs/slurm/nlp/ebt-xxs-bs_256_s1_lr_%A-%a.log
# export RUN_NAME="ebt-d26-bs_256_s1_lr_"
export RUN_NAME="ebt-xxs-bs_256_s1_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/shared-storage-user/lixueyan/nar}"

# 训练脚本默认 wandb_offline=True，因此不会联网上传；会将 wandb 文件写到本地 logs/。
# 只有你显式传 --wandb_offline False 并配置 WANDB_API_KEY 时才会尝试联网。
mkdir -p logs/slurm/nlp/
if command -v module >/dev/null 2>&1; then
    module purge
fi

lr=(0.0002)
alpha=(500)
alpha_lr=(1500)
MAX_STEP=50000
WARMUP_STEP=1000

"$PYTHON_BIN" train.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 256 \
\
--gpus -1 \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 2 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps ${MAX_STEP} \
--max_scheduling_steps ${MAX_STEP} \
--warm_up_steps ${WARMUP_STEP} \
\
--dataset_name "fineweb" \
--num_workers 12 \
--val_check_interval 1000 \
--limit_val_batches 100 \
--val_sanity 1 \
\
--wandb_project 'nlp_pretrain' \
\
--log_model_archi \
--log_every_n_steps 100 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}