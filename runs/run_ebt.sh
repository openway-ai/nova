### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
source .venv/bin/activate

cd nova/ebt
### LOG INFO ###
#SBATCH --job-name=ebt-xxs-bs_256_s1_lr_
#SBATCH --output=logs/slurm/nlp/ebt-xxs-bs_256_s1_lr_%A-%a.log
# export RUN_NAME="ebt-xxs-bs_256_s1_lr_"

# export RUN_NAME="ebt-medium-bs_256_s1_lr_"
export RUN_NAME="ebt-small-bs_256_s1_lr_"


# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"

HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# export WANDB_API_KEY="<Your Wandb API>"
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"  # Replace with your key
export WANDB_MODE="offline"  # Set to "online" if you want real-time logging

mkdir -p logs/slurm/nlp/
module purge

lr=(0.0012)
alpha=(500)
alpha_lr=(1500)

current_time=$(date +"%Y%m%d_%H%M%S")

# --batch_size_per_device 32 \  # xxs
# --batch_size_per_device 16 \  # small

# --gpus "-1" \

# --tokenizer "EleutherAI/gpt-neox-20b" \


torchrun --standalone --nproc_per_node=6 train.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
--tokenizer "/mnt/shared-storage-user/puyuan/code/EBT/gpt-neox-20b-tokenizer" \
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
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "nanochat" \
--num_workers 12 \
--val_every_n_step 1000 \
--val_after_n_step 1 \
--val_steps 1000 \
--val_sanity 0 \
\
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
--wandb_watch \
2>&1 | tee "logs/${current_time}_ebt_small.log"

# ${SLURM_ARRAY_TASK_ID:+--is_slurm_run} \

# --val_after_n_step 0 \
