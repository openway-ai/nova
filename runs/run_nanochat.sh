### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
source .venv/nanochat/bin/activate

### LOG INFO ###
# SBATCH --job-name=nanochat-d6
# SBATCH --output=logs/slurm/nlp/nanochat-d6_%A-%a.log
export RUN_NAME="nanochat-d6"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)

export WANDB_API_KEY="wandb_v1_P44yHoBV4louYaIQ5dYJwS0Sdrd"
export WANDB_MODE=offline
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p logs/slurm/nlp/
module purge

DEPTH=6
NUM_ITER=10000
DEVICE_BATCH_SIZE=32

torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} -m scripts.base_train -- \
--run ${RUN_NAME} \
--depth ${DEPTH} \
--device-batch-size ${DEVICE_BATCH_SIZE} \
--num-iterations ${NUM_ITER} \
--window-pattern L \
--eval-every 1000 \
--core-metric-every -1 \
--sample-every -1 \
${SLURM_ARRAY_TASK_ID:+--model-tag ${RUN_NAME}_${SLURM_ARRAY_TASK_ID}}

# A few notes on the choices:

#   - --run nanochat-d6 — not dummy, so wandb will be initialized. Since we're offline, add WANDB_MODE=offline if needed (or change to --run dummy to skip wandb         
#   entirely)
#   - --window-pattern L — full attention instead of sliding window, required since this machine doesn't have FA3 (non-H100), otherwise GPU utilization would be terrible
#   - --num-iterations 10000 — matches EBT's 10K steps                                                                                                                   
#   - --eval-every 1000 — matches EBT's val_check_interval 1000                                                                                                          
#   - --core-metric-every -1 and --sample-every -1 — disabled since they may require network or extra setup 