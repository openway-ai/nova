#!/bin/bash
################################################################################
# EBT d26 SFT 训练脚本
#
# 加载预训练 checkpoint，在 NanoChat SFT 数据集上进行监督微调
# 数据: SmolTalk + MMLU + GSM8K + CustomJSON + Spelling
# 训练逻辑与 run_ebt_optimized_0313.sh 一致，仅更换数据集和学习率
################################################################################

set -e

### 基础配置 ###
export RUN_NAME="ebt-d26-sft"
export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NANOCHAT_SFT_DATA_DIR="$NANOCHAT_BASE_DIR/sft_data"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/

################################################################################
# 预训练 Checkpoint
################################################################################
PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "❌ checkpoint 不存在: $PRETRAIN_CKPT"; exit 1
fi

################################################################################
# SFT 数据检查
################################################################################
MISSING=()
[ ! -d "$NANOCHAT_SFT_DATA_DIR/smol-smoltalk" ] && MISSING+=("smol-smoltalk")
[ ! -d "$NANOCHAT_SFT_DATA_DIR/mmlu" ] && MISSING+=("mmlu")
[ ! -d "$NANOCHAT_SFT_DATA_DIR/gsm8k" ] && MISSING+=("gsm8k")
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "❌ 缺少 SFT 数据: ${MISSING[*]}"; exit 1
fi
echo "✓ SFT 数据就绪"

################################################################################
# EBT 核心参数 (与预训练一致)
################################################################################
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
DENOISING_INITIAL_CONDITION="random_noise"

################################################################################
# SFT 训练参数
################################################################################
DEVICE_BATCH_SIZE=8
GRAD_ACCUM=8
CONTEXT_LENGTH=256
NUM_GPUS=8

# SFT 数据 ~856K 条, 以 131k tokens/step 计, ~3000 步 ≈ 1 epoch
MAX_STEPS=5000
MAX_SCHEDULING_STEPS=5000

# SFT 学习率: 预训练的 1/5, 防止灾难性遗忘
PEAK_LR=0.00005
WARM_UP_STEPS=250
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=10

WEIGHT_DECAY=0.01
BETA1=0.9
BETA2=0.999
GRADIENT_CLIP_VAL=1.0

VAL_CHECK_INTERVAL=500
LIMIT_VAL_BATCHES=30
NUM_WORKERS=8

################################################################################
# 日志
################################################################################
current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_ebt_d26_sft.log"
mkdir -p logs

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  EBT d26 SFT (监督微调)"
echo "════════════════════════════════════════════════════════════════"
echo "  Checkpoint:  $PRETRAIN_CKPT"
echo "  数据集:      nanochat_sft (SmolTalk+MMLU+GSM8K+Spelling)"
echo "  Peak LR:     $PEAK_LR (预训练的 1/5)"
echo "  Max Steps:   $MAX_STEPS"
echo "  日志:        $LOG_FILE"
echo "════════════════════════════════════════════════════════════════"
echo ""
read -p "按 Enter 开始，Ctrl+C 取消..."

################################################################################
# 启动训练
################################################################################
set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/train.py \
    --run_name ${RUN_NAME}_${current_time} \
    --modality "NLP" \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --resume_training_ckpt "$PRETRAIN_CKPT" \
    --dataset_name "nanochat_sft" \
    --normalize_initial_condition \
    --ebt_type ${EBT_TYPE} \
    --denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
    --mcmc_step_size_learnable \
    --mcmc_step_size ${MCMC_STEP_SIZE} \
    --mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
    --mcmc_num_steps ${MCMC_NUM_STEPS} \
    --context_length ${CONTEXT_LENGTH} \
    --gpus "-1" \
    --peak_learning_rate ${PEAK_LR} \
    --batch_size_per_device ${DEVICE_BATCH_SIZE} \
    --accumulate_grad_batches ${GRAD_ACCUM} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --weight_decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --min_lr_scale ${MIN_LR_SCALE} \
    --max_steps ${MAX_STEPS} \
    --max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
    --warm_up_steps ${WARM_UP_STEPS} \
    --warm_up_base_lr_divider ${WARM_UP_BASE_LR_DIVIDER} \
    --num_workers ${NUM_WORKERS} \
    --val_check_interval ${VAL_CHECK_INTERVAL} \
    --limit_val_batches ${LIMIT_VAL_BATCHES} \
    --val_sanity 1 \
    --validation_split_pct 0.0027 \
    --wandb_project 'nlp_sft' \
    --log_model_archi \
    --set_matmul_precision "medium" \
    --wandb_watch \
    2>&1 | tee -a "${LOG_FILE}"

TRAIN_EXIT_CODE=$?
set -e

echo ""
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✓ SFT 训练完成 | 日志: ${LOG_FILE}"
else
    echo "✗ SFT 训练失败 (exit=$TRAIN_EXIT_CODE) | 日志: ${LOG_FILE}"
fi
