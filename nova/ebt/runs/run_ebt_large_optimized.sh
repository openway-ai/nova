#!/bin/bash

################################################################################
# EBT Large 优化训练脚本
#
# 基于 TRAINING_CONFIG_COMPARISON.md 的分析和建议
# 参考 NanoChat 的科学训练方法（Chinchilla scaling laws）
#
# 模型配置:
#   - Large: 396M 参数 (24 layers, 1536 dim, 16 heads)
#   - Context length: 256
#
# 训练策略:
#   - 使用全部 370 个 shards (369 train + 1 val)
#   - 基于 Chinchilla 定律计算训练步数
#   - 参考 NanoChat 的 batch size 和学习率策略
################################################################################

### SLURM 配置 (如果使用 SLURM) ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-large-optimized
#SBATCH --output=logs/slurm/nlp/ebt-large-optimized_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-large-optimized"
export MODEL_NAME="${RUN_NAME%%-*}"  # "ebt"
export MODEL_SIZE="large"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"  # 改为 "online" 启用实时同步

# 创建日志目录
mkdir -p logs/slurm/nlp/
module purge

################################################################################
# 训练超参数（基于 Chinchilla 定律和 NanoChat 的最佳实践）
################################################################################

# 模型参数量计算:
# Large: 24 layers × 1536 dim × 16 heads ≈ 396M params
#
# Chinchilla 建议: data:param = 20:1
# 保守策略: data:param = 12:1 (略微 overtrain，类似 NanoChat 的 8.25)
#
# 目标 tokens: 396M × 12 = 4.75B tokens
#
# NanoChat 全部 370 shards ≈ 20.5B characters ≈ 10B tokens
# 我们使用 370 shards，训练到 4.75B tokens

# Batch Size 配置（参考 NanoChat）
# NanoChat: 8 GPUs × 16 per-device × 4 grad-accum × 2048 seq = 1,048,576 tokens/step
#
# EBT 调整:
# - Context length 256 (比 NanoChat 的 2048 小 8 倍)
# - 为了达到类似的 batch size，需要 8 倍的 micro-batch size
# - 但由于 EBT 的 MCMC 机制计算量更大，适当减小到 524,288 tokens/step
#
# 配置: 8 GPUs × 64 per-device × 1 grad-accum × 256 seq = 131,072 tokens/step
# 或: 8 GPUs × 32 per-device × 2 grad-accum × 256 seq = 131,072 tokens/step
#
# 为了更接近 NanoChat，我们选择更大的 batch size:
# 8 GPUs × 128 per-device × 1 grad-accum × 256 seq = 262,144 tokens/step

DEVICE_BATCH_SIZE=128
GRAD_ACCUM=1
NUM_GPUS=8
CONTEXT_LENGTH=256

# 计算 effective batch size (tokens per step)
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE tokens/step"

# 训练步数计算
# Total tokens: 4.75B
# Tokens per step: 262,144
# Steps: 4.75B / 262,144 ≈ 18,120 steps
#
# 为了充分利用 370 shards (约 10B tokens)，我们可以训练更长
# 使用 data:param = 25:1 (介于 Chinchilla 20 和 conservative 10 之间)
# Total tokens: 396M × 25 = 9.9B tokens ≈ 全部 370 shards
# Steps: 9.9B / 262,144 ≈ 37,700 steps

MAX_STEPS=37700
MAX_SCHEDULING_STEPS=37700

# 学习率配置（参考 NanoChat 的策略）
# NanoChat Large (1.68B params):
#   - Matrix LR: 0.02 (Muon)
#   - Batch size scaled: sqrt(batch_size / 524k)
#
# EBT Large (396M params) 基础 LR:
#   - Base LR: 0.00025 (large 模型推荐值，见 utils.py line 81)
#   - Batch size: 262k vs reference 256k
#   - Batch size scaling: sqrt(262k / 256k) = sqrt(1.023) ≈ 1.01
#   - Adjusted LR: 0.00025 × 1.01 ≈ 0.00025 (几乎不变)

PEAK_LR=0.00025

# MCMC 步长学习率
# 原始配置: alpha_lr = base_lr × 1500 = 0.0012 × 1500 = 1.8
# 文档建议: 降低到 500 以提高稳定性
# Large 模型: 进一步降低到 300
MCMC_STEP_SIZE=500
MCMC_STEP_SIZE_LR_MULTIPLIER=300

# Warmup 配置
# NanoChat: warmup_ratio = 0.0 (无 warmup)
# EBT 原始: 10k steps (1% of 1M)
# 优化后: 2% of total steps = 0.02 × 37700 ≈ 750 steps
WARM_UP_STEPS=750

# 权重衰减
# NanoChat: 0.2 (Muon), depth-scaled
# EBT: 0.01 (AdamW)
# 保持 EBT 的设置
WEIGHT_DECAY=0.01

# 最小学习率比例
# NanoChat: final_lr_frac = 0.0
# EBT: min_lr_scale = 10 (即 final_lr = peak_lr / 10)
# 采用更温和的衰减: min_lr_scale = 20
MIN_LR_SCALE=20

################################################################################
# 验证和检查点配置
################################################################################

# 验证频率
# 原始: 每 1000 步
# 优化: 每 500 步（更频繁监控）
VAL_CHECK_INTERVAL=500

# 限制验证批次（加快验证速度）
LIMIT_VAL_BATCHES=100

# 其他配置
NUM_WORKERS=12
GRADIENT_CLIP_VAL=1.0

################################################################################
# 日志配置
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_ebt_large_optimized.log"

################################################################################
# 训练配置总结
################################################################################

cat << EOF

════════════════════════════════════════════════════════════════════════════════
  EBT Large 优化训练配置
════════════════════════════════════════════════════════════════════════════════

模型配置:
  Model Name:          ${MODEL_NAME}
  Model Size:          ${MODEL_SIZE}
  Layers:              24
  Embedding Dim:       1536
  Attention Heads:     16
  Parameters:          ~396M
  Context Length:      ${CONTEXT_LENGTH}

训练数据:
  Dataset:             NanoChat (370 shards)
  Total Tokens:        ~9.9B tokens
  Data:Param Ratio:    25:1 (Chinchilla 建议 20:1)

Batch 配置:
  Device Batch Size:   ${DEVICE_BATCH_SIZE}
  Gradient Accum:      ${GRAD_ACCUM}
  Num GPUs:            ${NUM_GPUS}
  Effective Batch:     ${EFFECTIVE_BATCH_SIZE} tokens/step

训练步数:
  Max Steps:           ${MAX_STEPS}
  Estimated Time:      ~24-36 小时 (8×H100)

学习率:
  Peak LR:             ${PEAK_LR}
  MCMC Step Size:      ${MCMC_STEP_SIZE}
  MCMC LR Multiplier:  ${MCMC_STEP_SIZE_LR_MULTIPLIER}
  Weight Decay:        ${WEIGHT_DECAY}
  Warmup Steps:        ${WARM_UP_STEPS}
  Min LR Scale:        ${MIN_LR_SCALE}

验证配置:
  Val Check Interval:  ${VAL_CHECK_INTERVAL}
  Limit Val Batches:   ${LIMIT_VAL_BATCHES}

输出:
  Log File:            ${LOG_FILE}

════════════════════════════════════════════════════════════════════════════════

EOF

read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 启动训练
################################################################################

torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/train.py \
--run_name ${RUN_NAME}_${current_time} \
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
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps 2 \
\
--context_length ${CONTEXT_LENGTH} \
\
--gpus "-1" \
\
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
\
--weight_decay ${WEIGHT_DECAY} \
--min_lr_scale ${MIN_LR_SCALE} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--warm_up_steps ${WARM_UP_STEPS} \
\
--dataset_name "nanochat" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
\
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
--wandb_watch \
2>&1 | tee "${LOG_FILE}"

################################################################################
# 训练完成后的说明
################################################################################

cat << EOF

════════════════════════════════════════════════════════════════════════════════
  训练完成！
════════════════════════════════════════════════════════════════════════════════

下一步:

1. 评估模型 CORE 性能:

   CKPT_PATH="logs/checkpoints/ebt-large-optimized_*/last.ckpt" \\
   MAX_PER_TASK=-1 \\
   bash runs/eval_ebt_core.sh

2. 与 baseline 对比:

   - Small (85M): ~0.35 CORE score (预期)
   - Large (396M): ~0.45-0.50 CORE score (预期，基于 scaling laws)

3. 如果要继续训练更多步数:

   - 修改 MAX_STEPS 和 MAX_SCHEDULING_STEPS
   - 添加 --resume_from_checkpoint 参数

4. 微调建议:

   如果希望进一步提升性能，可以尝试:
   - 增加 data:param ratio 到 30:1 或 40:1
   - 调整学习率或 warmup 策略
   - 增加 batch size（如果 GPU 内存允许）

════════════════════════════════════════════════════════════════════════════════

EOF
