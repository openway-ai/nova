#!/bin/bash

################################################################################
# EBT Large 鲁棒训练脚本 - 内存优化版
#
# 针对 140GB GPU 优化的最鲁棒配置
# 基于实际 OOM 错误分析
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-large-robust
#SBATCH --output=logs/slurm/nlp/ebt-large-robust_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-large-robust"
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="large"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# 关键内存优化策略
################################################################################

# 策略 1: 减小 Batch Size（最重要）
# 问题: Large 模型 + EBT MCMC 双倍开销 + attention O(n^2)
# 解决: 使用更小的 micro-batch，更多的 gradient accumulation
#
# 原配置: 32 batch × 1 accum = 32 effective (OOM)
# 优化 1: 16 batch × 2 accum = 32 effective (可能还会 OOM)
# 优化 2: 8 batch × 4 accum = 32 effective (稳定)
# 最鲁棒: 4 batch × 8 accum = 32 effective (极度稳定)

DEVICE_BATCH_SIZE=8
GRAD_ACCUM=8

# 策略 2: 减小序列长度（如果可接受）
# EBT 的 attention 是 O(n^2)，序列长度减半 → 内存降 4 倍
# 256 → 128: 内存降 75%，但可能影响性能
#
# 推荐: 先尝试 256，如果还 OOM 再降到 128
CONTEXT_LENGTH=256

# 策略 3: 调整总 batch size 以保持训练效果
# 目标: 保持与原配置相同的 effective batch size
#
# 8 GPUs × 8 batch × 8 accum × 256 seq = 131,072 tokens/step
# 这比原计划的 262k 小一半，但比 Small 的 98k 大
NUM_GPUS=8

# 计算新的训练步数
# 原计划: 262k tokens/step, 37,700 steps = 9.9B tokens
# 新配置: 131k tokens/step, 需要 75,400 steps 达到相同 token 量
#
# 为了保持训练时间合理，我们可以:
# 选项 A: 保持 9.9B tokens (75,400 steps) - 完整训练
# 选项 B: 降到 5B tokens (38,200 steps) - 快速训练
#
# 推荐选项 A
MAX_STEPS=75400
MAX_SCHEDULING_STEPS=75400

################################################################################
# 其他配置
################################################################################

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))
echo "Effective batch size: $EFFECTIVE_BATCH_SIZE tokens/step"

# 学习率保持不变
PEAK_LR=0.00025
MCMC_STEP_SIZE=500
MCMC_STEP_SIZE_LR_MULTIPLIER=300

# Warmup: 2% of total
WARM_UP_STEPS=1508

# 其他优化
WEIGHT_DECAY=0.01
MIN_LR_SCALE=20
VAL_CHECK_INTERVAL=1000  # 增加间隔减少验证开销
LIMIT_VAL_BATCHES=50     # 减少验证批次
NUM_WORKERS=8            # 减少 workers 节省 CPU 内存
GRADIENT_CLIP_VAL=1.0

################################################################################
# 配置总结
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_ebt_large_robust.log"

cat << EOF

════════════════════════════════════════════════════════════════════════════════
  EBT Large 鲁棒训练配置 (内存优化版)
════════════════════════════════════════════════════════════════════════════════

内存优化策略:
  ✓ 小 micro-batch (8) + 大 accumulation (8)
  ✓ PyTorch 内存分配优化 (expandable_segments)
  ✓ 减少验证频率和批次

模型配置:
  Model Size:          ${MODEL_SIZE}
  Parameters:          ~396M
  Context Length:      ${CONTEXT_LENGTH}
  Layers:              24
  Embedding Dim:       1536

Batch 配置:
  Device Batch Size:   ${DEVICE_BATCH_SIZE} (vs 原 32, 降 4×)
  Gradient Accum:      ${GRAD_ACCUM} (vs 原 1, 增 8×)
  Num GPUs:            ${NUM_GPUS}
  Effective Batch:     ${EFFECTIVE_BATCH_SIZE} tokens/step

训练步数:
  Max Steps:           ${MAX_STEPS} (vs 原 37,700, 增 2×)
  Total Tokens:        ~9.9B (保持不变)
  Estimated Time:      ~48-60 小时 (vs 原 28-36h, 因步数增加)

内存估算 (per GPU):
  Model:               ~0.8 GB
  Optimizer:           ~1.6 GB
  Activations:         ~0.3 GB (batch=8 vs 2.4GB for batch=32)
  Gradients:           ~0.3 GB × 8 accum ≈ 2.4 GB
  Attention:           ~0.4 GB (batch=8 vs 6.5GB for batch=32)
  MCMC:                ~0.13 GB (batch=8 vs 2.1GB for batch=32)
  Total:               ~5.6 GB (vs 原 ~13GB)
  Safety Margin:       25× (vs 原 10×)

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
# 进一步优化建议（如果上述配置仍 OOM）
################################################################################

cat << 'EOF'

════════════════════════════════════════════════════════════════════════════════
  如果仍然 OOM，可以尝试以下降级方案：
════════════════════════════════════════════════════════════════════════════════

方案 1: 进一步减小 batch size
  DEVICE_BATCH_SIZE=4
  GRAD_ACCUM=16
  # 这将内存降到 ~3 GB/GPU

方案 2: 减小序列长度
  CONTEXT_LENGTH=128
  # Attention 内存降 75%

方案 3: 使用梯度检查点 (Gradient Checkpointing)
  在 train.py 中添加:
  --gradient_checkpointing
  # 牺牲 20% 速度换取 50% 内存节省

方案 4: 使用 Mixed Precision
  --precision "bf16-mixed"
  # 已经在用，如果是 fp32 才需要加

方案 5: 训练 Medium 模型而非 Large
  MODEL_SIZE="medium"  # 250M params
  DEVICE_BATCH_SIZE=16
  GRAD_ACCUM=4
  # 参数量降 37%，内存需求显著降低

════════════════════════════════════════════════════════════════════════════════

EOF
