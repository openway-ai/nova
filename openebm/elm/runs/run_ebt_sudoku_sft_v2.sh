#!/bin/bash

################################################################################
# EBT Sudoku SFT V2 训练脚本
# 改进：RRN 作训练集 + 对称增强 + 60% Sudoku / 40% 通用 SFT 混合
# 修复 v1 过拟合问题（train_loss→0.01 而 valid_loss→1.5）
################################################################################

set -e

### 基础配置 ###


export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

### 预训练权重 ###
# after sft_train c2048
PRETRAIN_CKPT="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-20260426/sft_train.v4/checkpoints/s=step=562-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=1.2264.ckpt"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_SFT_DATA_DIR="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="Your WandB API Key"
export WANDB_MODE="offline"

# Offline 模式
export NANOCHAT_OFFLINE_MODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Sudoku V2 数据缓存目录
export SUDOKU_DATA_DIR_V2="${SUDOKU_DATA_DIR_V2:-/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2}"

# Sudoku 在 sudoku_mixed 数据集中的采样占比 (剩余为 nanochat 通用 SFT)
export SUDOKU_RATIO="${SUDOKU_RATIO:-0.6}"

export EXP_ID="d26-ctx2048-sudoku-mixed-${SUDOKU_RATIO}-20260508"

################################################################################
# EBT 核心超参数 (对齐预训练配置)
################################################################################
MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=750
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置
################################################################################
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
NUM_GPUS=${NUM_GPUS:-8}

# V2 保持 2048 上下文（RRN 样本较长 + 多 prompt 模板 + 通用 SFT 样本更长）
CONTEXT_LENGTH=2048
DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32

################################################################################
# SFT 学习率配置
################################################################################
PEAK_LR=0.00005
SFT_MUON_LR=0.002
SFT_EMBEDDING_LR=0.03
SFT_VOCAB_TO_EMBED_LR=0.001
SFT_SCALAR_LR=0.004

################################################################################
# 优化器配置（v2: 启用 weight decay 抑制过拟合）
################################################################################
WEIGHT_DECAY=0.01
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 训练步数（v2: RRN 训练集较大，可训更久；更频繁验证捕获早停点）
################################################################################
MAX_STEPS=3000
MAX_SCHEDULING_STEPS=3000

################################################################################
# 验证与数据加载配置
################################################################################
VAL_CHECK_INTERVAL=100
LIMIT_VAL_BATCHES=50
NUM_WORKERS=4

################################################################################
# 优化选项配置
################################################################################
OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.05 --warmdown_ratio 0.2 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr ${SFT_MUON_LR} --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr ${SFT_EMBEDDING_LR} --adamw_vocab_to_embed_lr ${SFT_VOCAB_TO_EMBED_LR} --adamw_scalar_lr ${SFT_SCALAR_LR} --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################
COMPILE_FLAGS="--compile_model --compile_mode full"
WANDB_FLAGS=""

################################################################################
# Checkpoint 管理配置
################################################################################
SAVE_TOP_K=2

################################################################################
# 数据集检查
################################################################################
echo ""
echo "=================================="
echo "  检查 Sudoku V2 数据集"
echo "=================================="
echo "数据集路径: ${SUDOKU_DATA_DIR_V2}"

MISSING=()
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_train.pt" ] && MISSING+=("rrn_train.pt")
[ ! -f "${SUDOKU_DATA_DIR_V2}/rrn_val.pt" ] && MISSING+=("rrn_val.pt")
[ ! -f "${SUDOKU_DATA_DIR_V2}/satnet_test.pt" ] && MISSING+=("satnet_test.pt")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "✗ 缺少以下 V2 数据文件:"
    for f in "${MISSING[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "请先运行 V2 数据准备脚本:"
    echo "  python openebm/elm/data/prepare_sudoku_data_v2.py --data_dir ${SUDOKU_DATA_DIR_V2}"
    exit 1
fi

echo "✓ rrn_train.pt"
echo "✓ rrn_val.pt"
echo "✓ satnet_test.pt"

# 检查 checkpoint
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "✗ Checkpoint 不存在: ${PRETRAIN_CKPT}"
    exit 1
fi
echo "✓ Checkpoint: ${PRETRAIN_CKPT}"
echo ""

################################################################################
# 实验目录布局
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/exp_layout.sh"

export PRETRAIN_CKPT="${PRETRAIN_CKPT}"
exp_init_sft "$0"
export RUN_NAME="${EXP_ID}-sudoku-mixed-sft"
export EXP_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

exp_save_hparams "${EXP_DIR}/sft_train" \
    "model_size=${MODEL_SIZE}" \
    "context_length=${CONTEXT_LENGTH}" \
    "peak_lr=${PEAK_LR}" \
    "sft_muon_lr=${SFT_MUON_LR}" \
    "sft_embedding_lr=${SFT_EMBEDDING_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "device_batch_size=${DEVICE_BATCH_SIZE}" \
    "grad_accum=${GRAD_ACCUM}" \
    "num_gpus=${NUM_GPUS}" \
    "max_steps=${MAX_STEPS}" \
    "mcmc_step_size=${MCMC_STEP_SIZE}" \
    "mcmc_lr_multiplier=${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
    "pretrain_ckpt=${PRETRAIN_CKPT}" \
    "optimizer=muon_adamw" \
    "dataset=sudoku_mixed" \
    "sudoku_ratio=${SUDOKU_RATIO}"

LOG_FILE="${EXP_LOG_FILE}"
current_time=$(date +"%Y%m%d_%H%M%S")

################################################################################
# 显示配置
################################################################################
echo "=================================="
echo "  EBT Sudoku SFT V2 训练"
echo "=================================="
echo "Pretrain ckpt: ${PRETRAIN_CKPT}"
echo "Dataset:       sudoku_mixed (sudoku_ratio=${SUDOKU_RATIO}, rest=nanochat SFT)"
echo "Peak LR:       ${PEAK_LR}"
echo "Weight decay:  ${WEIGHT_DECAY}"
echo "Max steps:     ${MAX_STEPS}"
echo "Val interval:  ${VAL_CHECK_INTERVAL}"
echo "Log file:      ${LOG_FILE}"
echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################
cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT Sudoku SFT V2 训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Pretrain From:   ${PRETRAIN_CKPT}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
################################################################################

LOG_HEADER

{
echo "================================================================================"
echo "[SYSTEM INFO]"
echo "================================================================================"
echo "Hostname:         $(hostname)"
echo "User:             $(whoami)"
echo "Python:           $(python3 --version 2>&1)"
echo "PyTorch:          $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "CUDA Available:   $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU Count:        $(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""
echo "================================================================================"
echo "[Sudoku SFT V2 训练配置]"
echo "================================================================================"
echo "Pretrain Checkpoint: ${PRETRAIN_CKPT}"
echo "Dataset:             sudoku_mixed (sudoku_ratio=${SUDOKU_RATIO})"
echo "Peak LR:             ${PEAK_LR}"
echo "Weight Decay:        ${WEIGHT_DECAY}"
echo "Max Steps:           ${MAX_STEPS}"
echo "Val Check Interval:  ${VAL_CHECK_INTERVAL}"
echo ""
echo "================================================================================"
echo "[开始训练]"
echo "================================================================================"
echo ""

set -e

torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/train.py \
--run_name ${RUN_NAME}_${current_time} \
--checkpoint_dir "${EXP_CKPT_DIR}" \
--wandb_save_dir "${EXP_WANDB_DIR}" \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
--pretokenize_dataset \
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
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--dataset_name "sudoku_mixed" \
--sudoku_ratio ${SUDOKU_RATIO} \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.1 \
--wandb_project 'nlp_sudoku_sft_v2' \
--log_model_archi \
--set_matmul_precision "medium" \
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps ${VAL_CHECK_INTERVAL} \
--float_precision "bf16-mixed" \
--finetuning_model_ckpt ${PRETRAIN_CKPT} \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

exp_save_status "${EXP_DIR}/sft_train" "sudoku_sft_v2_train" "$TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\033[0;32m✓ Sudoku SFT V2 训练成功完成\033[0m"
else
    echo -e "\033[0;31m✗ Sudoku SFT V2 训练异常退出 (exit code: $TRAIN_EXIT_CODE)\033[0m"
fi

} 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush() }' | tee -a "${LOG_FILE}"

echo ""
echo "日志已保存到: ${LOG_FILE}"
