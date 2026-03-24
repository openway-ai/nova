# EBT CORE 评估脚本使用指南

## 概述

`eval_ebt_core.sh` 是用于评估 EBT 模型在 CORE benchmark 上性能的脚本。已优化支持：

- ✅ 灵活配置每个 benchmark 的样本数
- ✅ 自动检测和使用所有可用 GPU
- ✅ 评估完整数据集以获得准确的 CORE score
- ✅ 支持快速测试模式

## 快速开始

### 1. 完整评估（推荐用于正式评估）

评估所有样本以获得完整、准确的 CORE score：

```bash
cd /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/runs

# 使用所有检测到的 GPU，评估所有样本
CKPT_PATH="/path/to/checkpoint.ckpt" \
MAX_PER_TASK=-1 \
bash eval_ebt_core.sh
```

### 2. 快速测试模式

用于快速验证模型或调试：

```bash
# 每个任务评估 100 个样本
CKPT_PATH="/path/to/checkpoint.ckpt" \
MAX_PER_TASK=100 \
bash eval_ebt_core.sh
```

### 3. 指定 GPU 数量

```bash
# 使用 4 个 GPU
CKPT_PATH="/path/to/checkpoint.ckpt" \
NUM_GPUS=4 \
bash eval_ebt_core.sh

# 使用 1 个 GPU
CKPT_PATH="/path/to/checkpoint.ckpt" \
NUM_GPUS=1 \
bash eval_ebt_core.sh
```

## 高级配置

### 为不同任务设置不同样本数

某些任务可能需要更多样本来获得稳定结果，可以为每个任务单独配置：

```bash
CKPT_PATH="/path/to/checkpoint.ckpt" \
MAX_PER_TASK=50 \
TASK_SAMPLES="hellaswag_zeroshot:500,arc_easy:200,winogrande:300" \
bash eval_ebt_core.sh
```

**说明：**
- `MAX_PER_TASK=50`: 全局默认每个任务 50 个样本
- `TASK_SAMPLES`: 为特定任务覆盖默认值
  - `hellaswag_zeroshot` 使用 500 个样本
  - `arc_easy` 使用 200 个样本
  - `winogrande` 使用 300 个样本
  - 其他任务使用全局默认 50 个样本

### 完整配置示例

```bash
CKPT_PATH="/path/to/checkpoint.ckpt" \
MAX_PER_TASK=100 \
TASK_SAMPLES="hellaswag_zeroshot:-1,arc_challenge:-1" \
NUM_GPUS=8 \
DEVICE_BATCH_SIZE=32 \
EVAL_MODES="core" \
bash eval_ebt_core.sh
```

**说明：**
- 大部分任务评估 100 个样本
- `hellaswag_zeroshot` 和 `arc_challenge` 评估所有样本（-1）
- 使用 8 个 GPU
- 每个设备 batch size 为 32

## 配置参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `CKPT_PATH` | EBT checkpoint 路径 | `/path/to/model.ckpt` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_PER_TASK` | `-1` | 全局默认每个任务的样本数<br>`-1` = 所有样本（完整评估）<br>`> 0` = 指定数量 |
| `TASK_SAMPLES` | `""` | 任务特定样本数<br>格式: `"task1:num1,task2:num2"` |
| `NUM_GPUS` | `-1` | GPU 数量<br>`-1` = 自动检测所有可用 GPU<br>`> 0` = 指定数量 |
| `DEVICE_BATCH_SIZE` | `16` | 每个设备的 batch size |
| `EVAL_MODES` | `"core"` | 评估模式（目前仅支持 core） |
| `TOKENIZER_PATH` | 自动 | Tokenizer 路径（通常不需要修改） |

## 输出文件

评估完成后，结果保存在：

```
logs/core_eval/<run_name>/<checkpoint_name>/
├── core_results.json       # JSON 格式的完整结果
└── core_results.csv        # CSV 格式的结果表格
```

日志文件：
```
logs/core_eval_<timestamp>.log
```

## 评估时间估算

基于 GTX 4090，每个任务的评估时间：

| 任务类型 | 样本数 | 估算时间 |
|---------|--------|---------|
| 小型任务 (< 1000 样本) | 全部 | 1-5 分钟 |
| 中型任务 (1000-5000 样本) | 全部 | 5-20 分钟 |
| 大型任务 (> 10000 样本) | 全部 | 20-60 分钟 |

**完整 CORE 评估**（22 个任务，所有样本）：
- 单 GPU: 约 4-8 小时
- 8 GPU: 约 30-60 分钟（多 GPU 加速尚未实现）

## 常见任务列表

CORE benchmark 包含 22 个任务：

1. `hellaswag_zeroshot` (10,042 样本)
2. `arc_easy` (2,376 样本)
3. `arc_challenge` (1,172 样本)
4. `winogrande` (1,267 样本)
5. `boolq` (3,270 样本)
6. `piqa` (1,838 样本)
7. `siqa` (1,954 样本)
8. `openbookqa` (500 样本)
9. `copa` (100 样本)
10. ... (更多任务)

## 使用建议

### 开发/调试阶段
```bash
# 快速测试（每个任务 10 个样本）
MAX_PER_TASK=10 bash eval_ebt_core.sh
```

### 初步验证阶段
```bash
# 中等规模测试（每个任务 100-500 个样本）
MAX_PER_TASK=100 bash eval_ebt_core.sh
```

### 正式评估阶段
```bash
# 完整评估（所有样本）
MAX_PER_TASK=-1 bash eval_ebt_core.sh
```

### 混合策略（推荐）
```bash
# 大型任务采样，小型任务全部评估
MAX_PER_TASK=1000 \
TASK_SAMPLES="copa:-1,openbookqa:-1,rte:-1" \
bash eval_ebt_core.sh
```

## 故障排查

### 错误：找不到 eval_bundle

```bash
# 下载评估数据集
cd /mnt/shared-storage-user/puyuan/code/nanochat
bash runs/download_eval_bundle.sh
```

### 错误：CUDA out of memory

减小 batch size：
```bash
DEVICE_BATCH_SIZE=8 bash eval_ebt_core.sh
```

### 错误：Checkpoint 不存在

检查路径是否正确：
```bash
ls -lh /path/to/checkpoint.ckpt
```

## 示例：完整工作流程

```bash
# 1. 设置 checkpoint 路径
export CKPT_PATH="/mnt/shared-storage-user/puyuan/code/nova/nova/ebt/logs/checkpoints/ebt-small-bs_256_s1_lr_0.0012_2026-03-05_01-01-55_/last.ckpt"

# 2. 快速测试（验证环境）
MAX_PER_TASK=10 bash eval_ebt_core.sh

# 3. 中等规模测试（初步评估）
MAX_PER_TASK=100 bash eval_ebt_core.sh

# 4. 完整评估（正式结果）
MAX_PER_TASK=-1 NUM_GPUS=-1 bash eval_ebt_core.sh
```

## 结果解读

### CORE Metric

CORE metric 是所有任务的 centered accuracy 的平均值，范围 [0, 1]：

- `< 0.3`: 性能较低
- `0.3 - 0.5`: 中等性能
- `0.5 - 0.7`: 良好性能
- `> 0.7`: 优秀性能

### Centered Accuracy

Centered accuracy 是相对于随机基线的标准化准确率：

```
centered_acc = (acc - random_baseline) / (1 - random_baseline)
```

这样可以公平地比较不同任务（不同的随机基线）。

## 技术说明

### 多 GPU 支持

当前版本会自动检测所有可用 GPU，但实际推理仍在单个 GPU 上进行。未来版本将支持：
- 数据并行（多个 GPU 同时处理不同样本）
- 任务并行（不同 GPU 评估不同任务）

### 样本采样策略

使用固定随机种子（1337）确保可重复性：
- 相同的 `MAX_PER_TASK` 设置会得到相同的样本子集
- 样本随机打乱后按顺序选取前 N 个
