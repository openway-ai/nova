# EBT 模型评估执行说明

训练流程（base_train / sft_train）完成后，使用以下两套脚本执行评估。

---

## 前置条件

1. **Checkpoint**：base_train 或 sft_train 产出的 `.ckpt` 文件，确认文件存在
2. **eval_bundle 数据**：位于 `~/.cache/nanochat/eval_bundle/`（`HOME` 被脚本重写为 nanochat 目录）。如缺失：
   ```bash
   cd /mnt/shared-storage-user/puyuan/code/nanochat
   bash runs/download_eval_bundle.sh
   ```
3. **Tokenizer**：默认路径 `/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer`
4. **Conda 环境**：`conda activate nanochat`
5. **硬件**：至少 1 张 GPU。脚本会自动检测所有可用 GPU 并行评估

---

## 两套评估对比

| | eval_ebt_core.sh | eval_ebt_nanochat_gsm8k.sh |
|---|---|---|
| **评估内容** | CORE benchmark（多选/schema/LM 类任务） | NanoChat 分片 PPL + GSM8K 数学推理 |
| **核心指标** | CORE Metric（centered accuracy 均值） | PPL / Loss + GSM8K Accuracy |
| **Python 入口** | `openebm.elm.scripts.ebt_core_eval` | `openebm.elm.train --only_test` |
| **多 GPU** | mp.spawn 并行，贪心调度任务 | PyTorch Lightning DDP |
| **输出格式** | `core_results.json` / `.csv` / trajectory JSONL | `results.jsonl` + 日志 |
| **典型耗时** | 数十分钟（取决于任务数和样本量） | 数分钟 |
| **适用场景** | 全面能力评估，跨模型对比 | 快速检查语言建模和数学推理能力 |

---

## 评估一：CORE Benchmark（eval_ebt_core.sh）

### 关键参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CKPT_PATH` | 脚本内硬编码 | checkpoint 路径（**必须修改或通过环境变量传入**） |
| `EVAL_MODES` | `core` | 评估模式 |
| `MAX_PER_TASK` | `-1` | 每个任务的最大样本数，`-1` = 全部（完整 CORE score） |
| `TASK_SAMPLES` | 空 | 为特定任务指定样本数，格式 `"task1:100,task2:200"` |
| `DEVICE_BATCH_SIZE` | `24` | 每设备 batch size |
| `NUM_GPUS` | `-1` | GPU 数量，`-1` = 自动检测 |
| `TOKENIZER_PATH` | `.cache/nanochat/tokenizer` | tokenizer 目录 |
| `DTYPE` | `bfloat16` | 推理精度 |
| `EXP_ID` | 空 | 实验 ID（设置后输出到 `ebt_runs/<EXP_ID>/core_eval/`） |

### 执行步骤

```bash
# 1. 进入 repo 根目录
cd /mnt/shared-storage-user/puyuan/code/OpenEBM

# 2. 激活环境
conda activate nanochat

# 3a. 使用 EXP_ID 模式（推荐，输出到统一实验目录）
EXP_ID="d26-ctx2048-20260426" \
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
bash openebm/elm/runs/eval_ebt_core.sh

# 3b. 快速测试（每个任务 100 个样本）
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
MAX_PER_TASK=100 \
bash openebm/elm/runs/eval_ebt_core.sh

# 3c. 指定 GPU 数量
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
NUM_GPUS=4 \
bash openebm/elm/runs/eval_ebt_core.sh
```

### 输出目录结构

**EXP_ID 模式**（设置了 `EXP_ID`）：

```
logs/ebt_runs/<EXP_ID>/core_eval/step<N>_<timestamp>/
├── config/
│   ├── run_script.sh          # 脚本快照
│   ├── eval_params.json       # 评估参数
│   └── eval_ckpt_ref.json     # checkpoint 来源信息
├── results/
│   ├── core_results.json      # 完整结果（accuracy + centered + core_metric）
│   ├── core_results.csv       # CSV 格式结果
│   ├── timing_summary.json    # 耗时统计（也用于下次贪心调度）
│   └── trajectories/          # 各任务的推理轨迹 JSONL
│       └── <task_name>/samples.jsonl
├── eval.log                   # 完整日志
└── status.json                # 退出状态
```

**兼容模式**（未设置 `EXP_ID`）：

```
logs/core_eval/<run_short>_<timestamp>/<ckpt_filename>/
├── core_results.json
├── core_results.csv
├── timing_summary.json
└── trajectories/
```

### 结果查看

```bash
# 查看 CORE 分数
cat results/core_results.json | python -m json.tool

# 查看 CSV
cat results/core_results.csv

# 从日志提取摘要行
grep "EVAL_SUMMARY" eval.log
# 输出示例: [EVAL_SUMMARY] dataset=core core_metric=0.1234 avg_accuracy=0.4567 num_tasks=14 total_time=1234.5s

# 查看各任务得分
grep "CORE Metric:" eval.log
```

**核心指标含义**：
- **Accuracy**：任务原始准确率
- **Centered**：`(accuracy - random_baseline) / (1 - random_baseline)`，消除随机猜测影响
- **CORE Metric**：所有任务 centered accuracy 的算术平均

---

## 评估二：NanoChat PPL + GSM8K（eval_ebt_nanochat_gsm8k.sh）

### 关键参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CKPT_PATH` | 脚本内硬编码 | checkpoint 路径（**必须修改或通过环境变量传入**） |
| `GPUS` | `-1` | GPU 数量，`-1` = 自动检测 |
| `BATCH_SIZE` | `4` | 每设备 batch size |
| `LIMIT_TEST_BATCHES` | `100` | 测试 batch 数量上限 |
| `EVAL_SHARD_INDICES` | `0,15` | NanoChat 评估分片索引 |
| `MAX_SAMPLES_PER_SHARD` | `5`（入口脚本） / `50`（子脚本独立运行） | 每分片样本数 |
| `ENABLE_GENERATION` | `true` | 是否启用文本生成 |
| `GENERATION_SPLIT_RATIO` | `0.5` | 生成时的上下文/续写分割比例 |
| `MIN_GENERATION_LENGTH` | `64` | 最小生成长度（tokens） |
| `EVAL_TASK` | `gsm8k` | GSM8K 评估任务 |
| `USE_WANDB` | `false` | 是否启用 WandB |
| `TOKENIZER_PATH` | `.cache/nanochat/tokenizer` | tokenizer 目录 |
| `EXP_ID` | 空 | 实验 ID（设置后输出到 `ebt_runs/<EXP_ID>/nanochat_gsm8k_eval/`） |

### 执行步骤

```bash
# 1. 进入 repo 根目录
cd /mnt/shared-storage-user/puyuan/code/OpenEBM

# 2. 激活环境
conda activate nanochat

# 3a. 使用 EXP_ID 模式（推荐，输出到统一实验目录）
EXP_ID="d26-ctx2048-20260426" \
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
bash openebm/elm/runs/eval_ebt_nanochat_gsm8k.sh

# 3b. 快速测试（无 EXP_ID）
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
bash openebm/elm/runs/eval_ebt_nanochat_gsm8k.sh

# 或：单独运行 NanoChat 分片评估
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
bash openebm/elm/runs/eval_nanochat_shards.sh

# 或：单独运行 GSM8K 评估
CKPT_PATH="/path/to/your/checkpoint.ckpt" \
EVAL_TASK=gsm8k \
bash openebm/elm/runs/eval_ebt.sh
```

### eval_ebt.sh 支持的任务类型

除 GSM8K 外，`eval_ebt.sh` 还支持以下任务（通过 `EVAL_TASK` 指定）：

| EVAL_TASK | 说明 |
|---|---|
| `nanochat` | NanoChat 数据集 PPL 测试 |
| `gsm8k` | GSM8K 数学推理 |
| `arc` | ARC 科学问答 |
| `humaneval` | HumanEval 代码生成 |
| `mmlu` | MMLU 多任务理解 |
| `smoltalk` | SmolTalk 对话 |
| `spellingbee` | SpellingBee |

### 输出目录结构

**EXP_ID 模式**（设置了 `EXP_ID`）：

```
logs/ebt_runs/<EXP_ID>/nanochat_gsm8k_eval/step<N>_<timestamp>/
├── config/
│   ├── run_script.sh          # 脚本快照
│   └── eval_ckpt_ref.json     # checkpoint 来源信息
├── master.log                 # 主日志（入口脚本全程记录）
├── nanochat_shards.log        # NanoChat 分片评估日志
├── gsm8k.log                  # GSM8K 评估日志
└── NLP/
    ├── nanochat_shard_eval/   # NanoChat 推理结果
    │   └── <run_name>/<ckpt_name>/results.jsonl
    └── gsm8k/                 # GSM8K 推理结果
        └── <run_name>/<ckpt_name>/results.jsonl
```

**兼容模式**（未设置 `EXP_ID`）：

```
logs/nanochat_gsm8k_eval/<run_short>_<timestamp>/
├── master.log                # 主日志（入口脚本全程记录）
├── nanochat_shards.log       # NanoChat 分片评估日志
├── gsm8k.log                 # GSM8K 评估日志
└── NLP/
    ├── nanochat_shard_eval/   # NanoChat 推理结果
    │   └── <run_name>/<ckpt_name>/results.jsonl
    └── gsm8k/                 # GSM8K 推理结果
        └── <run_name>/<ckpt_name>/results.jsonl
```

> 自动清理：兼容模式下，同一模型最多保留最近 10 个运行目录，旧的自动删除。EXP_ID 模式下不自动清理。

### 结果查看

```bash
# 查看汇总报告（在 master.log 末尾）
tail -30 master.log

# NanoChat PPL 指标
grep "TEST EPOCH SUMMARY" nanochat_shards.log -A 10

# GSM8K 准确率
grep "EVAL_SUMMARY" gsm8k.log

# NanoChat PPL 详情
grep "Mean:" nanochat_shards.log
```

---

## 常见问题 / 注意事项

1. **CUBLAS_STATUS_INVALID_VALUE**：脚本已通过 `LD_LIBRARY_PATH` 将 pip 安装的 nvidia cuBLAS 优先于系统库。如仍报错，确认 conda 环境为 `nanochat` 且 pytorch 版本匹配

2. **显存不足 (OOM)**：
   - CORE eval：降低 `DEVICE_BATCH_SIZE`（默认 24，可降到 8 或 4）
   - NanoChat/GSM8K：降低 `BATCH_SIZE`（默认 4，可降到 1）

3. **eval_bundle 不存在**：脚本启动时会检查 `$NANOCHAT_BASE_DIR/eval_bundle/` 是否存在，缺失时会提示下载命令

4. **CKPT_PATH 硬编码**：两个入口脚本都在文件内硬编码了默认 checkpoint 路径。建议通过环境变量覆盖而非修改脚本：
   ```bash
   CKPT_PATH="/your/path.ckpt" bash openebm/elm/runs/eval_ebt_core.sh
   ```

5. **贪心调度**：多 GPU CORE eval 第一次运行使用 round-robin 分配任务，之后会自动读取上次的 `timing_summary.json` 进行贪心调度以均衡 GPU 负载

6. **离线模式**：脚本默认设置 `NANOCHAT_OFFLINE_MODE=1`，不会尝试从网络下载数据

7. **多次评估**：EXP_ID 模式下每次运行会创建新的 `core_eval/step<N>_<timestamp>/` 或 `nanochat_gsm8k_eval/step<N>_<timestamp>/` 子目录，不会覆盖历史结果
