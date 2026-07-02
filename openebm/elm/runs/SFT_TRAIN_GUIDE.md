# EBT SFT 训练执行说明

基于 `run_ebt_sft.sh`，在 base_train (ctx2048) 完成后进行 SFT 微调。

---

## 前置条件

1. base_train 阶段已完成，有可用的 `.ckpt` 文件
2. SFT 离线数据集已下载到指定路径，包含：
   - `smol-smoltalk/`、`mmlu/`、`gsm8k/` 目录
   - `words_alpha.txt` 文件
3. 如果数据集缺失，在联网环境执行：
   ```bash
   cd /mnt/shared-storage-user/puyuan/code/nanochat
   python runs/download_sft_datasets.py
   ```

---

## 必须修改的参数

打开 `openebm/elm/runs/run_ebt_sft.sh`，修改以下位置：

### 1. EXP_ID（第 13 行）

```bash
export EXP_ID="<你的实验ID>"  # 例如 "d26-ctx2048-20260501"
```

用于组织输出目录：`logs/ebt_runs/<EXP_ID>/sft_train/`。
如果同一 EXP_ID 下已有 `sft_train/`，脚本会自动创建 `sft_train.v2/`、`sft_train.v3/` 等。

### 2. PRETRAIN_CKPT（第 34 行）

```bash
PRETRAIN_CKPT="/你的/base_train/checkpoint路径.ckpt"
```

指向你 base_train 产出的 checkpoint。确认文件存在，脚本启动时会自动校验。

### 3. CONTEXT_LENGTH（第 71-75 行）

```bash
# 注释掉不需要的那行，只保留一个：
# CONTEXT_LENGTH=1024
CONTEXT_LENGTH=2048
DEVICE_BATCH_SIZE=1   # ctx2048 时 batch=1 防 OOM
```

必须与 base_train 阶段的上下文长度一致。ctx2048 对应 `DEVICE_BATCH_SIZE=1`。

---

## 可选调整的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PEAK_LR` | 0.00005 | 主学习率，base_train 的 1/5 |
| `SFT_MUON_LR` | 0.002 | Muon 优化器 LR，base_train 的 1/10 |
| `SFT_EMBEDDING_LR` | 0.03 | Embedding LR，base_train 的 1/10 |
| `SFT_VOCAB_TO_EMBED_LR` | 0.001 | Vocab→Embed LR，base_train 的 1/10 |
| `SFT_SCALAR_LR` | 0.004 | Scalar LR，base_train 的 1/10 |
| `MAX_STEPS` | 3000 | 总训练步数 |
| `GRAD_ACCUM` | 32 | 梯度累积步数，有效 batch = GPU数 × DEVICE_BS × GRAD_ACCUM |
| `VAL_CHECK_INTERVAL` | 500 | 每 N 步做一次验证 |
| `WANDB_API_KEY` | 需填写 | 如不用 WandB 可保持 `WANDB_MODE="offline"` |

---

## 执行步骤

```bash
# 1. 进入 repo 根目录
cd /mnt/shared-storage-user/puyuan/code/OpenEBM

# 2. 激活 conda 环境
conda activate nanochat

# 3. 启动 SFT 训练
bash openebm/elm/runs/run_ebt_sft.sh
```

脚本会先打印配置摘要并等待确认（按 Enter 继续，Ctrl+C 取消）。

---

## 输出目录结构

```
logs/ebt_runs/<EXP_ID>/sft_train/
├── config/
│   ├── run_script.sh        # 本次运行的脚本快照
│   ├── hparams.json         # 超参数记录
│   ├── base_ckpt_ref.json   # base checkpoint 来源信息
│   └── git_info.txt         # git commit 信息
├── checkpoints/             # SFT checkpoint 输出
├── logs/
│   └── train.log            # 训练日志
└── status.json              # 训练完成状态
```

---

## 注意事项

1. **学习率是绝对值**：`SFT_MUON_LR` / `SFT_EMBEDDING_LR` 等不受 `PEAK_LR` 缩放。如果 base_train 用了不同的 LR，SFT 需要同比例调整，否则容易 loss NaN
2. **EBT 超参数不要改**：第 56-63 行的 MCMC 相关参数必须与 base_train 一致（`MCMC_STEP_SIZE=500.0`、`MCMC_NUM_STEPS=2`、`EBT_TYPE="time_embed"` 等）
3. **GPU 数量自动检测**：默认用所有可用 GPU。如需指定：`NUM_GPUS=4 bash openebm/elm/runs/run_ebt_sft.sh`
4. **ctx2048 显存紧张**：`DEVICE_BATCH_SIZE` 必须为 1，通过 `GRAD_ACCUM=32` 补偿有效 batch size
5. **lr schedule**：使用 `linear_warmdown`，warmup 占 5%，warmdown 占 20%，最终 LR 降到 0。`--warm_up_steps` 等旧参数在此模式下不生效
