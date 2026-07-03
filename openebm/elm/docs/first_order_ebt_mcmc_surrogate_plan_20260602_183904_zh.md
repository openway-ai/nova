# OpenEBM EBT 一阶 MCMC Surrogate 适配方案

## 背景

当前 OpenEBM 的 EBT 训练在 `openebm/elm/modeling_ebt.py` 中通过 MCMC refinement 更新预测 token/logit，然后对 refinement 终点计算 CE。训练态下最后一步 MCMC 使用：

```python
torch.autograd.grad(energy.sum(), predicted_tokens, create_graph=True)
```

后续 CE backward 会穿过 `grad_y E_theta` 回传到模型参数，因此包含 `d/dtheta(dE/dy)` 的二阶路径。该路径在 DDP/ZeRO-1/ZeRO-2 参数复制场景下可运行，但与 FSDP2/ZeRO-3 的参数 sharding 生命周期冲突，并且会显著放大 MCMC 多步 activation 峰值。

## 关键判断

不能把可训练的一阶 EBT 简化成“把 `create_graph=True` 改成 `False`”。原因是当前 CE loss 直接作用在 MCMC 后的 `predicted_tokens` 上；如果 sampler 的 `grad_y E_theta` 不保留图，CE 到 transformer/embedding 参数的梯度路径会被切断，最多只能训练 `alpha` 一类显式参与 `y_next = y - alpha * grad` 的标量。

因此，一阶训练需要改成经典 EBM/CD 风格：

1. MCMC sampler 只负责产生 detached negative sample。
2. 对 positive target 和 detached negative sample 重新计算能量。
3. 用普通一阶 contrastive/ranking energy loss 训练 `E_theta`。

这条路径没有 `d/dtheta(dE/dy)`，因此可以与 FSDP2/ZeRO-3 参数 sharding 组合。

## 当前实现

新增 CLI：

- `--mcmc_gradient_mode {second_order,first_order_debug,first_order_cd}`
- `--first_order_cd_loss_coeff`
- `--first_order_cd_loss_type {ce,margin}`
- `--first_order_cd_margin`
- `--first_order_cd_alpha_ce_coeff`

模式语义：

- `second_order`：默认值，完全保持当前 EBT 二阶 CE 目标。
- `first_order_debug`：MCMC `autograd.grad(create_graph=False)`，仅用于确认 FSDP/ZeRO-3 问题来自二阶路径，不是完整训练目标。
- `first_order_cd`：训练用一阶 surrogate。MCMC 采样不建二阶图，最终 negative logits detach 后重新计算 fake energy；真实 next token 构造 positive energy；主损失为正负能量 CE 或 margin ranking。

代码路径：

- `openebm/elm/train.py`：新增 CLI。
- `openebm/elm/modeling_ebt.py`：
  - `_mcmc_step_excluded()` 根据 `mcmc_gradient_mode` 决定 `create_graph`。
  - `forward_loss_wrapper()` 在 `first_order_cd` 训练态下使用 contrastive energy surrogate 作为主 loss。
  - `calculate_contrastive_loss()` 支持 detached fake token 重新计算能量，并修复 `-1` target mask。
- `openebm/elm/train_engines/fsdp2.py`：
  - 对 `second_order` 打印高风险 warning。
  - 对 `first_order_debug` 明确标记为诊断用途。
- `openebm/elm/train_engines/deepspeed_zero.py`：
  - ZeRO-1/2 仍允许当前二阶目标。
  - ZeRO-3 仅在 `--mcmc_gradient_mode first_order_cd` 下允许进入实验路径。
- `openebm/elm/runs/run_ebt_muon_adamw_c1024.sh` 与 `c2048.sh`：
  - 新增环境变量入口，默认不传，保持旧行为。

## 启用方式

FSDP2 一阶 surrogate：

```bash
MCMC_GRADIENT_MODE=first_order_cd TRAIN_ENGINE=fsdp2 \
bash openebm/elm/runs/run_ebt_muon_adamw_c1024.sh
```

ZeRO-3 一阶 surrogate：

```bash
MCMC_GRADIENT_MODE=first_order_cd TRAIN_ENGINE=zero-3 \
bash openebm/elm/runs/run_ebt_muon_adamw_c1024.sh
```

可选参数：

```bash
FIRST_ORDER_CD_LOSS_TYPE=margin
FIRST_ORDER_CD_MARGIN=1.0
FIRST_ORDER_CD_LOSS_COEFF=1.0
FIRST_ORDER_CD_ALPHA_CE_COEFF=0.0
```

## 与二阶目标的差异

二阶目标训练的是“沿当前能量梯度走一步/多步后 CE 下降”的可微优化过程。一阶 CD surrogate 训练的是“真实 target 能量低、MCMC negative 能量高”的能量排序。它避免了二阶图，但引入目标偏差。

该偏差是扩展到大模型的工程折中：一阶路径牺牲部分精确 unrolled optimization 梯度，换取参数 sharding、较低 activation retention、以及更稳定的多卡训练。

## 验证计划

1. 默认回归：不传 `MCMC_GRADIENT_MODE`，确认 `second_order` 行为和历史训练一致。
2. 单卡 smoke：`--mcmc_gradient_mode first_order_cd --max_steps 2`，确认 loss 有梯度、无 NaN。
3. 2/8 卡 FSDP2：`TRAIN_ENGINE=fsdp2 MCMC_GRADIENT_MODE=first_order_cd`，确认 transformer block wrap 后可 backward。
4. ZeRO-3 smoke：`TRAIN_ENGINE=zero-3 MCMC_GRADIENT_MODE=first_order_cd`，确认 DeepSpeed stage 3 初始化和训练 step。
5. 梯度检查：确认 transformer/embedding/vocab_to_embed 参数在 `first_order_cd` 下有非零梯度。
6. 显存对比：同 batch/context，比较 `second_order`、`first_order_debug`、`first_order_cd` 的 peak memory。
7. 质量对比：tiny model 上比较 MCMC refinement 后 CE 是否下降、`E_pos - E_neg` gap 是否扩大、增加 inference MCMC steps 是否改善。

## 限制

- `first_order_cd` 不是原始 EBT 二阶 CE 目标，训练曲线和最终质量需要重新标定。
- 默认不训练 `alpha`；如需让 detached sampler 的 CE 调 alpha，可设置 `--first_order_cd_alpha_ce_coeff > 0`。
- FSDP2/ZeRO-3 下仍不建议打开 `torch.compile` 和 activation checkpointing 的复杂组合，需逐步验证。
- 4B/8B 仍可能受单步 attention activation、context length、checkpoint 策略约束；一阶 surrogate 只是解除二阶 MCMC 与参数 sharding 冲突的必要步骤。
