# dev-openebm-sudoku-rl-fsdp2-merge 当前状态

更新日期：2026-07-02

## 元信息

| 项目 | 当前状态 |
|---|---|
| 文档文件名 | 保留 `sudoku_rl_fsdp2_merge_status_20260702.md` |
| 保留原因 | 文件名仍准确对应分支、主题和更新日期；本次是同一状态文档的内容刷新，不需要另建日期或分支名。 |
| 当前分支 | `dev-openebm-sudoku-rl-fsdp2-merge` |
| 当前 HEAD | `7fb55bc fix(rjob): checkout sudoku compare branch in eval job` |
| 基分支 | `origin/dev-openebm-sudoku` |
| 已合入 PR #30 | `dev-openebm-sudoku-rl`，本地 merge commit `893fd3e` |
| 已合入 PR #33 | `dev-openebm-train-engine`，本地 merge commit `ea866e2` |
| 主要修复提交 | `217050e` review 修复；`095c687` free-embed TF-head ckpt 加载修复；`7fb55bc` compare rjob 分支 checkout 修复 |

本轮状态依据来自当前分支代码、运行脚本、rjob 状态和以下本地产物：

- `openebm/elm/train.py`
- `openebm/elm/train_engines/*`
- `openebm/elm/modeling_ebt.py`
- `openebm/elm/tf_head.py`
- `openebm/elm/rl/*`
- `openebm/elm/runs/run_ebt_sudoku_rl.sh`
- `openebm/elm/runs/run_ebt_gsm8k_rl.sh`
- `openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
- `openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh`
- `logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-170632`
- `logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-182711`
- `logs/ebt_runs/d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-182721`

---

## 1. 训练引擎（train-engine）适配情况

### 默认路径

默认训练路径仍是 `--train_engine lightning_ddp`。如果不显式传入 `--train_engine`，`prepare_train_engine()` 和 `apply_train_engine()` 都直接返回，不改变原有 Lightning/DDP 行为。

这满足合并约束：默认 DDP 路径保持不变，train-engine 是可选开关。

### 支持范围

| 引擎 | 当前代码状态 | 可用性与限制 |
|---|---|---|
| `lightning_ddp` | 默认路径 | 保持原有行为。仍是当前最稳妥的 EBT/RL 路径。 |
| `fsdp2` | 已接入 native composable FSDP2 | 默认按 transformer block 包裹；embedding、`vocab_to_embed`、`alpha`、replay buffer、transformer root 剩余参数保持复制。代码会默认关闭 `torch.compile`、禁用 activation checkpointing、强制保守 `reshard_after_forward=false`。`second_order` 会告警，推荐配合 first-order surrogate。 |
| `zero-1` / `zero-2` | 已通过 Lightning `DeepSpeedStrategy` 接入 | 参数仍复制，因此代码允许 exact second-order EBT；但 DeepSpeed optimizer flatten/partition 与 Muon 矩阵更新不兼容，Muon 会回退到 layered AdamW。 |
| `zero-3` | 已接入但 gated | 只允许 `first_order_cd`、`first_order_nce`、`proposal_aware_nce`。exact second-order 被显式拒绝，因为 ZeRO-3 参数分片与 MCMC 二阶 autograd 路径不兼容。 |
| `megatron` | 保留枚举 | 解析存在，但未实现，会抛 `NotImplementedError`。 |

### 已知限制与验证状态

- FSDP2 与 ZeRO-3 的主要目标是降低分片训练下的二阶 autograd 风险，但它们还不是生产级默认路径。
- 当前分支代码层面已有 guardrail；本轮没有看到融合分支上完成的 FSDP2/ZeRO-3 新训练验证。
- 本地历史目录中有 ZeRO-2、ZeRO-3、FSDP2 相关产物，但 FSDP2 `d26-c2048-fsdp2-paNCE...` 日志为 `DRY_RUN`，不能作为真实训练收敛证据。
- ZeRO-2 历史日志显示能跑到较后 step；ZeRO-3 first-order 历史日志存在，但质量指标和与二阶目标的一致性仍需单独验证。

结论：train-engine 适配已经具备可实验入口；默认 DDP 未被破坏；FSDP2/ZeRO-3 仍应按实验功能使用。

---

## 2. 一阶近似（first-order approximation）的原因与实现情况

### 引入原因

原始 EBT objective 在 MCMC 内部使用：

```text
torch.autograd.grad(..., create_graph=True)
```

这会从最终 CE loss 反传穿过 `grad_y E_theta`，形成二阶 autograd 路径。在 FSDP2 或 ZeRO-3 这类参数分片训练中，采样器保留的图会遇到参数 reshard/partition、显存占用和 double-backward 稳定性问题。

因此分支引入 first-order MCMC 近似：保留 MCMC 产生负样本或 proposal 的能力，但尽量避免在分片参数上保留二阶 autograd 图。

### 已实现模式

| 模式 | 当前含义 |
|---|---|
| `second_order` | 默认 exact EBT 目标。DDP 路径保留；FSDP2/ZeRO-3 下风险高。 |
| `first_order_debug` | 诊断模式，只把 MCMC 的 `create_graph` 关掉；不是完整训练目标。 |
| `first_order_cd` | 使用 detached MCMC negative sample，并重算正负样本 energy；支持 CE/margin 形式和可选 trajectory-local CD 辅助项。 |
| `first_order_nce` | graph-safe NCE 风格 surrogate。 |
| `proposal_aware_nce` | proposal-aware NCE；支持 `uniform` / `mcmc_final` proposal、base coeff、ranking auxiliary、relaxed-CD auxiliary 等配置。 |

### 验证状态

- 一阶近似不是二阶目标的数学等价替代。
- 当前代码已经实现多种 first-order surrogate，但尚未完成充分实验验证。
- 必须明确：一阶近似性能是否与二阶一致，目前尚未确认。
- 对 FSDP2/ZeRO-3 来说，first-order 是当前可继续探索的方向；它还不能作为已验证的最终训练方案。

待确认项：

- `first_order_cd` / `first_order_nce` / `proposal_aware_nce` 与 `second_order` 在同数据、同模型规模上的质量差距。
- FSDP2 + first-order 在长跑中的稳定性、吞吐和 checkpoint 可恢复性。
- ZeRO-3 + first-order 是否能提供可接受的质量/显存折中。

---

## 3. RL 实现情况

### 已落地范围

当前分支包含 Sudoku/GSM8K RL 初始管线：

- 训练入口：`openebm/elm/rl/train_rl_sudoku.py`、`openebm/elm/rl/train_rl_gsm8k.py`
- 共享 trainer/config：`openebm/elm/rl/ebm_grpo_trainer.py`、`openebm/elm/rl/ebm_grpo_config.py`
- rollout：`openebm/elm/rl/rollout.py`
- reward：`openebm/elm/rl/rewards.py`、`openebm/elm/rl/gsm8k_rewards.py`
- energy/logprob 工具：`openebm/elm/rl/logprobs.py`
- 数据集：`openebm/elm/rl/sudoku_dataset_rl.py`、`openebm/elm/rl/gsm8k_dataset_rl.py`
- 运行脚本：`openebm/elm/runs/run_ebt_sudoku_rl.sh`、`openebm/elm/runs/run_ebt_gsm8k_rl.sh`
- rjob wrapper：`openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`、`openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh`
- 分析脚本：`openebm/elm/scripts/analyze_rl_run.py`

### Rollout / Reward / Loss

| 部分 | 当前状态 |
|---|---|
| rollout | 已实现 autoregressive generation、多 completion/group、stop token 处理、completion mask、trajectory dump。 |
| Sudoku reward | 已实现 format、clue preservation、blank accuracy、constraint validity、full solve 等组件。 |
| GSM8K reward | 已实现 answer extraction、partial credit、exact match、format、length penalty 等组件。 |
| loss | 支持 `energy_gspo`、`energy_reinforce`、`token_logprobs`。当前 rjob 默认使用 `energy_gspo`。 |
| KL / anchor | `energy_gspo` / `energy_reinforce` 使用 sequence-energy anchor，默认 `symmetric_huber`。 |
| token logprob 路径 | 保留为研究/调试路径；需要更高阶图，当前不是稳定主路径。 |

### SFT -> RL checkpoint 加载

review comment 相关修复已经落地：

- 支持 `model.*` 前缀剥离。
- 支持 `_orig_mod.*` 与 `transformer._orig_mod.* -> transformer.*` 重映射。
- 支持 `transformer_eager.* -> transformer.*` 兼容。
- 对 `transformer.*`、`tf_head.*`、`vocab_to_embed.*`、`embeddings.*`、`alpha` 等关键 unexpected key fail-fast，避免静默丢权重。
- 对 missing key fail-fast，避免随机初始化部分模型后继续 RL。
- 已加入 free-embed TF-head 支持：`openebm/elm/tf_head.py` 和 `modeling_ebt.py` 中的 `use_tf_head` / `free_embedding_mcmc` 逻辑。

实际日志确认：

- 旧 Sudoku RL run `20260702-170632` 忽略了 `tf_head.proj.weight`，加载参数为 `973,643,010`，rollout 产生非 Sudoku 文本并 reward 全 0。
- 修复后 Sudoku RL run `20260702-182711` 已重映射 576 个 compiled/eager transformer keys，并加载参数 `1,028,168,962`，说明 `tf_head` 权重已进入模型。

### 稳定性防护与 review comment 修复

| review / 稳定性项 | 当前处理 |
|---|---|
| skip-degen zero-fill 可能移动 Muon 参数 | 新增 `optimizer_utils.py`，Muon group 缺 grad 时跳过整组，不再 zero-fill；测试在 `test_rl_review_fixes.py`。 |
| SFT ckpt `_orig_mod` key 重映射 | Sudoku/GSM8K loader 均已处理。 |
| IterableDataset rank/worker 重复采样 | 新增 `data_sharding.py`，全局 shuffle 后按 rank/worker stride 分片；测试覆盖 disjointness。 |
| DDP skip consensus | 默认改为 `any`，任一 rank 发现坏 rollout 时全局跳过。 |
| analyzer 旧指标 | `analyze_rl_run.py` 已使用 `old_policy_energy_drift`、`log_ratio_clamp_rate`，不再依赖已移除的 `energy_ppo_kl` / 旧 `clamp_rate` 命名。 |

### 当前新阻塞点

free-embed TF-head checkpoint 的权重加载已修复，但 Sudoku RL rollout 仍失败在生成路径：

```text
TypeError: 'NoneType' object is not subscriptable
openebm/elm/rl/rollout.py: logits[:, -1]
```

从代码看，`generate_completions()` 通过 `call_model_forward_decode(... return_raw_logits=True)` 取 `model.forward(...)[0][-1]`。但 `free_embedding_mcmc=True` 时，`model.forward()` 当前会把 `predicted_tokens_for_loss` 设为 `None`；TF-head logits 只在 `forward_loss_wrapper()` 中通过 `return_pred_hiddens=True` 后再调用 `self.tf_head(...)` 生成。也就是说，训练 loss 路径知道如何使用 TF-head，但 rollout 生成路径还没有接上 TF-head。

结论：当前 Sudoku RL 的下一步不是调参，而是修复 free-embed + TF-head 的 autoregressive decode 路径。

---

## 4. 目前的实验结果情况

### 当前 rjob 状态

截至本次检查（2026-07-02 19:04 左右，Asia/Hong_Kong）：

| 任务 | rjob metadata | 状态 | 备注 |
|---|---|---|---|
| 旧 Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-9c79e` | Stopped | `tf_head.proj.weight` 被忽略，reward 全 0。 |
| 修复后 Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-5f2bd` | rjob 显示 Succeeded | 训练日志和 analyzer 显示实际失败在 rollout `logits=None`。 |
| 修复后 GSM8K RL | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-18-2d801` | Running | 已落盘 step 0 指标；heartbeat 显示已进入后续 step。 |
| Sudoku compare fix2 | `sudoku-compare-freeembed-eval-fix2-54663787` | Inqueue / STARTING | 已修复脚本缺失问题；仍在等资源，暂无评测产物。 |

### Sudoku RL：旧 run `20260702-170632`

目录：

```text
logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-170632
```

关键事实：

- SFT checkpoint：`d26-ctx2048-sudoku-sft-freeembed-direct-1node-8gpu-20260625-190601/...valid_loss=0.4366.ckpt`
- 日志显示 `tf_head.proj.weight` 是 unexpected key，被忽略。
- 加载模型参数：`973,643,010`。
- step 0、5、10、15、20、25、30、35 均为 `reward=0.000±0.000`。
- 每步都因 `degenerate_group_rate` 和 `low_reward_std` 被 skip。
- first completion 是非 Sudoku 文本，`parse_ok=False`、`digit_count=0`。

结论：该 run 不能说明 RL 算法收敛性，只说明 checkpoint 加载遗漏导致 rollout 无效。这个问题已由 `095c687` 部分修复。

### Sudoku RL：修复后 run `20260702-182711`

目录：

```text
logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-182711
```

关键事实：

- rjob metadata 显示 Succeeded，但 `analysis_report.md` 标记 Status 为 `failed`。
- checkpoint 已正确加载 TF-head：
  - 重映射 576 个 compiled/eager transformer keys。
  - 参数数变为 `1,028,168,962`。
- 训练在 step 0 rollout 阶段失败：
  - `openebm/elm/rl/rollout.py` 中访问 `logits[:, -1]`。
  - `logits` 实际为 `None`。
- `analysis_metrics.csv` 只有表头，没有有效 step 指标。

结论：Sudoku RL 当前仍未收敛；更准确地说，修复 ckpt 加载后还没有跑到可评价的 RL 更新阶段。当前阻塞是 free-embed TF-head decode 路径未接通。

### GSM8K RL：run `20260702-182721`

目录：

```text
logs/ebt_runs/d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-182721
```

当前已落盘事实：

- rjob 正在 Running。
- checkpoint：`d26-ctx2048-sudoku-mixed-0.6-20260508/...valid_loss=0.4782.ckpt`
- 模型加载参数：`613,080,834`。
- 数据：
  - train: GSM8K HF arrow，`7473` examples。
  - val/test: GSM8K HF arrow，`1319` examples。
- step 0 rank-0 聚合日志：
  - `reward=0.100±0.000`
  - `parse_rate=1.0`
  - `answer_acc=0.0`
  - `exact_match=0.0`
  - 因 `degenerate_group_rate` 与 `low_reward_std` 被 skip。
- heartbeat 最近显示已进入 `step=8`，但磁盘 `train.log` 当前只落盘到 step 0 附近，后续趋势待确认。

结论：GSM8K 管线已经进入 rollout/reward/skip 流程，不存在 Sudoku 当前的 `logits=None` 崩溃；但 reward 方差仍为 0，首步未更新，收敛趋势待 run 完成和分析报告生成后判断。

### 总体结论

- train-engine 融合已完成代码层适配，但 FSDP2/ZeRO-3 仍是实验路径。
- 一阶近似已实现多种 surrogate，但尚未证明质量等同于二阶目标。
- RL 管线的 review comment 修复已落地，尤其是 ckpt key remap、数据分片、skip consensus 和 Muon skip 逻辑。
- Sudoku RL 当前尚未收敛；最新真实状态是“checkpoint 加载修复后，rollout decode 仍需修复”。
- GSM8K RL 正在运行，首步显示低 reward 且被 skip，最终趋势待确认。

### 后续建议

1. 优先修复 `call_model_forward_decode()` / `generate_completions()` 对 `use_tf_head + free_embedding_mcmc` 的支持：decode 时应拿到 TF-head vocab logits，而不是 `None`。
2. 修复 rjob wrapper 的退出码传播：当前 Sudoku run 训练失败但 rjob metadata 显示 Succeeded，容易误导状态判断。
3. 修复后先用小样本/短步数 smoke run 验证 Sudoku：至少确认 completion 能 parse、reward 非全 0、`analysis_metrics.csv` 有有效 step。
4. GSM8K run 完成后生成 `analysis_report.md`，再判断 reward、skip、energy drift 和是否需要调 reward shaping / sampling。
5. train-engine 侧继续做独立验证矩阵：DDP second-order baseline、FSDP2 first-order、ZeRO-2 second-order、ZeRO-3 first-order，并明确每组的质量和稳定性差异。

待确认项汇总：

- 一阶近似性能是否能接近二阶目标：待确认。
- FSDP2/ZeRO-3 在融合分支上的长跑稳定性：待确认。
- 修复 free-embed TF-head decode 后 Sudoku RL 是否能产生有效 reward：待确认。
- GSM8K 当前 Running run 的最终指标和收敛趋势：待确认。
