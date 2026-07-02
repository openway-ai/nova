# Sudoku/GSM8K RL rjob 迭代记录

## 元信息

- 分支：`dev-openebm-sudoku-rl-fsdp2-merge`
- 记录开始时间：`2026-07-02 19:11 +0800`
- 目标：修复当前 Sudoku/GSM8K RL rjob 异常，关闭旧任务，重启训练，并持续监控 reward、loss、KL/energy、reward_std、degenerate/skip rate、NaN/Inf 与 rollout 质量。

## 第 1 轮：修复当前失败与重启前准备

### 时间

`2026-07-02 19:11 +0800`

### 输入任务与旧 rjob

| 任务 | rjob | 状态 | 关键现象 |
| --- | --- | --- | --- |
| Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-5f2bd` | rjob 显示 `Succeeded`，训练实际失败 | `rollout.py` 中 `logits[:, -1]` 报 `TypeError: 'NoneType' object is not subscriptable` |
| GSM8K RL | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-18-2d801` | `Running` | step 0 reward `0.100±0.000`，parse_rate 1.0，exact_match 0.0，被 `degenerate_group_rate`、`low_reward_std` 跳过 |

### 根因判断

1. Sudoku 失败不是 reward shaping 或 KL 参数问题，而是解码路径缺失 TF-head logits。
   - free-embedding MCMC checkpoint 使用 `use_tf_head`。
   - `EBT_NLP.forward(..., return_raw_logits=True)` 在 free-embedding 模式下返回的 `predicted_distributions[-1]` 为 `None`。
   - SFT loss 路径会额外用 `return_pred_hiddens=True` 并调用 `model.tf_head(...)` 生成 vocab logits，但 rollout 解码路径 `call_model_forward_decode()` 没有复用这段逻辑。
2. rjob 成功状态不可靠。
   - `run_ebt_sudoku_rl.sh` / `run_ebt_gsm8k_rl.sh` 把训练块接到 `awk | tee` 管道；未读取 `PIPESTATUS[0]`，导致左侧 `torchrun` 失败时脚本最终可能返回 `tee` 的 0。
   - rjob wrapper 之前也没有在失败后保留真实训练退出码。
3. GSM8K 当前 checkpoint 加载和 rollout 文本基本正常；现阶段问题是 reward 方差低、skip 率高，需在新版本中继续监控是否长期无有效更新。

### 修复动作

| 文件 | 改动 |
| --- | --- |
| `openebm/elm/generate.py` | 为 `use_tf_head` 的 EBT 解码/PPL 路径调用 `return_pred_hiddens=True`，再用 `model.tf_head(pred_hidden, prev_embed)` 生成 vocab logits；非 TF-head 路径如果仍得到 `None` logits 会直接报更明确的 `RuntimeError`。 |
| `openebm/elm/rl/logprobs.py` | token-logprob 兜底路径支持 TF-head logits，避免同类 checkpoint 在 `token_logprobs` loss 下再次遇到 `None` logits。 |
| `openebm/elm/runs/run_ebt_sudoku_rl.sh` | 训练管道后读取 `PIPESTATUS[0]`，分析报告生成后用真实训练退出码退出。 |
| `openebm/elm/runs/run_ebt_gsm8k_rl.sh` | 同上。 |
| `openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh` | 捕获训练脚本退出码，仍生成 metadata/report，最终按真实退出码退出。 |
| `openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh` | 同上。 |

### 重启前待执行

1. 静态校验：`py_compile` 与 `bash -n`。已完成。
2. 停止旧 GSM8K rjob：`d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-18-2d801`。已确认 `Stopped`。
3. 提交新的 Sudoku/GSM8K rjob。已提交：

| 任务 | exp_id | rjob metadata | 初始状态 |
| --- | --- | --- | --- |
| Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-191349` | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-ba472` | `Inqueue/STARTING` |
| GSM8K RL | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-191400` | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-546f5` | `Inqueue/STARTING` |

4. 每轮监控追加记录：时间戳、rjob 状态、最新 step、reward/loss/KL/energy/skip/degenerate/NaN 指标、趋势判断与下一步动作。

## 监控记录

### 2026-07-02 19:14 +0800

| 任务 | rjob 状态 | 日志/指标状态 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Inqueue/STARTING` | 产物目录尚未生成 | 等待资源分配与容器启动。 |
| GSM8K RL | `Inqueue/STARTING` | 产物目录尚未生成 | 等待资源分配与容器启动。 |

调度事件：两个任务均显示 `0/2060 nodes are unavailable: 2053 task node selector does not match node labels, 7 Insufficient cpu`。当前不是训练代码异常。

下一轮：约 10 分钟后检查 rjob 状态、`logs/train.log`、`heartbeat.json` 和最新 `[GRPO-JSON]` 指标。

### 2026-07-02 19:21 +0800

| 任务 | rjob 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | step 0 rollout 成功；reward_mean `1.5927`，reward_std `0.4041`，unique_ratio `0.8333`，degenerate `0.0`，first sample `parse_ok=True`、clue_accuracy `1.0`、blank_accuracy_frac `0.6154`、full_solve `0.0` | TF-head logits 修复有效，已越过旧的 `NoneType` 崩溃，reward 不再全 0。 |
| GSM8K RL | `Inqueue/STARTING` | 尚未启动 | 继续等待资源。 |

异常：Sudoku step 0 被 `ddp_consensus` 跳过。rank0 本地 `skip_reasons=[]`，但 `skip_consensus=any` 表示任一 rank 触发 skip 都会全局跳过。step 0 后约 3 分钟未进入 step 1，train.log 不再更新。

根因判断：当前 skip 分支的 placeholder loss 对所有 trainable 参数构造 `(p * 0.0).sum()`，一次被跳过的 step 也会触发约 973M trainable 参数的零梯度 backward/DDP 同步。该设计保证 DDP 图完整，但对 skip 频繁的 RL 初期代价过高，表现为 step 0 后长时间停在 zero-backward/optimizer 阶段。

处理动作：

1. 停止当前 rjob：
   - Sudoku：`d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-ba472`，已确认 `Stopped`。
   - GSM8K：`d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-546f5`，已确认 `Stopped`。
2. 修复 `openebm/elm/rl/ebm_grpo_trainer.py`：
   - 新增 `_zero_placeholder_loss()`，跳过 step 时只挂一个 trainable 参数构造零 loss。
   - 保留 `find_unused_parameters=True` 的 DDP 行为，让未用参数由 DDP unused-parameter 路径处理。
   - 为 skip consensus 增加跨 rank reason 统计：`global_skip_reasons` 和 `global_skip_rank_count`，避免后续只看到模糊的 `ddp_consensus`。

下一轮：提交修复后重新提交 Sudoku/GSM8K rjob，观察 skipped step 是否能快速进入下一步，以及全局 skip 的真实原因分布。

### 2026-07-02 19:28 +0800

第二轮修复提交：`db24d33 fix(rl): make skipped ddp steps cheap`。

重新提交 rjob：

| 任务 | exp_id | rjob metadata | 初始状态 |
| --- | --- | --- | --- |
| Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-192731` | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-424ae` | `Running` |
| GSM8K RL | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-192741` | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-22e90` | `Inqueue/STARTING` |

下一轮：重点确认 Sudoku step 0 后是否能进入 step 1；若仍卡住，则说明 cheap placeholder 仍未解决 DDP skip 路径，需要进一步审查 DDP unused-parameter/optimizer hook 交互。

### 2026-07-02 19:35 +0800

第三轮结果：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | step 0 后停滞，已停止 | step 0 reward_mean `1.5927`，reward_std `0.4041`，unique_ratio `0.8333`；`global_skip_rank_count=1`，`global_skip_reasons=['degenerate_group_rate', 'low_reward_std']` | TF-head 与 reward 正常；问题集中在 hard skip 的 DDP 自动优化路径。 |
| GSM8K RL | `Inqueue/STARTING`，已停止 | 尚未启动 | 跟随 Sudoku 修复后重提。 |

根因更新：cheap placeholder 仍没有让 step 0 后进入 `optimizer_step_skipped` 或 step 1。说明当前 Lightning DDP automatic optimization 下，hard skip 分支本身不可靠；即使只挂一个参数，仍可能卡在 backward/DDP reducer 路径。由于 degenerate group 的 advantage 已经在 trainer 内置零，硬跳过不是当前优化版训练的必要条件。

处理动作：

1. 停止第三轮 rjob：
   - Sudoku：`d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-424ae`。
   - GSM8K：`d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-22e90`。
2. 将优化版训练脚本的 hard skip 改为 opt-in：
   - `SKIP_DEGENERATE_THRESHOLD=1.01`
   - `MIN_REWARD_STD_TO_UPDATE=0.0`
   - `MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE=0.0`
3. 保留 skip health logging、degenerate advantage 置零、以及 `skip_consensus` 配置，后续若要重新启用 hard skip 需要先单独验证 DDP skip 路径。

下一轮：提交配置修复后重启 Sudoku/GSM8K，重点确认 Sudoku 是否能在 step 0 后进入 step 1，并观察真实 loss/grad/energy 指标。
