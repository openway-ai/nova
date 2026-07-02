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
