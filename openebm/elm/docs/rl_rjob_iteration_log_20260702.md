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

### 2026-07-02 19:37 +0800

第三轮配置修复提交：`1e378b8 fix(rl): avoid hard skip stalls in optimized rjobs`。

重新提交 rjob：

| 任务 | exp_id | rjob metadata | 初始状态 |
| --- | --- | --- | --- |
| Sudoku RL | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-193700` | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-d8096` | `Starting` |
| GSM8K RL | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-193712` | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-df33f` | `Inqueue/STARTING` |

下一轮：等待 Sudoku 第一个 `[GRPO-JSON]`，确认 hard skip disabled 后 `skipped_step=0.0`，并观察是否出现 `loss_ready`/`backward_done` 或进入下一 step。

### 2026-07-02 19:44 +0800

第四轮初始监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，replica `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-d8096-cfcd2` | step 0: reward_mean `1.5927`，reward_std `0.4041`，range `[0.966, 1.921]`，unique_ratio `0.8333`，degenerate `0.0`，entropy_mean `0.0404`，`skipped_step=0.0`，`global_skip=False`；样例 `parse_ok=True`、clue_accuracy `1.0`、blank_accuracy_frac `0.6154`、full_solve `0.0` | hard skip disabled 后，step 0 已完成 `skip_consensus_done -> loss_ready -> backward`，heartbeat 推进到 step 1 `generate_start`。上一轮的 DDP skip 停滞已解除。 |
| GSM8K RL | `Inqueue/STARTING`，replica `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-df33f-cfcd2` | 尚未产生日志 | 继续等待资源。 |

补充判断：当前没有复现 `NoneType` logits/TF-head 加载问题；Sudoku rollout 已能解析并获得非零 reward。当前 reward 仍未收敛，`full_solve=0.0`，需要继续观察多步趋势、loss/grad_norm、KL/energy drift 和 rollout 是否退化。

下一轮：等待 step 1 的 `[GRPO-JSON]` 与 optimizer/step 进展；如 reward 长期不涨或出现 NaN/Inf、reward_std 归零、unique_ratio 崩塌，再进入下一轮修复。

### 2026-07-02 19:56 +0800

第四轮第二次监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 从 step 1 推进到 step 5 `generate_start`；rjob 管理侧仍为 `RUNNING` | 训练活性正常，已不再是 step 0 DDP 停滞。 |
| GSM8K RL | `Inqueue/STARTING` | 尚未产生日志 | 调度等待，非训练异常。 |

观测限制：当前 `log_interval=10`，`[GRPO]`/`[GRPO-JSON]` 只会在 step 0、10、20... 输出；trajectory 也按 `traj_log_interval=50` 写入。因此只有 step 0 reward 快照不是异常。下一轮应等待 step 10 左右的第二个 `[GRPO-JSON]`，再判断 reward 趋势。

下一轮：重点读取 step 10 的 reward/loss/energy/grad 指标；若 heartbeat 停止更新或出现异常栈，再区分是 rollout 卡住、DDP 同步卡住还是运行时错误。

### 2026-07-02 20:07 +0800

第四轮第三次监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 更新到 step 8 `generate_start`，最近更新时间 `2026-07-02 20:05:26`；train.log 仍只有 step 0 的 `[GRPO-JSON]` | 训练仍有活性，但尚未到 `log_interval=10`，因此还没有第二个 reward 快照。 |
| GSM8K RL | `Inqueue/STARTING` | 尚未产生日志 | 继续等待调度。 |

当前判断：无 `Traceback`、NaN/Inf 或 rjob 失败信号。Sudoku 每步生成耗时约 2 到 3 分钟，step 10 预计在下一轮短间隔内出现。

下一轮：短间隔检查 step 10 `[GRPO-JSON]`；拿到第二个快照后比较 step 0 -> step 10 的 reward_mean/reward_std/unique_ratio/energy drift。

### 2026-07-02 20:14 +0800

第四轮第四次监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 10 `generate_start`；已刷出 step 5 指标：reward_mean `1.1256`，reward_std `0.5061`，range `[0.483, 2.202]`，unique_ratio `0.9167`，degenerate `0.0`，entropy_mean `0.0312`，ref_energy_kl `0.000141`，grad_norm `15.456`，nan_grad_params `0` | reward 仍为非零且多样性正常；step 5 低于 step 0，但不同 prompt 间波动较大，暂不判定崩塌。 |
| GSM8K RL | `Running` | 已启动到 step 0 `training_step_start`；train/test HF arrow 加载成功；未见 `NoneType`、missing/unexpected key 或 checkpoint remap 报错 | GSM8K 不存在与 Sudoku compare 相同的 TF-head logits 崩溃；当前等待首个 reward 快照。 |

checkpoint 判断：

- Sudoku 使用目标 freeembed SFT ckpt：`s=step=2999...valid_loss=0.4366.ckpt`。
- GSM8K 使用现有脚本默认 `sudoku-mixed` SFT ckpt：`s=step=2990...valid_loss=0.4782.ckpt`。rjob 脚本中已有注释说明当前分支没有配置专用 GSM8K SFT ckpt，因此沿用该默认；是否应换专用 GSM8K SFT ckpt待确认。

下一轮：读取 Sudoku step 10 或 step 15 指标，以及 GSM8K step 0 reward；重点判断 reward 是否归零、reward_std 是否归零、unique_ratio 是否塌缩、是否出现 NaN/Inf。

### 2026-07-02 20:26 +0800

第四轮第五次监控与 GSM8K 修复：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 14 `generate_start`；最近已知 step 5 reward_mean `1.1256`，reward_std `0.5061`，unique_ratio `0.9167`，nan_grad_params `0` | 继续运行观察；当前不是 reward 全 0 或 DDP 停滞。 |
| GSM8K RL | 已停止旧 rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-df33f` | step 0: reward_mean `0.1`，reward_std `0.0`，advantage_var `0.0`，degenerate_group_rate `1.0`，loss `0.0`，unique_ratio `1.0`，parse_rate `1.0`，answer_acc `0.0` | 训练没有崩溃，但首批 rollout 没有有效学习信号；应修复 reward shaping 后重提。 |

GSM8K 根因：checkpoint 加载没有 `NoneType`、missing/unexpected key 或 TF-head remap 问题；真正的问题是奖励函数对“能解析出任意数字但答案错误”的样本统一给固定 `partial_credit=0.1`。当一个 prompt 的多条 completion 都错误但可解析时，组内 reward 全相同，GRPO advantage 被置零，loss 变成 0。

修复动作：

1. `openebm/elm/rl/gsm8k_rewards.py`
   - 将固定 parsed-answer 奖励从 `0.1` 降为 `0.05`。
   - 新增 `answer_proximity_score` 和 `answer_proximity` 奖励，按对称相对误差给错误数值答案一个 `0.0` 到 `0.25` 的连续 shaping 信号。
   - 将 exact-match 权重调整为 `0.75`，保持正确答案仍为主奖励，带 `####` 格式时总分上限仍为 `1.0`。
2. `openebm/elm/rl/train_rl_gsm8k.py`
   - 将 `answer_proximity` 纳入日志组件。
   - 补齐 `reward_min`、`reward_max`、`reward_zero_frac`、`response_len_mean/max`，避免监控报告里出现 `null`。

验证：

- `python -m py_compile openebm/elm/rl/gsm8k_rewards.py openebm/elm/rl/train_rl_gsm8k.py`
- 小样例检查：正确答案仍高分，错误但接近的答案获得比远错答案更高的 shaping 分。

下一轮：提交修复后重新提交 GSM8K rjob；Sudoku 保持当前 rjob 运行并继续监控。

### 2026-07-02 20:29 +0800

GSM8K 修复提交与重启：

| 项目 | 值 |
| --- | --- |
| 修复提交 | `2a0970e fix(rl): shape gsm8k numeric rewards` |
| 旧 GSM8K rjob | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-19-df33f`，已确认 `Stopped` |
| 新 GSM8K exp_id | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-202911` |
| 新 GSM8K rjob metadata | `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` |
| 预期报告路径 | `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-202911/sft_train/analysis_report.md` |

下一轮：确认新 GSM8K 是否启动，并检查 step 0 reward_std/advantage_var 是否从 0 恢复；继续监控当前 Sudoku rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-d8096`。

### 2026-07-02 20:40 +0800

第五轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 18 `generate_start`；step 10: reward_mean `1.4768`，reward_std `0.3290`，unique_ratio `1.0`，ref_energy_kl `0.001346`，grad_norm `25.378`，nan_grad_params `0`；step 15: reward_mean `0.8458`，reward_std `0.0722`，unique_ratio `1.0`，loss `0.02159` | reward 非零、无 NaN/Inf、未卡 DDP；但 step 15 reward_std 偏低且 reward_mean 下探，需要继续观察是否只是 prompt 波动还是开始退化。 |
| GSM8K RL | `Running`，新 rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | step 0: reward_mean `0.1800`，reward_std `0.0404`，advantage_var `0.0357`，degenerate_group_rate `0.0`，unique_ratio `1.0`，parse_rate `1.0`，answer_acc `0.0`，answer_proximity `0.1300` | reward shaping 修复生效：旧 run 的 `reward_std=0`、`advantage_var=0`、`degenerate_group_rate=1.0` 已恢复为非零学习信号。 |

当前结论：

- GSM8K 的异常已从“无学习信号”修到“有 shaped reward 方差”；下一步需要看 answer_acc/format 是否随训练改善。
- Sudoku 仍未收敛，`full_solve=0.0`；目前没有加载或 DDP 崩溃问题，下一步重点观察 reward_std 是否继续收缩、reward_mean 是否持续下行。

下一轮：继续默认 10 分钟监控；若 Sudoku 连续多个 logging step reward_std 接近 0 或 reward_mean 持续低位，则优先考虑提高采样温度/多样性、调弱过强更新或增加 reward/format 稳定项。

### 2026-07-02 20:52 +0800

第六轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 21 `generate_start`；step 20: reward_mean `1.5496`，reward_std `0.4808`，range `[0.923, 2.134]`，unique_ratio `0.9167`，loss `0.00292`，nan_grad_params 上一记录为 `0` | step 20 从 step 15 的低点恢复，暂无持续塌缩证据；继续观察。 |
| GSM8K RL | `Running` | heartbeat 到 step 10 `training_step_start`；step 5: reward_mean `0.1378`，reward_std `0.0417`，range `[0.086, 0.193]`，answer_proximity `0.088`，parse_rate `1.0`，answer_acc `0.0`，degenerate `0.0`，grad_norm step 0 为 `0.0120` | shaped reward 继续提供非零方差；GSM8K 仍没有 exact answer 改善，但已不再是全零 advantage。 |

当前结论：两个任务均运行中，没有 `Traceback`、NaN/Inf、DDP 停滞或 reward 全零。Sudoku 尚未收敛，GSM8K 仍处于 shaped reward 初期。

下一轮：继续看 Sudoku step 25/30 和 GSM8K step 10/15；若 GSM8K 长期只有 proximity 而 answer_acc/format 不动，下一步考虑 prompt 格式强化或专用 GSM8K SFT ckpt。

### 2026-07-02 21:03 +0800

第七轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 24 `skip_consensus_start`，`local_skip=0.0`、`unique_ratio=1.0`；最近完整指标仍为 step 20: reward_mean `1.5496`，reward_std `0.4808` | rjob 活跃且未触发 skip；等待 step 25/30 完整指标。 |
| GSM8K RL | `Running` | heartbeat 到 step 14 `skip_consensus_start`；step 10: reward_mean `0.1150`，reward_std `0.0855`，advantage_var `0.1599`，format `0.025`，answer_proximity `0.0400`，answer_acc `0.0`，degenerate `0.0`，grad_norm `0.0167`，nan_grad_params `0` | reward 方差继续非零，且 format 开始有少量信号；仍未出现 exact answer 提升。 |

当前结论：两个任务均正常运行。Sudoku 仍未收敛但没有持续退化证据；GSM8K 的 reward shaping 修复稳定生效。

下一轮：继续观察 Sudoku step 25/30 与 GSM8K step 15/20；如 GSM8K answer_acc 长期为 0，后续优化优先级为 prompt 格式约束、专用 GSM8K SFT ckpt、以及更强的 format reward。

### 2026-07-02 21:40 +0800

第八轮监控与 Sudoku 保守化修复：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | 旧 rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-d8096` 已停止 | step 25: reward_mean `0.0`，reward_std `0.0`，zero_frac `1.0`，degenerate_group_rate `1.0`，format/clue/blank/validity 全为 `0`，ref_energy_kl `1.7803`，loss `0.8901`，grad_norm `79.066`，max_param_grad `24.696`，nan_grad_params `0` | 出现明显 collapse-like rollout 和高 KL/高梯度脉冲；不再继续使用该超参组合。 |
| GSM8K RL | `Running` | step 25: reward_mean `0.1861`，reward_std `0.2575`，answer_acc `0.125`，exact_match `0.0938`，degenerate `0.0`，nan_grad_params `0` | GSM8K reward shaping 生效并开始出现正确答案样本，继续运行。 |

Sudoku 根因判断：不是 checkpoint/TF-head 加载问题，也不是 DDP skip 卡死；模型在当前 Sudoku RL 超参下出现了采样分布漂移，表现为 completion 到达最大长度、完全不可解析、能量相对 reference 大幅偏移。`MUON_LR=2e-4`、`LEARNING_RATE=5e-7`、`BETA=0.5` 对当前 RL 阶段偏激进。

修复动作：

1. `openebm/elm/runs/run_ebt_sudoku_rl.sh`
   - `TEMPERATURE`: `0.70 -> 0.60`
   - `TOP_P`: `0.80 -> 0.75`
   - `LEARNING_RATE`: `5e-7 -> 2e-7`
   - `MUON_LR`: `2e-4 -> 5e-5`
   - `BETA`: `0.5 -> 1.0`
   - `MAX_GRAD_PER_PARAM`: `0.02 -> 0.01`
   - `TRAJ_LOG_INTERVAL`: `50 -> 25`，便于捕获退化样本。
2. `openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
   - 同步上述默认值。
   - 显式传递 `GRADIENT_CLIP_VAL`、`MAX_GRAD_PER_PARAM`、`TRAJ_LOG_INTERVAL` 到 rjob 环境与 metadata。

验证：

- `bash -n openebm/elm/runs/run_ebt_sudoku_rl.sh`
- `bash -n openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
- `git diff --check` 对上述脚本通过。

下一轮：提交修复后重提 Sudoku rjob；GSM8K 保持当前 rjob 运行。

### 2026-07-02 21:42 +0800

Sudoku 保守版重启：

| 项目 | 值 |
| --- | --- |
| 修复提交 | `4cd3a4a fix(rl): make sudoku rjob more conservative` |
| 旧 Sudoku rjob | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-d8096`，已确认 `Stopped` |
| 新 Sudoku exp_id | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-214231` |
| 新 Sudoku rjob metadata | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-0fa37` |
| 预期报告路径 | `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-214231/sft_train/analysis_report.md` |

下一轮：确认新 Sudoku rjob 是否启动，并重点检查 step 0/5 是否避免全零 reward、高 KL 和高梯度脉冲。

### 2026-07-02 21:47 +0800

第九轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，新 rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-0fa37` | heartbeat 到 step 0 `skip_consensus_start`，`local_skip=0.0`、`unique_ratio=0.75`；训练日志仍停留在初始化/数据加载 stdout，尚未刷出 `rollout_ready` 或 `GRPO-JSON` | 新任务已进入首轮 rollout/skip 判断，不是启动失败；还没有足够指标判断 reward 趋势，需要继续等待 step 0 完整日志。 |
| GSM8K RL | `Running`，rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | 最新完整指标仍为 step 25: reward_mean `0.1861`，reward_std `0.2575`，answer_acc `0.125`，exact_match `0.0938`，degenerate `0.0`，nan_grad_params `0`; heartbeat 到 step 34 `training_step_start` | 运行健康，未复现 reward_std 为 0 的退化；继续观察 answer_acc/format 是否稳定提升。 |

当前结论：GSM8K 修复有效且继续运行；Sudoku 保守版已启动并进入 step 0，但完整 reward/KL/梯度指标尚未落盘，下一轮继续等待 `loss_ready` 和 `GRPO-JSON`。

### 2026-07-02 21:50 +0800

第十轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | step 0: reward_mean `1.5721`，reward_std `0.4567`，range `[0.966, 1.921]`，advantage_var `0.9167`，zero_frac `0.0`，degenerate_group_rate `0.0`，unique_ratio `0.75`，ref_energy_kl `0.0`，completion_len_mean `166.3`；heartbeat 进入 step 1 `generate_start` | 保守化超参避免了旧 run step 25 的全零 reward、全退化 rollout、高 KL 组合；但目前只是首个 logging step，尚不能判断收敛趋势。 |
| GSM8K RL | `Running` | step 30: reward_mean `0.1081`，reward_std `0.0553`，parse_rate `0.875`，answer_acc `0.0`，degenerate `0.0`，nan_grad_params `0`; step 35 已到 `loss_ready`，reward_mean `0.1594`，reward_std `0.0590`，unique_ratio `1.0` | reward 方差仍非零，没有回到旧异常；answer_acc 仍不稳定，需要继续观察。 |

当前结论：两个任务都未出现崩溃或 DDP skip；Sudoku 已从 collapse-like 状态恢复到可训练的非零 reward 分布。下一轮重点看 Sudoku step 5/10 是否保持 reward_std 和 KL 稳定。

### 2026-07-02 22:21 +0800

第十一轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | step 5: reward_mean `0.9253`，reward_std `0.1394`，range `[0.483, 0.966]`，advantage_var `0.4451`，zero_frac `0.0`，degenerate_group_rate `0.0`，unique_ratio `0.9167`，ref_energy_kl `1.1e-6`，grad_norm `7.2531`，max_param_grad `2.1642`，nan_grad_params `0`; heartbeat 已进入 step 11 `generate_start` | 未复现旧 run 的全零 reward/high-KL/high-grad collapse；但相较 step 0，blank_accuracy 和 constraint_validity 降到 `0`，reward 主要来自 format+clue，需继续观察 step 10/15 是否恢复或持续退化。 |
| GSM8K RL | `Running` | step 40: reward_mean `0.1902`，reward_std `0.0457`，answer_proximity `0.1402`，parse_rate `1.0`，answer_acc `0.0`，degenerate `0.0`，grad_norm `0.0102`，nan_grad_params `0` | reward shaping 继续提供非零学习信号；exact answer 不稳定，暂不重启。 |

当前结论：Sudoku 保守化修复使训练摆脱全零 reward 崩塌，但仍未显示收敛趋势；当前主要风险是 reward 退回 format/clue 的浅层信号。下一轮看 step 10/15，如 `blank_accuracy=0` 和 `validity=0` 持续存在且 reward_std 收缩，再考虑加强 Sudoku reward/采样约束或进一步降低更新强度。

### 2026-07-02 22:45 +0800

第十二轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | step 10: reward_mean `1.4172`，reward_std `0.3656`，blank_accuracy `0.2169`，constraint_validity `0.2188`，unique_ratio `1.0`，ref_energy_kl `8.09e-5`，grad_norm `23.4006`，nan_grad_params `0`; step 15: reward_mean `0.8667`，reward_std `0.0615`，blank_accuracy `0.0`，constraint_validity `0.0`，unique_ratio `1.0`; heartbeat 到 step 17 `generate_start` | 当前不是全零 collapse，也不是收敛；有效棋盘信号在 step 10 恢复、step 15 又回落，说明策略仍在浅层 format/clue reward 和有效填空 reward 之间波动。grad 有脉冲但低于旧 run step 25 的崩塌强度，暂不重启。 |
| GSM8K RL | `Running` | step 50: reward_mean `0.1919`，reward_std `0.0382`，parse_rate `1.0`，answer_acc `0.0`; step 45: grad_norm `0.0120`，nan_grad_params `0` | shaped reward 稳定非零；exact answer 仍未稳定提升，暂不重启。 |

当前结论：两个 rjob 都在运行。Sudoku 保守版尚未收敛，但也没有复现之前全零 reward/高 KL 失控。下一轮继续看 step 20/25；若再次出现 `zero_frac=1` 或 `degenerate=1`，立即停机修复；若只是长期停在 format/clue reward，后续优化重点转向 Sudoku reward shaping 和生成约束。

### 2026-07-02 22:58 +0800

第十三轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | heartbeat 到 step 22 `generate_start`；最新完整落盘 step 20: reward_mean `1.5944`，reward_std `0.4007`，range `[0.962, 2.152]`，blank_accuracy `0.373`，constraint_validity `0.231`，clue_preservation `0.490`，format `0.500`，degenerate `0.0`，unique_ratio `1.0`，ref_energy_kl `0.00147`，loss `0.00147` | step 20 从 step 15 的浅层 format/clue reward 回升，说明保守化重启暂未复现旧 run 的全零 reward/rollout collapse；但尚未越过旧异常最关键的 step 25，继续监控。 |
| GSM8K RL | `Running` | heartbeat 到 step 57；step 55: reward_mean `0.0552`，reward_std `0.0147`，parse_rate `1.0`，answer_acc `0.0`，answer_proximity `0.005`，partial_credit `0.050`，zero_frac `0.0`，degenerate `0.0`，unique_ratio `1.0` | shaped reward 仍非零且 rollout 未退化，但这一轮回落到主要依赖 partial_credit 的弱信号；暂不判断为崩溃，继续看后续是否恢复到 step 40/50 的 `0.19` 附近。 |

当前结论：没有触发重启条件。Sudoku reward 仍在波动但不是全零，GSM8K 也没有 NaN、skip 或 reward_std 归零。下一轮重点检查 Sudoku step 25 是否安全通过，以及 GSM8K exact/answer_proximity 是否持续下滑。

### 2026-07-02 23:16 +0800

第十四轮监控与修复：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | step 25 复现退化，旧 rjob 已停止：`d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-0fa37` | step 25: reward_mean `0.0`，reward_std `0.0`，zero_frac `1.0`，advantage_var `0.0`，degenerate_group_rate `1.0`，ref_energy_kl `0.5587`，loss `0.5587`，format/clue/blank/validity 全为 `0.0`；但 `skip_consensus_start` 中 `local_skip=0.0` | 根因不是 checkpoint 加载，而是 Sudoku 脚本把 `SKIP_DEGENERATE_THRESHOLD=1.01`、`MIN_REWARD_STD_TO_UPDATE=0.0`，等于关闭了 hard skip；all-zero rollout 被允许进入 KL-only 更新，产生高 KL loss 且没有任务 reward 信号。 |
| GSM8K RL | `Running`，继续保留 | step 60: reward_mean `0.2254`，reward_std `0.0559`，answer_proximity `0.1754`，parse_rate `1.0`，answer_acc `0.0`，degenerate `0.0`，nan_grad_params `0` | shaped reward 从 step 55 低点恢复，未触发重启条件。 |

修复动作：

1. `openebm/elm/runs/run_ebt_sudoku_rl.sh`
   - 恢复 `SKIP_DEGENERATE_THRESHOLD=0.9`。
   - 恢复 `MIN_REWARD_STD_TO_UPDATE=1e-4`。
   - 更新注释：skip 分支会返回 graph-attached zero loss，并在 optimizer hook 中将所有 grad 置为 `None`，避免 Muon/AdamW/weight_decay 更新参数。
2. `openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
   - 同步上述默认值与注释，确保 rjob 默认启用 all-zero / zero-variance rollout guard。

验证：

- `bash -n openebm/elm/runs/run_ebt_sudoku_rl.sh`
- `bash -n openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`

下一轮：提交并推送修复后，重新提交 Sudoku rjob；新 run 的关键判据是 step 25 类似坏 batch 应打印 `SKIPPED` / `skipped_step=1.0`，且不能出现 KL-only 参数更新。

### 2026-07-02 23:19 +0800

Sudoku skip-guard 修复版重启：

| 项目 | 值 |
| --- | --- |
| 修复提交 | `59c3615 fix(rl): skip degenerate sudoku rollouts` |
| 旧 Sudoku rjob | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-0fa37`，已确认 `Stopped` |
| 新 Sudoku exp_id | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-231904` |
| 新 Sudoku rjob metadata | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-fb2b4` |
| 提交时状态 | `Inqueue` / replica `STARTING` |
| 预期报告路径 | `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-231904/sft_train/sudoku_rl_analysis_report.md` |

下一轮：确认新 rjob 进入 `Running` 并开始写入 `train.log`；随后重点检查 step 25 附近是否触发 `skipped_step=1.0`，而不是再次执行 KL-only 更新。

### 2026-07-02 23:27 +0800

第十五轮监控与二次修复：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | 新 rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-fb2b4` 已停止 | step 0 rank0 指标健康：reward_mean `1.5721`，reward_std `0.4567`，degenerate_group_rate `0.0`，unique_ratio `0.75`；但 `skip_consensus=any` 下 `skip_rank_count=1` 导致 `global_skip=True`、整步 `SKIPPED` | 第一版 skip guard 修复了 KL-only 坏更新，但 `any` 对 8 卡、每 rank 1 prompt 过于保守：单个 rank 的坏 rollout 会丢弃其它 rank 的健康梯度，容易造成高 skip rate 和训练停滞。 |
| GSM8K RL | `Running`，继续保留 | heartbeat 到 step 70，上一完整 step 60 reward_mean `0.2254`、reward_std `0.0559`、nan_grad_params `0` | 未触发重启条件。 |

二次修复动作：

1. `openebm/elm/rl/ebm_grpo_trainer.py`
   - 新增 `skip_consensus=local` 模式。
   - 若所有 rank 都坏：全局 skip，并在 optimizer hook 中清空 grad 为 `None`。
   - 若仅部分 rank 坏：坏 rank 返回 zero full-graph loss，避免 KL-only 更新；健康 rank 正常计算 loss，并通过 DDP all-reduce 提供有效梯度。
2. `openebm/elm/rl/ebm_grpo_config.py`、`openebm/elm/rl/train_rl_sudoku.py`
   - 将 Sudoku 默认 `skip_consensus` 改为 `local`，CLI choices 增加 `local`。
3. `openebm/elm/runs/run_ebt_sudoku_rl.sh`、`openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
   - rjob 默认 `SKIP_CONSENSUS=local`。

验证：

- `bash -n openebm/elm/runs/run_ebt_sudoku_rl.sh`
- `bash -n openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh`
- `python -m py_compile openebm/elm/rl/ebm_grpo_trainer.py openebm/elm/rl/ebm_grpo_config.py openebm/elm/rl/train_rl_sudoku.py`

下一轮：提交并推送二次修复，重新提交 Sudoku rjob；新 run 预期 step 0 不应因为单个坏 rank 整步 skip，日志中应看到 `skip_consensus=local`，`global_skip=False`，并继续走正常 loss/backward。

### 2026-07-02 23:30 +0800

Sudoku local-skip 修复版重启：

| 项目 | 值 |
| --- | --- |
| 修复提交 | `418599c fix(rl): localize sudoku skip consensus` |
| 旧 Sudoku rjob | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-fb2b4`，已确认 `Stopped` |
| 新 Sudoku exp_id | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-233012` |
| 新 Sudoku rjob metadata | `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` |
| 提交时状态 | `Inqueue` / replica `STARTING` |
| 预期报告路径 | `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-233012/sft_train/sudoku_rl_analysis_report.md` |

下一轮：确认新 rjob 进入 `Running`，检查 hparams 中 `skip_consensus=local`，并读取 step 0 是否从 `global_skip=True` 变为可正常更新。

### 2026-07-02 23:37 +0800

第十六轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` | step 0: reward_mean `1.5721`，reward_std `0.4567`，blank_accuracy `0.3846`，constraint_validity `0.1990`，unique_ratio `0.75`；`skip_consensus_done`: `skip_consensus=local`，`skip_rank_count=1`，`global_skip=False`，`world_size=8`; 后续正常 `loss_ready`，loss `1.59e-7`，`skipped_step=0.0` | 二次修复通过启动验证：单个坏 rank 不再导致全局 skip，健康 rank 的有效 reward 信号继续进入训练；坏 rank 会按 local 模式零贡献，避免 KL-only 更新。 |
| GSM8K RL | `Running` | heartbeat 到 step 74；step 65 reward_mean `0.0881`，reward_std `0.0088`，step 70 reward_mean `0.1322`，reward_std `0.1024`，nan_grad_params `0` | 仍有非零 shaped reward，无崩溃；exact answer 尚未稳定提升。 |

当前结论：Sudoku local skip 逻辑修复有效。下一轮继续看 step 5/10 是否正常更新，并最终验证 step 25 附近 all-zero rollout 是否被全局 skip 或局部 zero，而不再执行 KL-only 更新。

### 2026-07-02 23:48 +0800

第十七轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running` | 当前 train.log 最新仍为 step 0：`global_skip=False`、`skip_rank_count=1`、loss `1.59e-7`、reward_mean `1.5721`；尚未刷出 step 0 `backward_done` 或 step 5 | 未见崩溃/异常关键字；考虑到前序 run 也存在长时间反向与日志刷盘延迟，暂不判断卡死。下一轮继续看 `backward_done`、step 5，以及是否出现 NaN/RuntimeError。 |
| GSM8K RL | `Running` | step 65 reward_mean `0.0881`、reward_std `0.0088`；step 70 reward_mean `0.1322`、reward_std `0.1024`；nan_grad_params `0` | 运行健康但 exact answer 仍未稳定提升；继续监控，不重启。 |

当前结论：Sudoku 第三版仍处于运行态，local skip 首轮验证有效，但尚未产生新的 step 日志；继续等待下一轮。

### 2026-07-02 23:49 +0800

第十七轮补充：

- 纠正 heartbeat 路径：Sudoku 第三版 heartbeat 位于 `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-233012/logs/heartbeat.json`，不是 `sft_train/logs/heartbeat.json`。
- 正确 heartbeat 显示当前已到 step 5 `generate_start`，时间戳 `2026-07-02 23:48:18`。
- 因此当前不是卡死，只是 `train.log` 尚未刷出 step 5 完整指标；下一轮继续等待 step 5 `rollout_ready/loss_ready/backward_done`。

### 2026-07-02 23:56 +0800

第十八轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，heartbeat 到 step 7 `generate_start` | step 5: reward_mean `0.9253`，reward_std `0.1394`，range `[0.483, 0.966]`，blank_accuracy `0.0`，constraint_validity `0.0`，unique_ratio `0.9167`; `skip_consensus=local`，`skip_rank_count=1`，`global_skip=False`; loss `1.25e-6`; grad_norm `2.2634`，max_param_grad `0.5610`，nan_params `0` | local-skip 连续验证通过：单个坏 rank 不再导致整步 skip，且 step 5 有正常 backward/grad；reward 回落到 format+clue 信号为主，属于旧 run 也出现过的波动，暂不重启。 |
| GSM8K RL | `Running`，heartbeat 到 step 82 | step 75: reward_mean `0.1230`，reward_std `0.0703`，parse_rate `1.0`，answer_acc `0.0`，nan_grad_params `0` | 运行健康但 exact answer 仍无稳定提升；继续观察。 |

当前结论：Sudoku 第三版已通过 step 0/5 的 local-skip 和梯度健康检查。下一轮继续等待 step 10/15，最终重点验证 step 25 是否避免 KL-only collapse。

### 2026-07-03 00:15 +0800

第十九轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` | heartbeat 从 step 11 `generate_start` 推进到 step 12 `skip_consensus_start`，时间戳 `2026-07-03 00:14:38`，`local_skip=0.0`，`unique_ratio=0.8333`；`train.log` 仍只完整落盘到 step 5 | 不是训练 hang；当前问题是主 `train.log` 落盘滞后/未继续刷新，短期以 heartbeat 判断进度。step 12 本地没有触发 skip，说明这一批仍有可训练信号；继续等待下一个完整 log_interval 指标。 |
| GSM8K RL | `Running`，rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | heartbeat 到 step 91；最新完整指标仍为 step 85：reward_mean `0.1125`，reward_std `0.0607`，parse_rate `1.0`，answer_acc `0.0`，nan_grad_params `0`；step 80 曾出现 answer_acc `0.125` | 运行健康，无 NaN/崩溃；GSM8K shaped reward 非零但 exact answer 尚未稳定提升。 |

当前结论：Sudoku 第三版仍在向前推进，local-skip 修复没有引入新的启动异常。下一轮继续等 step 15/20/25 附近的完整指标，重点确认是否再次出现 all-zero reward、KL-only loss 或全局跳步过高。

### 2026-07-03 00:27 +0800

第二十轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，heartbeat 到 step 16 `generate_start` | step 10: reward_mean `1.4172`，reward_std `0.3656`，blank_accuracy `0.2169`，constraint_validity `0.2188`，zero_frac `0.0`，degenerate_group_rate `0.0`，`skip_consensus=local`，`skip_rank_count=1`，`global_skip=False`，loss `1.10e-4`，grad_norm `2.3579`，nan_params `0`；step 15 已到 `loss_ready`，reward_mean `0.8667`，reward_std `0.0615`，`global_skip=False` | 主日志重新刷新，上一轮的日志停滞是长延迟而非 hang。Sudoku reward 有波动但保持非零，local-skip 继续按预期工作；尚未验证旧问题暴露的 step 25 附近。 |
| GSM8K RL | `Running`，heartbeat 到 step 97 | step 90: reward_mean `0.1469`，reward_std `0.0756`，parse_rate `1.0`，answer_acc `0.0`，degenerate_group_rate `0.0`，grad_norm `0.01895`，nan_params `0` | 运行稳定；shaped reward 非零但 exact answer 仍未稳定提升。 |

当前结论：两个作业均无需重启。下一轮继续等待 Sudoku step 20/25，重点确认 all-zero reward 是否被 skip/local-zero 保护，而不是再次形成 KL-only 更新。

### 2026-07-03 00:38 +0800

第二十一轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，heartbeat 到 step 19 `skip_consensus_start` | heartbeat: `local_skip=0.0`，`unique_ratio=1.0`，时间戳 `2026-07-03 00:38:20`；主日志最新完整指标仍以 step 10 为准，step 15 已到 `loss_ready`，reward_mean `0.8667`，reward_std `0.0615`，`global_skip=False` | 仍在正常推进，尚未到 step 25 关键回归点。当前没有 all-zero reward、NaN、RuntimeError 或 rjob 崩溃信号。 |
| GSM8K RL | `Running`，heartbeat 到 step 99 `backward_done` | heartbeat: grad_norm `0.019538`，max_grad `0.016206`，nan_grad_params `0`；最新完整指标 step 90 reward_mean `0.1469`，reward_std `0.0756`，parse_rate `1.0`，answer_acc `0.0` | 运行稳定；exact answer 仍未形成持续提升。 |

当前结论：本轮无需修复或重启。下一轮继续等待 Sudoku step 20/25 完整指标，验证旧退化点是否被保护逻辑覆盖。

### 2026-07-03 00:49 +0800

第二十二轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，heartbeat 到 step 23 `generate_start` | step 20: reward_mean `1.5944`，reward_std `0.4007`，blank_accuracy `0.373`，constraint_validity `0.231`，zero_frac 未见异常，`skip_rank_count=0`，`global_skip=False`，loss `0.002059`，主要来自 `ref_energy_kl=0.0021`，ratio/clip 仍为 `0`，nan_params 未见异常 | step 20 reward 和 blank/validity 信号恢复，且没有 skip/NaN/崩溃。KL 较 step 10/15 增大但当前仍是小量级，先继续观察，不重启。尚未到 step 25 关键回归点。 |
| GSM8K RL | `Running`，heartbeat 仍为 step 99 `backward_done` | heartbeat: grad_norm `0.019538`，max_grad `0.016206`，nan_grad_params `0`；最新完整指标 step 90 reward_mean `0.1469`，reward_std `0.0756`，answer_acc `0.0` | 运行稳定；没有新的异常信号。 |

当前结论：本轮无需修复或重启。下一轮继续等待 Sudoku step 25 完整指标，重点确认是否避免旧的 all-zero / KL-only collapse。

### 2026-07-03 01:08 +0800

第二十三轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` | heartbeat 已从旧问题点附近继续推进：step 27 `loss_ready`，loss `5.78e-4`；step 28 `generate_start`，时间戳 `2026-07-03 01:06:59`。主 `train.log` 完整指标仍刷到 step 20：reward_mean `1.5944`，reward_std `0.4007`，blank_accuracy `0.373`，constraint_validity `0.231`，`global_skip=False` | 已越过此前 step 25 all-zero / KL-only collapse 的时间点，rjob 未崩溃；但 step 25/27 的完整 reward 指标尚未落盘，仍需等日志刷新后确认是否只是被 local skip 保护，还是保持了非零 reward。当前没有 NaN、RuntimeError 或资源异常信号，不重启。 |
| GSM8K RL | `Running`，rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | heartbeat 到 step 107 `training_step_start`；step 100 reward_mean `0.0648`，reward_std `0.0452`，parse_rate `0.875`，answer_acc `0.0`；step 105 reward_mean `0.2589`，reward_std `0.2243`，answer_acc `0.125`；nan_params `0` | 运行健康，shaped reward 保持非零并偶发正确答案；exact/answer_acc 仍波动较大，暂不作为重启条件。 |

当前结论：两个 rjob 均继续运行。Sudoku 需要继续等待 step 25/30 完整日志，以确认旧退化点的回归验证结果；GSM8K 暂无需要干预的错误。

### 2026-07-03 01:15 +0800

第二十四轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` | heartbeat 到 step 30 `generate_start`，时间戳 `2026-07-03 01:13:21`；`sft_train/logs/train.log` 修改时间仍为 `2026-07-03 00:42:17`，完整指标仍停在 step 20；`rjob logs` 尾部同样没有比 step 20 更新的 GRPO 指标 | 训练进度和主指标日志存在明显解耦/延迟。由于 heartbeat 仍推进且 rjob 为 `Running`，暂不判断 hang，也不重启。旧 step 25 退化点已在时间线上被越过，但仍缺完整 reward/std/skip 指标确认。 |
| GSM8K RL | `Running`，rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | heartbeat 到 step 110 `training_step_start`；最新完整指标 step 105 reward_mean `0.2589`，reward_std `0.2243`，answer_acc `0.125`，loss `2.11e-6`，nan_params 未见异常 | 运行健康，继续保留；exact/answer_acc 仍是波动而非稳定提升。 |

当前结论：本轮不做代码改动或重启。下一轮继续等待 Sudoku step 25/30 的完整指标；若 heartbeat 停止更新或日志出现 NaN/RuntimeError/all-zero KL-only 更新，再进入修复分支。

### 2026-07-03 01:26 +0800

第二十五轮监控：

| 任务 | 状态 | 指标快照 | 判断 |
| --- | --- | --- | --- |
| Sudoku RL | `Running`，rjob `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-2-851cb` | heartbeat 到 step 33 `generate_start`；`train.log` 已补刷到 step 30 前半段。step 20 完整 backward: reward_mean `1.5944`，reward_std `0.4007`，grad_norm `10.5686`，nan_params `0`。step 25 本地 rank reward_mean `0.0`、reward_std `0.0`，触发 `local_zero_update=True`，`skip_rank_count=3/8`，`ref_energy_kl=0.0`，`total=0.0`，随后 backward grad_norm `7.2055`、nan_params `0`；step 30 同样在本地 rank 触发 `local_zero_update=True`，`skip_rank_count=3/8`，后续完整 backward 尚未落盘 | 旧问题点回归验证通过：all-zero rollout 没有再形成 KL-only 更新，而是被 local-zero 保护；健康 rank 仍可通过 DDP all-reduce 贡献梯度。当前不是 checkpoint 加载问题，也不是全局 reward=0 collapse，但局部 rank 连续退化仍需继续观察，若 skip_rank_count 扩大或长期持续，需要考虑调低生成退化风险或加强 reward/采样设置。 |
| GSM8K RL | `Running`，rjob `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-20-b18ca` | heartbeat 到 step 115 `training_step_start`；step 105 reward_mean `0.2589`，reward_std `0.2243`，answer_acc `0.125`，nan_params `0`；step 110 reward_mean `0.1318`，reward_std `0.0673`，answer_acc `0.0`，loss `1.81e-6`，grad_norm `0.0299`，nan_params `0` | 运行健康，shaped reward 保持非零；exact/answer_acc 继续波动。 |

当前结论：不重启。Sudoku 的保护逻辑已经挡住旧的 all-zero / KL-only 路径；下一轮继续看 step 30 backward 和 step 35/40 指标，判断局部退化是否恢复或扩大。
