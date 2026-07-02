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
