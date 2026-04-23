# SFT Exact Resume 当前代码改动清单

本文档描述当前

- 基线版本：`/mnt/shared-storage-user/liqinuo/nova-dev-ebt_copy`
- 当前版本：`/mnt/shared-storage-user/liqinuo/nova-dev-ebt_copy-sft-resume`

之间仍然保留的代码差异。

这份文档已经去除了之前为实验验证加入的 trace/hash 说明。当前仓库里不再保留：

- `ResumeTrace` 日志代码
- `input_hash / target_hash / rng_hash / optimizer_hash` 相关代码
- `compare_sft_resume_trace.py` 对比脚本

当前版本保留的是可提交的 SFT exact resume 主逻辑。

## 1. 当前仍有功能改动的文件

- `nova/ebt/dataset_sft.py`
- `nova/ebt/trainer.py`
- `nova/ebt/disk_aware_checkpoint.py`

另外保留的说明文档：

- `nova/ebt/SFT_EXACT_RESUME_CHANGES_0422.md`
- `nova/ebt/SFT_EXACT_RESUME_CODE_CHANGES_0423.md`

## 2. 核心改动

### 2.1 `dataset_sft.py`

这是 SFT exact resume 的主体改动。

当前版本把原本一次性、不可恢复的 SFT 数据流改成了可保存/恢复的状态机。

保留的关键变化：

1. `SFTIterableDataset` 新增持久状态字段
- `conv_buffer`
- `cursor`
- `consumed`
- `epoch`
- `it`
- `row_capacity`
- `buffer_size`
- `ddp_rank`
- `ddp_world_size`

2. 新增状态版本
- `STATE_VERSION = "sft_v1"`

3. 新增恢复相关方法
- `_can_restore(...)`
- `_initialize_fresh_state(...)`
- `_restore_state(...)`
- `_build_state_dict(...)`

4. `__iter__()` 不再每次从头开始
- `train` 时优先从 `resume_state_dict` 恢复
- `val` 仍然 fresh 初始化

5. 新增标准接口
- `get_dataloader_state()`
- `state_dict()`
- `load_state_dict()`

6. 新增 `SFTDataLoader`
- 让框架也能识别这条 dataloader 具有状态保存/恢复能力

7. `generate_sft_dataloader(...)` 真正消费 `resume_state_dict`

这部分是当前版本能做 SFT exact resume 的核心。

### 2.2 `trainer.py`

`trainer.py` 当前保留的改动集中在三件事：

1. 把 SFT dataloader 接进现有 exact resume 主链路
- `train_dataloader()` 在 `nanochat_sft` 分支里把 `resume_state_dict=resume_state` 传给 `generate_sft_dataloader(...)`

2. 保存/恢复 per-rank dataloader 和 RNG 状态
- `on_save_checkpoint()` 保存：
  - `dataloader_state_dict_by_rank`
  - `dataloader_state_dict`
  - `rng_states_by_rank`
- `on_load_checkpoint()` 按当前 rank 取回：
  - `self._dataloader_resume_state`
  - `self._rng_resume_state`
- `on_train_start()` 在第一个 training step 前恢复 RNG

3. SFT dataloader 恢复日志增强
- 如果恢复的是 SFT 状态，会打印：
  - `cursor`
  - `consumed`
  - `epoch`
  - `it`

注意：
- 之前为了实验验证加入的 `ResumeTrace`、hash、optimizer probe 已经移除
- 当前 `trainer.py` 不再保留这些实验性日志逻辑

### 2.3 `disk_aware_checkpoint.py`

当前保留的是 checkpoint 保存边界对齐修复：

1. 新增 `_temporarily_align_completed_for_save(...)`
- 在 `on_train_batch_end(...)` 中临时把 `batch_progress.completed` 对齐到保存边界
- 保存完成后再回滚

这部分是为了避免 resume 时出现“错一批”的 loop 边界问题。

## 3. 当前已经移除的实验性代码

以下内容已经从当前版本清理掉：

- `ResumeTrace` 单行日志
- `input_hash`
- `target_hash`
- `group_lrs` 的对比输出逻辑
- `rng_hash`
- `optimizer_hash`
- `adamw_first_*` / `muon_first_*` probe
- `compare_sft_resume_trace.py`
- `trace_resume_steps` / `trace_resume_enable_*` 参数
- `run_ebt_sft_liqinuo.sh`

也就是说，当前仓库里保留的是：

- SFT exact resume 主功能

而不是实验验证程序本身。

## 4. 如果要回退到基线，需要恢复哪些文件

如果你以后要把当前版本完整回退到 `nova-dev-ebt_copy` 基线，主要需要回退这 3 个文件：

- `nova/ebt/dataset_sft.py`
- `nova/ebt/trainer.py`
- `nova/ebt/disk_aware_checkpoint.py`

回退后会失去：

- SFT dataloader exact resume
- per-rank RNG exact resume
- checkpoint 边界对齐修复

## 5. 当前版本的定位

当前 `nova-dev-ebt_copy-sft-resume` 可以理解为：

- 在 `nova-dev-ebt_copy` 基础上
- 保留了 SFT exact resume 主链路
- 已清理掉仅用于实验验证的 trace/hash 对比代码

如果后面要提交 GitHub，这份文档描述的就是当前仍然保留的代码改动范围。
