# SFT Exact Resume Changes 2026-04-22

## Summary
本次修改目标是为 `nanochat_sft` 路径补齐 exact resume 能力，并尽量复用现有 pretrain resume 框架。

当前实现结果：

- SFT dataloader 现在支持保存和恢复运行时状态
- checkpoint 继续使用现有 `dataloader_state_dict_by_rank` / `rng_states_by_rank`
- SFT 分支现在会在 `train_dataloader()` 中消费 checkpoint 恢复出来的 per-rank dataloader state
- RNG、optimizer、lr scheduler 继续复用已有恢复逻辑

## Code Changes
### 1. `dataset_sft.py`
`SFTIterableDataset` 从“纯局部变量迭代器”改成“可状态化对象”。

新增/调整内容：

- `resume_state_dict`
- `last_state_dict`
- `get_dataloader_state()`
- 运行时状态对象属性：
  - `conv_buffer`
  - `cursor`
  - `consumed`
  - `epoch`
  - `it`
  - `buffer_size`
  - `row_capacity`
  - `rank`
  - `world_size`
- `state_version = "sft_v1"`

保存语义：

- state 表示“当前 batch 已经产出之后，下一批将从哪里继续”
- 不再依赖重新跳过若干 batch
- 直接从保存的 `conv_buffer + cursor + consumed + epoch + it` 继续生成

兼容策略：

- 若 `resume_state_dict` 缺失，或不是 `sft_v1/conv_buffer` 风格状态，则忽略恢复，按原始 fresh 流启动

### 2. `trainer.py`
SFT 分支补齐了 exact resume 接线。

改动内容：

- `train_dataloader()` 的 `nanochat_sft` 分支现在会传入：
  - `resume_state_dict=resume_state`
- `on_load_checkpoint()` 的 dataloader 恢复日志对 SFT state 做了单独打印：
  - `cursor`
  - `consumed`
  - `epoch`
  - `it`

未改内容：

- `on_save_checkpoint()` 继续沿用现有 per-rank dataloader/RNG 保存逻辑
- `on_train_start()` 继续沿用现有 per-rank RNG 恢复逻辑
- optimizer / scheduler / loop state 逻辑未改

## State Format
SFT dataloader state 结构：

```python
{
    "state_version": "sft_v1",
    "split": "train",
    "cursor": int,
    "consumed": int,
    "epoch": int,
    "it": int,
    "buffer_size": int,
    "row_capacity": int,
    "rank": int,
    "world_size": int,
    "conv_buffer": list[list[int]],
}
```

其中：

- `cursor`：下一个要从 `TaskMixture` 读取的位置
- `consumed`：已消耗样本计数
- `epoch`：数据流轮次
- `it`：已产出的 train batch 数
- `conv_buffer`：当前尚未用完的 conversation buffer

## Validation Guidance
建议验证顺序：

1. 先做短程 SFT save/resume smoke test
2. 再做正式 A/B 95-step 对照
3. 比较：
   - dataloader state
   - RNG
   - LR
   - loss 45-95 曲线

若后续实验出现 resume 边界错位，再单独补 checkpoint 保存边界修复。
