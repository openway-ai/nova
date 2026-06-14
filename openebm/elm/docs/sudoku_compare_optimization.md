# Sudoku Compare Optimization Notes

Date: 2026-06-14

## 诊断结论

`qwen3p6_27b: 0%|0/1` 这条进度在旧实现里表示 batch 数，不表示 test split 样本总数。最新 smoke 日志中的命令包含 `--num-samples 1 --batch-size 1`，因此 tqdm 的 `total=1` 是正常的单样本 smoke 表现。

这次日志不是永久卡死。`qwen3p6_27b` 先经历 vLLM 多卡引擎初始化、权重加载、KV cache profiling 和 FlashInfer JIT/autotune，然后单个样本生成到 `max_tokens=8192`，耗时约 369.6s，最后正常写出 `summary.json`。真正的问题是旧进度条太粗，且 Qwen3.6-27B 输出长推理时没有及时给出最终 9x9 答案，导致生成预算被打满。

## 改动清单

### `openebm/elm/scripts/sudoku_compare/common.py`

- 重写 LLM Sudoku 答案抽取逻辑：优先抽取最后一个完整 1-9 的 9x9 最终答案。
- 不再把包含 `0`/`.` 的复述题面或中间未完成网格当成已解析答案。
- 支持 `<think>...</think>` 后答案、`Final answer:`/`Solution:`/`Answer:` 等标记、Markdown 表格、9 行网格、连续 81 位数字。
- `ParseResult` 增加 `failure_reason` 和 `candidate_count`，便于定位解析失败。

### `openebm/elm/scripts/sudoku_compare/eval_llm_sudoku.py`

- tqdm 改成样本级 total：`total=len(pending)`，不再显示 batch 总数造成误解。
- 每个 batch 写入 `progress.jsonl`，包含 `batch_start/batch_end`、已完成样本数、batch tokens、累计 tokens、tokens/s、samples/s。
- 每个模型 `config.json` 写入实际 runtime 信息：`CUDA_VISIBLE_DEVICES`、torch 可见 GPU、GPU 名称/显存、TP/PP/DP、batch size、launcher strategy。
- vLLM 增加 `--data-parallel-size` 和 `--data-parallel-size-local` 参数，但默认保持 `data_parallel_size=1`。第二轮 code review 确认 vLLM 0.18 的 offline `LLM(data_parallel_size>1)` 会拒绝 single-process 用法，因此脚本默认改为小模型多进程分 GPU 并发 + 大 batch。
- `--data-parallel-size > 1` 现在必须显式配合 `--distributed-executor-backend external_launcher`，否则会在加载模型前直接报错，避免挂死在 vLLM 初始化阶段。
- `--overwrite` 会清理旧 `traces/` 导出，避免异常中断后旧轨迹残留。
- resume 旧结果时，如果已有 `test.jsonl` 但缺少 `traces/all.jsonl`，会从 `test.jsonl` 补出最低限度 trace，并标记 `trace_source=test_jsonl_response`。
- 新增完整轨迹导出到 `traces/`：
  - `all.jsonl`: 每条样本的完整轨迹。
  - `correct_examples.jsonl`: 完全答对代表样本，数量由 `--case-examples-per-type` 控制。
  - `wrong_examples.jsonl`: 答错代表样本，数量由 `--case-examples-per-type` 控制。
  - `correct_5.jsonl`: 完全答对代表样本。
  - `wrong_5.jsonl`: 答错代表样本。
  - `near_miss.jsonl`: 接近正确的样本。
  - `parse_failed.jsonl`: 解析失败样本，独立文件。
  - `invalid_sudoku.jsonl`: 可解析但不满足 Sudoku 约束的样本。
  - `typical_cases.jsonl`: 合并后的典型 case bank。
  - `trace_summary.json`: 轨迹文件摘要。

### `openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh`

- 默认 Python 使用 `/mnt/shared-storage-user/puyuan/conda_envs/nanochat/bin/python`。
- 自动检测 GPU 数量，写入 `run_env.txt` 和启动日志。
- 默认启用 vLLM + FlashInfer：
  - `ATTENTION_BACKEND=FLASHINFER`
  - `FLASHINFER_DISABLE_VERSION_CHECK=1`
  - `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer`
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
  - `--enforce-eager`
- 默认关闭 Qwen chat template thinking：`THINKING=disable`，避免无界长推理挤占最终答案。
- 分组 token budget：
  - 小模型：`SMALL_MAX_TOKENS=2048`
  - Qwen3.6-27B：`QWEN27_MAX_TOKENS=4096`
  - DeepSeek-R1：`R1_MAX_TOKENS=8192`
  - 显式设置 `MAX_TOKENS=...` 会覆盖全部组。
- 小模型默认策略：多卡时 `SMALL_PARALLEL_STRATEGY=model_parallel_processes_batched`，`SMALL_TP=1`，`SMALL_DP=1`，`SMALL_BATCH_SIZE=16`，`PARALLEL_SMALL_MODELS=1`。脚本会把小模型拆成独立进程，分别设置 `CUDA_VISIBLE_DEVICES=<gpu>`；若检测不到多卡则回退为单进程大 batch：`SMALL_BATCH_SIZE=8`。
- 大模型默认策略：tensor parallel，`QWEN27_TP=min(GPU_COUNT,4)`，`R1_TP=min(GPU_COUNT,8)`。
- 输出结构统一为：
  - `runs/sudoku_compare/logs/<RUN_ID>/`
  - `runs/sudoku_compare/outputs/<RUN_ID>/ebm`
  - `runs/sudoku_compare/outputs/<RUN_ID>/llm`
  - `runs/sudoku_compare/outputs/<RUN_ID>/reports`
  - latest symlink: `latest_outputs/latest_ebm/latest_llm/latest_reports/logs/latest`

### `openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh`

- 同步 full run 的 Python、FlashInfer、eager、thinking、分组 token budget、GPU 检测和结构化输出策略。
- 大模型 smoke TP 自动按实际 GPU 数夹紧，避免默认 TP 大于可见 GPU 数。
- 每个 smoke 模型也会生成 `progress.jsonl` 和 `traces/`。

### `openebm/elm/scripts/sudoku_compare/case_study.py`

- 默认 `--cases-per-type=5`。
- case study Markdown 中增加 parser strategy、parse failure、tokens、elapsed 等信息。
- 明确提示完整原始输出在各模型 `traces/` 目录中。

## 多 GPU 策略

脚本启动时会检测实际可见 GPU：

1. 优先用 `torch.cuda.device_count()`。
2. 如果 torch 看不到 GPU，再尝试 `nvidia-smi --query-gpu=index --format=csv,noheader`。
3. 检测结果写入 `run_env.txt`、启动 stdout、每个模型的 `config.json`。

小模型组默认使用“多进程分 GPU + 大 batch”。这是第二轮 code review 后的修正：vLLM 0.18 源码中 offline `LLM(data_parallel_size>1)` 要求 external launcher，不能在单 Python 进程里直接使用。因此默认配置如下：

```bash
SMALL_TP=1
SMALL_DP=1
SMALL_BATCH_SIZE=16
PARALLEL_SMALL_MODELS=1
```

当 `GPU_COUNT>=小模型数量` 且 `GPU_IDS` 可用时，脚本会并发启动：

```text
env CUDA_VISIBLE_DEVICES=0 ... --models qwen3_1p7b --run-summary-name run_summary_qwen3_1p7b.json
env CUDA_VISIBLE_DEVICES=1 ... --models llama3p2_1b --run-summary-name run_summary_llama3p2_1b.json
```

大模型组默认使用 tensor parallel：

```bash
QWEN27_TP=min(GPU_COUNT, 4)
R1_TP=min(GPU_COUNT, 8)
QWEN27_DP=1
R1_DP=1
```

如果需要强制顺序跑小模型：

```bash
PARALLEL_SMALL_MODELS=0 SMALL_BATCH_SIZE=8 bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

如果确实要用 vLLM external-launcher DP，需要自行使用外部 launcher，并显式传入：

```bash
DISTRIBUTED_EXECUTOR_BACKEND=external_launcher SMALL_DP=2 bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

如果要手动指定可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

依赖：

- `torch`: GPU 检测、EBM 推理、transformers backend。
- `vllm`: 默认 LLM backend，负责 TP 推理；offline DP 只在 external launcher 模式下启用，默认脚本不直接使用 single-process DP。
- `transformers`: tokenizer 和可选 transformers backend。
- `flashinfer-python` / `flashinfer-cubin`: FlashInfer attention backend。
- `nvidia-smi`: 可选 fallback GPU 检测。
- `accelerate`: 仅在 transformers backend 使用 `device_map=auto` 时相关；默认 vLLM 路径不依赖 accelerate 启动。

## 轨迹格式

LLM 轨迹位于：

```text
runs/sudoku_compare/outputs/<RUN_ID>/llm/<model>/traces/
runs/sudoku_compare_smoke/llm_one_sample/<RUN_ID>/<model>/traces/
```

目录中主要文件：

- `all.jsonl`: 每条样本的完整轨迹。
- `correct_examples.jsonl` / `wrong_examples.jsonl`: 可配置数量的代表样本。
- `correct_5.jsonl` / `wrong_5.jsonl`: 固定 5 条的兼容别名。
- `near_miss.jsonl`: 接近正确的样本。
- `parse_failed.jsonl`: 解析失败样本。
- `invalid_sudoku.jsonl`: 可解析但违反 Sudoku 约束的样本。

`all.jsonl` 每行包含：

- `idx`, `split`, `model`
- `puzzle_text`
- `prompt_used`，仅当 `SAVE_PROMPTS=1` 时写入完整 prompt
- `full_model_output`
- `parsed_answer`
- `parsed_answer_text`
- `correct_answer_text`
- `is_correct`, `fully_solved`, `parsed`, `is_valid_sudoku`
- `parser_strategy`, `parse_failure_reason`, `parser_candidate_count`
- `correct`, `total`, `given_total`, `given_correct`, `filled_total`, `filled_correct`, `filled_cell_acc`
- `elapsed_s`, `tokens_generated`
- `case_note`

原有 `test.jsonl` 仍保留，默认只存截断 response，便于轻量汇总；完整输出以 `traces/all.jsonl` 为准。

## 答案解析策略

新解析器按优先级处理：

1. 取最后一个 `Final answer:` / `Final grid:` / `Completed grid:` / `Solution:` / `Answer:` 后的内容。
2. 取 `</think>` 之后内容。
3. 取剥离 `<think>...</think>` 后的全文。
4. 最后才看 raw text。

每个 section 中优先级为：

1. 9 行网格或 Markdown 表格。
2. 连续 81 位数字。
3. 显式 answer section 中的 81 个空白分隔数字。
4. 显式 answer section 中的宽松 all-digits fallback。

只有 81 个 cell 且全部为 1-9 才算 parsed。包含 0 或空格占位的中间网格会进入 `parse_failed.jsonl`，不会再抬高 `valid_rate`。

## 运行命令

单样本 smoke：

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/smoke_one_sample_each_llm.sh
```

完整 Sudoku test split：

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

只跑小模型快速检查：

```bash
RUN_QWEN27=0 RUN_R1=0 NUM_SAMPLES=32 bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

手动覆盖并行策略：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPU_COUNT=4 \
PARALLEL_SMALL_MODELS=1 SMALL_DP=1 SMALL_BATCH_SIZE=16 \
QWEN27_TP=4 R1_TP=4 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

回退非 FlashInfer attention：

```bash
ATTENTION_BACKEND=FLASH_ATTN bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

全局 token budget 覆盖：

```bash
MAX_TOKENS=2048 bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

## 验证记录

本次改动已做：

- `bash -n` 检查两个 shell 脚本。
- `py_compile` 检查 `common.py`、`eval_llm_sudoku.py`、`case_study.py`。
- full run dry-run，确认命令中包含 `--thinking disable`、`--attention-backend FLASHINFER`、`--enforce-eager`、`--data-parallel-size`、分组 `--max-tokens`、trace 参数。
- smoke dry-run，确认四个模型命令均包含上述参数。
- 解析器样例测试，覆盖 `<think>` 后最终答案、Markdown 表格、复述题面但无完整答案。

## 2026-06-14 第二轮 Code Review 记录

本轮 review 发现并修复了以下问题：

- **vLLM offline DP 误用**：源码确认 `LLM(data_parallel_size>1)` 不支持 single-process offline usage。默认 `SMALL_DP=GPU_COUNT` 已改为 `SMALL_DP=1`，并以多进程分 GPU 并发跑小模型；`eval_llm_sudoku.py` 对 DP>1 做提前报错。
- **并发 summary 竞态**：小模型并发时不再共写 `run_summary.json`，改为 `run_summary_qwen3_1p7b.json` / `run_summary_llama3p2_1b.json`。
- **旧 trace 残留**：`--overwrite` 会清理旧 trace 导出文件。
- **resume 缺 trace**：老结果只有 `test.jsonl` 时会补出最低限度 trace，避免 case study 轨迹目录缺失。
- **trace 文件命名**：新增 `correct_examples.jsonl` / `wrong_examples.jsonl`，保留 `correct_5.jsonl` / `wrong_5.jsonl` 作为固定 5 条兼容别名。
- **环境可诊断性**：`VERIFY_ENV=1` 时记录 vLLM `LLM` 是否接受 `**kwargs`、`EngineArgs` 是否包含 `data_parallel_size`，并明确记录 offline DP 需要 external launcher。

新增验证：

- `py_compile` 通过。
- 两个 shell 脚本 `bash -n` 通过。
- `--data-parallel-size 2` 未设置 external launcher 时能在模型加载前明确失败。
- 模拟 `GPU_COUNT=2 GPU_IDS=0,1` 的 dry-run，确认小模型拆为两个并发命令并分别设置 `CUDA_VISIBLE_DEVICES=0/1`。

未在本次修复中启动真实完整 test split；真实运行会占用 GPU 和较长时间。

## 2026-06-14 第三轮 Code Review: Staged Evaluation

本轮新增分阶段顺序执行 option，详见 `sudoku_compare_staged_eval.md`。

新增能力：

- `run_all_sudoku_compare.sh` 支持 `STAGE_SIZE=<N>` 或 `--stage-size N` / `--chunk-size N`。
- 默认 `STAGE_SIZE=0`，保持原一次性全量评估行为。
- staged 模式每轮顺序固定为：
  `LLM small (Qwen3-1.7B, Llama-3.2-1B) -> Qwen3.6-27B -> DeepSeek-R1 -> EBM -> reports`。
- 每轮输出两套报告：
  - `reports/stages/stage_XXXX/current/`: 本轮 `[stage_start, stage_end)`。
  - `reports/stages/stage_XXXX/cumulative/`: 截至当前 `[START_INDEX, stage_end)`。
- staged 模式默认 `RUN_ID=staged_<STAGE_SIZE>`，同一命令中断后重跑会回到同一输出目录续跑；如需新实验可显式设置 `RUN_ID`。
- 新增 `staged_manifest.json`，记录 stage 事件与每个模型的 `completed_end`。
- EBM 每轮排在最后；先尝试从 `EBM_REUSE_DIR`、上一次 latest EBM、当前 EBM out dir 复用已有 rows，覆盖不足时才用 `--resume --keep_shards` 跑累计 EBM。
- `eval_sudoku_samples.py` 新增 `--start_index` / `--start_index_test` 等参数，EBM evaluator 支持确定性 `[start, stop)` 范围。
- EBM 日志现在打印 `index_ranges`，并明确 tqdm 是 rank0 本地 shard，不是全局样本总数。
- 大模型 run summary 覆盖问题已修复：Qwen3.6-27B 和 DeepSeek-R1 分别写 `run_summary_<model>.json`。
- staged manifest 更新路径已优化：`stage-event` 不再导入 heavy common/EBM 依赖，避免多 stage 时额外放大调度开销。

新增脚本：

- `openebm/elm/scripts/sudoku_compare/staged_eval.py`
  - `total`: 查询 split 总数。
  - `materialize-ebm`: 从已有 EBM rows 物化 staged cumulative view。
  - `report`: 写 current/cumulative comparison 和 case study。
  - `stage-event`: 更新 `staged_manifest.json`。

新增文档：

- `openebm/elm/docs/sudoku_compare_staged_eval.md`

示例：

```bash
STAGE_SIZE=100 bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

只评估前 1250 条，每轮 100 条：

```bash
NUM_SAMPLES=1250 STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

复用已有 EBM：

```bash
EBM_REUSE_DIR=/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_ebm \
STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

验证：

- `bash -n openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh` 通过。
- `py_compile` 覆盖 staged helper、EBM evaluator、LLM evaluator、common、case study。
- `NUM_SAMPLES=3 STAGE_SIZE=2 DRY_RUN=1` staged dry-run 通过，确认两轮区间、模型顺序、GPU 分配、manifest 更新。
- 使用伪造 rows 验证 `staged_eval.py materialize-ebm` 和 `staged_eval.py report` 能实际生成 current/cumulative 报告。
- `staged_eval.py stage-event` 轻量路径约 0.4 秒完成。
