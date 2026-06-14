# Sudoku Compare Staged Evaluation

Date: 2026-06-14

## 诊断与目标

原先 `run_all_sudoku_compare.sh` 是一次性顺序执行：EBM、LLM small、Qwen3.6-27B、DeepSeek-R1，最后统一 aggregate/case study。这样要等所有模型完整跑完才能看到对比结果。

本次新增 staged evaluation：通过 `STAGE_SIZE` 或 `--stage-size` 按固定 test 索引顺序分阶段推进，每轮结束后立即输出本轮和累计报告。

默认 `STAGE_SIZE=0`，完全保持原一次性全量行为。

## 整体时序

设 `--stage-size 100`，`START_INDEX=0`，目标总数为 test split 全部 10000 条，则每轮流程为：

```text
for stage in 1..ceil(total / stage_size):
    stage_start = START_INDEX + (stage - 1) * stage_size
    stage_end   = min(START_INDEX + stage * stage_size, START_INDEX + total)

    # 当前轮新增区间
    current = [stage_start, stage_end)

    # 实际传给各 evaluator 的累计区间
    cumulative = [START_INDEX, stage_end)

    LLM small: Qwen3-1.7B + Llama-3.2-1B
    Qwen3.6-27B
    DeepSeek-R1
    EBM, placed last
    write reports/current for current
    write reports/cumulative for cumulative
    update staged_manifest.json
```

LLM evaluator 始终带 `--resume`。第 2 轮虽然选择累计 `[0,200)`，但已有 `[0,100)` 会跳过，只补跑 `[100,200)`；summary 和 trace 则保持累计视图。

EBM 放在每轮最后。它会先尝试复用已有 EBM rows；只有找不到覆盖累计区间的 EBM 结果时，才运行 EBM evaluator。EBM staged fallback 会使用 `--resume --keep_shards`，避免下一轮重新推理已经完成的样本。

## 新增用法

环境变量方式：

```bash
STAGE_SIZE=100 bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

命令行参数方式：

```bash
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh --stage-size 100
```

只覆盖前 1250 条，每轮 100 条：

```bash
NUM_SAMPLES=1250 STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

复用指定 EBM 结果，避免重复推理：

```bash
EBM_REUSE_DIR=/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_ebm \
STAGE_SIZE=100 \
bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

默认 staged run id 为 `staged_<STAGE_SIZE>`，例如 `staged_100`。这让同一命令中断后直接重跑能回到同一输出目录续跑。若要开始全新 staged 实验，显式设置：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S) STAGE_SIZE=100 bash /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

## 输出结构

staged 模式沿用原输出根目录：

```text
runs/sudoku_compare/outputs/<RUN_ID>/
  ebm/
  llm/<model>/
  reports/
    stages/stage_0001/current/
    stages/stage_0001/cumulative/
    stages/stage_0002/current/
    stages/stage_0002/cumulative/
    latest_stage_current -> stages/stage_XXXX/current
    latest_stage_cumulative -> stages/stage_XXXX/cumulative
  staged_manifest.json
```

每个 stage report 目录包含：

- `comparison.csv`
- `comparison.md`
- `case_study.md`
- `summary.json`

`current` 是本轮区间 `[stage_start, stage_end)` 的结果；`cumulative` 是截至当前的累计区间 `[START_INDEX, stage_end)`。

## 断点续跑机制

LLM：

- 每个模型继续写自己的 `llm/<model>/test.jsonl`、`summary.json`、`progress.jsonl`、`traces/`。
- `eval_llm_sudoku.py --resume` 会读取已有 `test.jsonl`，只生成缺失 idx。
- 每轮传入累计区间，所以 summary/traces 一直是截至当前 stage 的累计结果。

EBM：

- 每轮先调用 `staged_eval.py materialize-ebm`，从 `EBM_REUSE_DIR`、上一轮 latest EBM、当前 `EBM_OUT_DIR` 中寻找覆盖累计区间的 rows。
- 若找到覆盖 rows，则物化当前累计 EBM view 到 `EBM_OUT_DIR/results/test.jsonl` 和 `summary.json`。
- 若找不到，则运行 `eval_sudoku_samples.py --resume --keep_shards --start_index_test START --num_samples_test COUNT`。
- EBM fallback 保留 shards，下一轮能跳过已完成 idx。

全局：

- `staged_manifest.json` 记录每轮 start/finish/skip 事件。
- `staged_manifest.json.models.<model>.completed_end` 记录每个模型已完成到哪个绝对 test idx 右边界。
- `logs/<RUN_ID>/status.tsv` 记录每条实际命令、日志和返回码。

## 样本切分与边界

样本顺序固定为 test split 原始顺序，不打乱。区间语义统一为 `[start, end)`。

最后一轮不足 `STAGE_SIZE` 时自动缩短。例如 `NUM_SAMPLES=3 STAGE_SIZE=2` 会执行：

- stage 1: current `[0,2)`, cumulative `[0,2)`
- stage 2: current `[2,3)`, cumulative `[0,3)`

若 `STAGE_SIZE` 大于目标样本数，则退化为单轮。

## 评估口径

报告中的主要字段：

- `board_acc`: 完整 9x9 全部 cell 正确才算 solved。
- `cell_acc`: 只统计原始题目空格位置的 filled-cell accuracy。
- `valid_rate` / `parsed_pct`: 能否解析出完整 1-9 的 9x9 grid。
- 解析失败样本计入 `n` 分母，`board_acc=0`，`cell_acc` 对该样本贡献为 0 个正确 filled cell。

该口径与现有 `common.summarize_rows()` 一致，因此 staged、LLM summary、EBM materialized summary 的 acc 可直接比较。

## 轨迹存储 Review

LLM 原始轨迹仍由 `eval_llm_sudoku.py` 维护：

- `test.jsonl`: 轻量累计结果，默认 response 可截断。
- `traces/all.jsonl`: 完整轨迹，含输入题目、完整模型输出、解析答案、正确答案、判定结果、parser 信息、tokens、耗时。
- `traces/parse_failed.jsonl`: 解析失败独立文件。
- `traces/correct_examples.jsonl` / `wrong_examples.jsonl` / `near_miss.jsonl` / `invalid_sudoku.jsonl`: case study 代表样本。

staged 模式不会并发写同一个模型目录。小模型并发时每个模型写各自目录，因此没有同文件写冲突。每轮结束时 trace exports 会按 idx 去重并原子重写，保证累计视图一致。

本轮 review 还修复了大模型 `run_summary.json` 覆盖问题：Qwen3.6-27B 和 DeepSeek-R1 现在分别写 `run_summary_<model>.json`。

另一个 review 发现是 manifest 更新性能：最初 `stage-event` 会导入完整 common/EBM 依赖，dry-run 中每次事件会有明显延迟。现在 `staged_eval.py` 对 common 做懒加载，`stage-event` 只执行轻量 JSON 读写，避免在多 stage 实验中放大为额外瓶颈。

## 修改文件

- `openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh`
  - 新增 `--stage-size` / `--chunk-size` 参数解析。
  - 新增 staged loop、manifest 事件、EBM 复用/物化、每轮 current/cumulative report。
  - staged 模式默认 `RUN_ID=staged_<STAGE_SIZE>`，便于中断后直接重跑续跑。
  - EBM fallback 使用 `--resume --keep_shards`。
  - Qwen27/R1 改为独立 `run_summary_<model>.json`。

- `openebm/elm/scripts/sudoku_compare/staged_eval.py`
  - 新增 staged helper：`total`、`materialize-ebm`、`report`、`stage-event`。
  - 负责 EBM rows 覆盖检测、EBM staged view 物化、current/cumulative aggregate、case study、manifest 更新。
  - `stage-event` 使用轻量 JSON writer，common/EBM 依赖仅在 total/materialize/report 子命令中懒加载。

- `openebm/elm/scripts/eval_sudoku_samples.py`
  - 新增 `--start_index`、`--start_index_test` 等参数。
  - EBM evaluator 支持任意确定性 `[start, stop)` 范围。
  - 日志打印 `index_ranges`，并注明 tqdm 是 rank0 本地 shard，不是全局样本总数。

- `openebm/elm/docs/sudoku_compare_staged_eval.md`
  - 本文档。

## 验证记录

已执行：

- `bash -n openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh`
- `py_compile`:
  - `openebm/elm/scripts/sudoku_compare/staged_eval.py`
  - `openebm/elm/scripts/eval_sudoku_samples.py`
  - `openebm/elm/scripts/sudoku_compare/eval_llm_sudoku.py`
  - `openebm/elm/scripts/sudoku_compare/common.py`
  - `openebm/elm/scripts/sudoku_compare/case_study.py`
- staged dry-run:

```bash
RESULTS_ROOT=/tmp/openebm_sudoku_staged_dryrun \
DRY_RUN=1 VERIFY_ENV=0 RUN_EBM=1 RUN_LLMS=1 RUN_SMALL=1 RUN_QWEN27=1 RUN_R1=1 RUN_REPORTS=1 \
GPU_COUNT=2 GPU_IDS=0,1 NUM_SAMPLES=3 STAGE_SIZE=2 RUN_ID=codex_staged_dryrun \
bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

验证结果：

- test split size 识别为 10000。
- `NUM_SAMPLES=3 STAGE_SIZE=2` 正确分为两轮。
- 每轮顺序为 Small → Qwen27 → R1 → EBM materialize → report。
- 第 1 轮累计 `--num-samples 2`，第 2 轮累计 `--num-samples 3`。
- 小模型并发命令分别设置 `CUDA_VISIBLE_DEVICES=0/1`。
- `staged_manifest.json.models` 中所有模型 `completed_end=3`。

还用伪造 rows 执行了 `staged_eval.py materialize-ebm` 和 `staged_eval.py report`，确认 `current/cumulative` 的 `comparison.csv`、`comparison.md`、`case_study.md`、`summary.json` 能实际落盘。

`staged_eval.py stage-event` 轻量路径约 0.4 秒完成，不再触发 heavy import。

未启动真实大模型/EBM 推理；真实运行会占用 GPU 并耗时较长。
