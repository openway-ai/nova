# Sudoku EBM vs LLM Comparison

This directory contains a reproducible comparison pipeline for SATNet Sudoku
test-split evaluation.

## Files

- `common.py`: shared SATNet test loading, board formatting, LLM output parsing,
  and metric computation. It reuses `openebm.elm.scripts.eval_sudoku_samples`
  helpers when available.
- `eval_llm_sudoku.py`: evaluates local Hugging Face models with vLLM by
  default, or transformers as fallback.
- `aggregate_results.py`: merges EBM and LLM summaries into a markdown table
  and CSV.
- `case_study.py`: selects representative cases and exports a markdown report.
- `run_all_sudoku_compare.sh`: one-command runner for EBM, LLMs, aggregation,
  and case-study export.

## Metric Mapping

The LLM scripts align with `eval_sudoku_samples.py` on data and per-sample
fields:

- `board_acc`: `fully_solved_pct`, all 81 cells must match the reference.
- `cell_acc`: `filled_cell_acc`, only cells that were `0` in the original
  puzzle are counted.
- `valid_rate`: `parsed_pct`, the parser extracted a final 81-digit grid.
- `avg_time`: `avg_s_per_sample`.

`valid_sudoku_rate` is also saved in JSON summaries for diagnosis, but it is
not the main `valid_rate` column because the requested validity metric is
format parseability.

## Default Paths

Data:

```bash
/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2
```

Model aliases:

```text
qwen3_1p7b        /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-1.7B
llama3p2_1b      /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
qwen3p6_27b      /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.6-27B
deepseek_r1_0528 /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528
```

Output root:

```bash
/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare
```

## One-Command Run

Dry-run the full command plan first:

```bash
DRY_RUN=1 bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

Smoke test on 16 test samples, skipping the two largest models:

```bash
NUM_SAMPLES=16 RUN_QWEN27=0 RUN_R1=0 \
  bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

Full run:

```bash
bash openebm/elm/scripts/sudoku_compare/run_all_sudoku_compare.sh
```

Useful environment overrides:

```bash
RESULTS_ROOT=/path/to/out
PYTHON=/path/to/python
NUM_SAMPLES=128              # -1 means full test split
RUN_EBM=0                    # reuse existing EBM result
RUN_SMALL=0 RUN_QWEN27=0 RUN_R1=0
SMALL_BATCH_SIZE=4
QWEN27_TP=8
R1_TP=8
RESPONSE_LOG=full            # default is truncated
SAVE_PROMPTS=1               # default is not to store prompts in JSONL
```

The runner writes command logs under:

```text
<RESULTS_ROOT>/logs/
```

## 1. Run EBM on Test Split

Use the existing entry point. Passing `--out_dir` makes downstream scripts
find the result deterministically.

```bash
python -m openebm.elm.scripts.eval_sudoku_samples \
  -c /mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-mixed-v3p1-20260529/sft_train.v4/checkpoints/s=step=4184-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss_balanced=valid_loss_balanced=0.2672.ckpt \
  --splits test \
  --full_test \
  --no_per_sample_print \
  --out_dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/ebm
```

If you want to reproduce the original val+test command exactly, keep
`--full_val --full_test`; the aggregator will read the `test` section.

## 2. Run LLM Evaluations

Small models:

```bash
python -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models qwen3_1p7b llama3p2_1b \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/llm \
  --backend vllm \
  --batch-size 8 \
  --max-tokens 8192 \
  --temperature 0 \
  --seed 0 \
  --response-log truncated \
  --resume
```

27B model, example multi-GPU run:

```bash
python -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models qwen3p6_27b \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/llm \
  --backend vllm \
  --tensor-parallel-size 4 \
  --batch-size 2 \
  --max-tokens 8192 \
  --max-model-len 16384 \
  --temperature 0 \
  --response-log truncated \
  --resume
```

DeepSeek-R1-0528, example large-model run:

```bash
python -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models deepseek_r1_0528 \
  --out-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/llm \
  --backend vllm \
  --tensor-parallel-size 8 \
  --batch-size 1 \
  --max-tokens 8192 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --temperature 0 \
  --response-log truncated \
  --resume
```

Use a custom local path with `NAME=PATH`:

```bash
python -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku \
  --models my_model=/path/to/local/hf/model \
  --backend transformers \
  --batch-size 1
```

Each model writes:

```text
<out-dir>/<model>/config.json
<out-dir>/<model>/test.jsonl
<out-dir>/<model>/summary.json
```

By default, `eval_llm_sudoku.py` parses the full in-memory model output but
stores only the tail of long responses (`--response-log truncated`,
`--response-log-chars 12000`). This keeps thinking/CoT models from producing
very large JSONL files. Use `--response-log full` when you need full raw-output
auditing.

## 3. Aggregate Results

```bash
python -m openebm.elm.scripts.sudoku_compare.aggregate_results \
  --ebm-result-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/ebm \
  --llm-root /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/llm \
  --csv-out /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/comparison.csv
```

This prints a markdown table with:

```text
model / params / board_acc / cell_acc / valid_rate / avg_time
```

## 4. Generate Case Study

```bash
python -m openebm.elm.scripts.sudoku_compare.case_study \
  --ebm-result-dir /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/ebm \
  --llm-root /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/llm \
  --out-md /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/case_study.md
```

The report includes:

- EBM correct and same-scale LLM wrong.
- EBM correct and large LLM wrong.
- Low-given hard puzzles.

Predicted grids mark wrong cells with square brackets, so the markdown can be
used directly in the comparison report.
