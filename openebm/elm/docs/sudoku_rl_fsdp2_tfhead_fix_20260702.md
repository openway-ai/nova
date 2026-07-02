# Sudoku RL FSDP2 Merge TF-Head Fix - 2026-07-02

## Context

Branch: `dev-openebm-sudoku-rl-fsdp2-merge`

Fix commit: `095c68757c19bf75a5a10e0a24e48c1d8ae731c7`

Old Sudoku rjob:

- Show name: `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-170632`
- Rjob metadata name: `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-9c79e`
- Replica: `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-9c79e-cfcd2`

## Root Cause

The old rjob was not failing on the nested `_orig_mod` transformer key remap. The log showed:

- `Renamed 576 compiled/eager transformer keys to transformer.*`
- `WARNING: 1 unexpected keys (ignored): ['tf_head.proj.weight']...`
- Loaded model params: `973,643,010`

The SFT checkpoint was trained with the free-embedding direct-unembed path:

- `use_tf_head=True`
- `tf_head_type=direct_unembed`
- `free_embedding_mcmc=True`
- checkpoint key: `model.tf_head.proj.weight`

The fusion branch did not have the TF-head/free-embedding model path, so the trained vocab decoder head was silently dropped. Rollouts then produced non-Sudoku text such as repeated English fragments, `digit_count=0`, `parse_ok=False`, and all rewards stayed at zero.

## Fix

Implemented on `dev-openebm-sudoku-rl-fsdp2-merge`:

- Added `openebm/elm/tf_head.py`.
- Restored optional TF-head/free-embedding support in `openebm/elm/modeling_ebt.py`.
- Added `return_pred_hidden` support to `openebm/elm/ar_ebt_time_embed.py`.
- Updated RL energy scoring in `openebm/elm/rl/logprobs.py` to use D-dimensional token embeddings for free-embedding checkpoints.
- Hardened Sudoku/GSM8K checkpoint loaders to fail fast when trainable keys such as `tf_head.*`, `transformer.*`, `vocab_to_embed.*`, `embeddings.*`, or `alpha` are unexpected.
- Removed an unused Sudoku loader import that caused local `/root/.cache/nanochat` side effects.
- Added `openebm/elm/runs/rjob/run_gsm8k_rl_optimized_rjob.sh`.

Local verification:

- `py_compile` passed for changed Python modules.
- The free-embedding Sudoku SFT checkpoint now loads with `tf_head` present.
- Loaded model params after fix: `1,028,168,962`, which includes the direct-unembed head.

## Relaunched Rjobs

Sudoku:

- Show name / exp id: `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-182711`
- Rjob metadata name: `d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-1-5f2bd`
- Expected analysis report:
  `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-fsdp2-merge-20260702-182711/sft_train/analysis_report.md`

GSM8K:

- Show name / exp id: `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-182721`
- Rjob metadata name: `d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-18-2d801`
- Expected analysis report:
  `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-gsm8k-rl-fsdp2-merge-20260702-182721/sft_train/analysis_report.md`

At submission time both new jobs were `Inqueue / STARTING`.
