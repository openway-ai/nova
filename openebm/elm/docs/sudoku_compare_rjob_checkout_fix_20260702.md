# Sudoku Compare Rjob Checkout Fix - 2026-07-02

## Context

Branch: `dev-openebm-sudoku-rl-fsdp2-merge`

Failed rjob replica:

- `sudoku-compare-freeembed-eval-fix-11893875-cfcd2`

Failure:

```text
bash: line 1: /mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/rjob/run_sudoku_compare_freeembed_eval.sh: No such file or directory
```

## Root Cause

The submit command executed the absolute script path from the shared repository
checkout. That checkout was on `dev-openebm-sudoku-rl-fsdp2-merge`, while the
Sudoku compare scripts currently live on `dev-openebm-sudoku-compare`.

As a result, the container never reached the evaluation code. It exited before
environment initialization because the script file was absent on the active
branch.

## Fix

`openebm/elm/runs/rjob/run_sudoku_compare_freeembed_eval.sh` was added to the
fusion branch as a submit wrapper. The wrapper now:

- Resolves `COMPARE_BRANCH=dev-openebm-sudoku-compare`.
- Pins `COMPARE_COMMIT` before submission.
- In the rjob container, creates a detached worktree at
  `/tmp/openebm-sudoku-compare-${RUN_ID}` from `COMPARE_COMMIT`.
- Runs the compare script from that worktree with `INSIDE_RJOB=1`.
- Keeps checkpoints, data, logs, and report outputs under the shared
  OpenEBM paths.

This is equivalent to checking out the compare branch inside the job, but avoids
mutating the shared repository checkout used by other work.

## Compare Commit

Default compare commit at fix time:

- `dev-openebm-sudoku-compare`
- `f57a0d610c7f74ab5498712cfb7d09622a54dee7`
- `fix(pu): load compiled transformer ckpts for sudoku compare`

## Expected Output

The relaunched job uses:

- Rjob name: `sudoku-compare-freeembed-eval-fix2`
- Run prefix: `sudoku-compare-freeembed-eval-fix2`
- Results root:
  `/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare`
- Final report:
  `/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/outputs/${RUN_ID}/reports/analysis_report.md`
- Latest report symlink:
  `/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/sudoku_compare/latest_analysis_report.md`
