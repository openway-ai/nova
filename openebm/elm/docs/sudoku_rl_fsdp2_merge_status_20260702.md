# Sudoku RL + Train-Engine Merge Status

Date: 2026-07-02

## Branch and Inputs

| Item | Value |
|---|---|
| Fusion branch | `dev-openebm-sudoku-rl-fsdp2-merge` |
| Base branch used locally | `origin/dev-openebm-sudoku` |
| Merged PR #30 branch | `origin/dev-openebm-sudoku-rl` via merge commit `893fd3e` |
| Merged PR #33 branch | `origin/dev-openebm-train-engine` via merge commit `ea866e2` |
| Review-fix commit | `217050e fix(rl): address sudoku RL review stability issues` |
| Sudoku SFT checkpoint for optimized RL rjob | `/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-sft-freeembed-direct-1node-8gpu-20260625-190601/sft_train/checkpoints/s=step=2999-d26-ctx2048-lr5e-05-bs1x32-muon_adamw-valid_loss=valid_loss=0.4366.ckpt` |
| Sudoku RL data | `/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2` |

## Train-Engine Status

The default training path remains `--train_engine lightning_ddp`; existing DDP behavior is preserved unless a train-engine flag is explicitly set.

Supported or parsed engines in the fused branch:

| Engine | Status | Scope and limits |
|---|---|---|
| `lightning_ddp` | Default | Existing path; compatible with current exact second-order EBT objective. |
| `fsdp2` | MVP / guarded | Uses native composable FSDP2, wraps transformer blocks by default, keeps EBT-specific modules replicated, disables `torch.compile` by default, forces conservative FSDP settings, and warns on exact second-order MCMC. Muon under FSDP2 is opt-in and DTensor matrices route through guarded policy. |
| `zero-1`, `zero-2` | Implemented through Lightning `DeepSpeedStrategy` | Parameters remain replicated, so exact second-order EBT training is allowed. Muon is replaced by layered AdamW because DeepSpeed optimizer flattening is incompatible with Muon matrix updates. |
| `zero-3` | Experimental / gated | Only enabled with first-order surrogate modes (`first_order_cd`, `first_order_nce`, `proposal_aware_nce`). Exact second-order MCMC remains unsupported with ZeRO-3 parameter sharding. |
| `megatron` | Reserved | Parsed but not implemented. |

Key limitation: FSDP2 and ZeRO-3 do not yet provide a fully validated production path for the original exact second-order EBT objective. They should be treated as opt-in experimental engines.

## First-Order Approximation

The original EBT training objective backpropagates through `torch.autograd.grad(..., create_graph=True)` inside MCMC. That creates a second-order path from the final CE loss back through `grad_y E_theta`. This is fragile under parameter sharding because FSDP2/ZeRO-3 can reshard or partition parameters across the retained sampler graph.

The first-order path changes the training objective:

| Mode | Purpose |
|---|---|
| `second_order` | Default exact EBT objective. |
| `first_order_debug` | Diagnostic only; disables create-graph in MCMC but is not a full training objective. |
| `first_order_cd` | Detached MCMC negative sample + recomputed positive/negative energies. |
| `first_order_nce` | Graph-safe NCE-style surrogate. |
| `proposal_aware_nce` | Proposal-aware surrogate with optional MCMC-final proposal and ranking/relaxed-CD terms. |

This approximation exists to avoid second-order autograd crashes and excessive retained activation graphs under sharded training. It is not mathematically identical to the second-order objective.

Current status: first-order approximation has not completed sufficient validation; whether its training quality matches the second-order path is still unknown.

## EBM-RL Status

PR #30 adds initial Sudoku/GSM8K RL support around EBT:

| Area | Status |
|---|---|
| Rollout | Autoregressive rollout with special-token stop handling. |
| Reward | Sudoku format, clue preservation, blank accuracy, and full-solve rewards; GSM8K answer extraction reward. |
| Losses | Energy-GSPO, Energy-REINFORCE, and token-logprob research path. |
| Stability | KL/energy anchor, per-parameter grad clipping, collapse logging, reward-degeneration skip, and trajectory dumps. |
| Checkpoint loading | SFT checkpoint load with Lightning and `torch.compile` prefix remapping. |
| Reports | `openebm/elm/scripts/analyze_rl_run.py` writes Markdown, CSV, and plots from `logs/train.log`. |

Current status: Sudoku RL has not converged yet. The optimized rjob is meant to test stability fixes and collect a fresh analysis report, not to claim convergence.

## Review Comment Fixes

| Review item | Resolution |
|---|---|
| Skip-degenerate zero-fill might move Muon parameters | Added `openebm/elm/rl/optimizer_utils.py` and wrapped MuonAdamW so Muon groups with missing gradients are skipped instead of zero-filled. Added a unit test for no-update behavior. |
| SFT checkpoint missed `model.transformer._orig_mod.* -> transformer.*` remap | Updated both Sudoku and GSM8K RL checkpoint loaders to strip `model.` and transformer-scoped `_orig_mod.` prefixes. |
| Sudoku/GSM8K iterable datasets only changed seed but repeated indices across ranks/workers | Added `openebm/elm/rl/data_sharding.py`; each epoch now shuffles once and assigns disjoint rank/worker strides. Added a coverage/disjointness unit test. |
| DDP skip consensus should be `any` | Changed config, CLI default, and `run_ebt_sudoku_rl.sh` default to `any`, so any bad rank skips the global step. |
| Analyzer still referenced removed metrics | Replaced `energy_ppo_kl` parsing with `old_policy_energy_drift` and renamed internal clamp summary to `log_ratio_clamp_rate`. |

## Optimized Sudoku RL rjob

Script:

```bash
openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh
```

Submit from the fusion worktree:

```bash
bash openebm/elm/runs/rjob/run_sudoku_rl_optimized_rjob.sh
```

The submit-side script records the current fusion commit and the rjob creates a detached worktree at `/tmp/openebm-${RUN_ID}` before running:

```bash
bash openebm/elm/runs/run_ebt_sudoku_rl.sh
```

Default optimized run configuration:

| Key | Value |
|---|---|
| `RUN_ID` | `d26-ctx2048-sudoku-rl-fsdp2-merge-<timestamp>` |
| GPUs | `8` |
| `RL_LOSS_TYPE` | `energy_gspo` |
| `MAX_STEPS` | `1000` |
| `NUM_GENERATIONS` | `12` |
| `TEMPERATURE` | `0.70` |
| `TOP_P` | `0.80` |
| `BETA` | `0.5` |
| `ENERGY_KL_MODE` | `symmetric_huber` |
| `SKIP_CONSENSUS` | `any` |
| `MIN_REWARD_STD_TO_UPDATE` | `0.01` |
| `MIN_UNIQUE_COMPLETION_RATIO_TO_UPDATE` | `0.3` |

The script writes metadata to:

```text
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/config/fusion_rjob_metadata.txt
```

## Analysis Report

The RL run script invokes:

```bash
python openebm/elm/scripts/analyze_rl_run.py "${EXP_SFT_DIR}"
```

The rjob wrapper runs it again after training if `logs/train.log` exists. Expected outputs:

```text
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_report.md
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/sudoku_rl_analysis_report.md
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_metrics.csv
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_overview.png
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_metrics.png
${EBT_RUNS_ROOT}/${RUN_ID}/sft_train/analysis_stability.png
```

`analysis_report.md` is the detailed single-run analyzer report. `sudoku_rl_analysis_report.md` is a compact comparison report generated by `openebm/elm/scripts/generate_sudoku_rl_analysis_report.py`; it compares the optimized run against `BASELINE_RL_RUN_DIR`, which defaults to:

```text
/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/ebt_runs/d26-ctx2048-sudoku-rl-gspo-20260531/sft_train.v5
```

The reports summarize reward, solve/full-solve trend, degeneration/skips, gradient health, warnings/errors, checkpoint snapshots, and baseline deltas.
