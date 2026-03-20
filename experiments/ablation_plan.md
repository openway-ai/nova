# EBT vs NanoChat Ablation Plan

## Motivation

We want to fairly compare EBT xxs against nanochat d6. Both use the same architecture (6 layers, 384 dim), but they differ along three axes:

1. **Context length**: EBT baseline uses 256; nanochat uses 2048
2. **Effective batch size / tokens per step**: EBT baseline uses 16 sequences × 256 = 4,096 tokens/step; nanochat uses 32 × 4 GPUs × 2048 = 262,144 tokens/step
3. **Optimizer**: EBT uses AdamW; nanochat uses Muon (DistMuonAdamW)

By varying one factor at a time, we can isolate each variable's contribution to any performance gap.

## Baseline

**EBT xxs** (`run_ebt.sh`)
**Config**: context_length=256, batch_size_per_device=2, accumulate_grad_batches=2, optimizer=adamw, peak_lr=0.0002
→ val_bpb = **1.8016**

## Experiment 1: Fix Context Length

**Script**: `runs/run_ebt_ctx2048.sh`
**Change from Baseline**: `context_length` 256 → 2048
**Config**: context_length=2048, batch_size_per_device=2, accumulate_grad_batches=2, optimizer=adamw, peak_lr=0.0002
**Isolates**: effect of context length on EBT training quality.
→ val_bpb = **1.6628**

## Experiment 2: Fix Batch Size

**Script**: `runs/run_ebt_ctx2048_bs.sh`
**Change from Exp 1**: `accumulate_grad_batches` 2 → 16, `--lr_scaling_rule` enabled (scales lr = base_lr × effective_batch_sequences / 256; effective_batch = 4 GPUs × 2 × 16 = 128 sequences → 0.0002 × 128/256 = 0.0001)
**Config**: context_length=2048, batch_size_per_device=2, accumulate_grad_batches=16, optimizer=adamw, peak_lr=0.0001 (after scaling)
**Isolates**: effect of effective batch size on EBT training quality.
→ val_bpb = **1.6066**

## Experiment 3: Fix Optimizer

**Script**: `runs/run_ebt_ctx2048_bs_muon.sh`
**Change from Exp 2**: `optimizer` adamw → muon, `peak_learning_rate` 0.001 → 0.02, `--lr_scaling_rule` removed
**Config**: context_length=2048, batch_size_per_device=2, accumulate_grad_batches=16, optimizer=muon, peak_lr=0.02
**Isolates**: effect of Muon vs AdamW on EBT training quality.
→ val_bpb = **1.1462**

After this experiment, EBT and nanochat d6 differ only in the EBT-specific training procedure (MCMC steps, denoising, etc.), enabling a clean architectural comparison.

## NanoChat d6 Reference

**Script**: `runs/run_nanochat.sh` (run from `/root/nova`)
**Command**:
```bash
cd /root/nova
bash runs/run_nanochat.sh
```
**Sync after run**:
```bash
WANDB_API_KEY=wandb_v1_P44yHoBV4louYaIQ5dYJwS0Sdrd wandb sync --sync-all logs/wandb/
```
→ val_bpb = **0.9931**

## Implementation Notes

- `DistMuonAdamW` from `nanochat/optim.py`: 2D weight matrices → Muon, everything else (embeddings, biases, scalars) → AdamW with lr=0.3
- Muon integration added to `nova/ebt/trainer.py` via `--optimizer muon` flag
- Backups: `train.py.bak`, `trainer.py.bak`

## Results Summary

| Experiment | Context | Tokens/step | Optimizer | val_bpb | Time |
|---|---|---|---|---|---|
| EBT xxs baseline | 256 | 4,096 | AdamW | 1.8016 | 0.47h |
| Exp 1: ctx2048 | 2048 | 16,384 | AdamW | 1.6628 | 1.44h |
| Exp 2: ctx2048 + bs | 2048 | 262,144 | AdamW | 1.6066 | 11.51h |
| Exp 3: ctx2048 + bs + muon | 2048 | 262,144 | Muon | 1.1462 | 11.50h |
| nanochat d6 | 2048 | 262,144 | Muon | 0.9931 | 0.16h |
