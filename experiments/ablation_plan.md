# EBT vs NanoChat Ablation Plan

## Motivation

We want to fairly compare EBT xxs against nanochat d6. Both use the same architecture (6 layers, 384 dim), but they differ along three axes:

1. **Context length**: EBT baseline uses 256; nanochat uses 2048
2. **Effective batch size / tokens per step**: EBT baseline uses 16 sequences × 256 = 4,096 tokens/step; nanochat uses 32 × 4 GPUs × 2048 = 262,144 tokens/step
3. **Optimizer**: EBT uses AdamW; nanochat uses Muon (DistMuonAdamW)

By varying one factor at a time, we can isolate each variable's contribution to any performance gap.

## Baseline

**EBT xxs** (`run_ebt.sh`)
**Change**: —
**Config**:
- context_length=256, batch_size_per_device=2, accumulate_grad_batches=2 (4× GPU) → effective batch: 16 sequences / 4,096 tokens
- optimizer=adamw, peak_lr=0.0002, min_lr_scale=10, weight_decay=0.01, grad_clip=1.0
- steps=10000, warmup_steps=1000
- MCMC num_steps=2, mcmc_step_size=500 (learnable, lr_multiplier=1500), denoising_init=random_noise, normalize_init=true

**Isolates**: —
→ val_bpb = **1.8016** | val_loss = 5.8108 | val_ppl = 349.1 | time ≈ 28 min

## Experiment 1: Fix Context Length

**Script**: `runs/run_ebt_ctx2048.sh`
**Change from Baseline**: `context_length` 256 → 2048
**Config**:
- context_length=2048, batch_size_per_device=2, accumulate_grad_batches=2 (4× GPU) → effective batch: 16 sequences / 32,768 tokens
- optimizer=adamw, peak_lr=0.0002, min_lr_scale=10, weight_decay=0.01, grad_clip=1.0
- steps=10000, warmup_steps=1000
- MCMC num_steps=2, mcmc_step_size=500 (learnable, lr_multiplier=1500), denoising_init=random_noise, normalize_init=true

**Isolates**: effect of context length on EBT training quality.
→ val_bpb = **1.6628** | time ≈ 1.44 h

## Experiment 2: Fix Batch Size

**Script**: `runs/run_ebt_ctx2048_bs.sh`
**Change from Exp 1**: accumulate_grad_batches 2 → 16, + lr_scaling_rule (lr scales with effective batch: 0.0002 × 128/256 = 0.0001)
**Config**:
- context_length=2048, batch_size_per_device=2, accumulate_grad_batches=16 (4× GPU) → effective batch: 128 sequences / 262,144 tokens
- optimizer=adamw, peak_lr=0.0002 (scaled to ~0.0001 via lr_scaling_rule), min_lr_scale=10, weight_decay=0.01, grad_clip=1.0
- steps=10000, warmup_steps=1000
- MCMC num_steps=2, mcmc_step_size=500 (learnable, lr_multiplier=1500), denoising_init=random_noise, normalize_init=true

**Isolates**: effect of effective batch size on EBT training quality.
→ val_bpb = **1.6066** | time ≈ 11.51 h

## Experiment 3: Fix Optimizer

**Script**: `runs/run_ebt_ctx2048_bs_muon.sh`
**Change from Exp 2**: optimizer adamw → muon (DistMuonAdamW), peak_lr 0.0002 → 0.02, lr_scaling_rule removed
**Config**:
- context_length=2048, batch_size_per_device=2, accumulate_grad_batches=16 (4× GPU) → effective batch: 128 sequences / 262,144 tokens
- optimizer=muon: 2D weight matrices → Muon at lr=0.02 (momentum=0.95, ns_steps=5, beta2=0.99); embeddings/biases/scalars → AdamW at lr=0.3; weight_decay=0.01, grad_clip=1.0
- steps=10000, warmup_steps=1000, min_lr_scale=10
- MCMC num_steps=2, mcmc_step_size=500 (learnable, lr_multiplier=1500), denoising_init=random_noise, normalize_init=true

**Isolates**: effect of Muon vs AdamW on EBT training quality.
→ val_bpb = **1.1462** | time ≈ 11.50 h

After this experiment, EBT and nanochat d6 differ only in the EBT-specific training procedure (MCMC steps, denoising, etc.), enabling a clean architectural comparison.

## NanoChat d6 Reference

**Script**: `runs/run_nanochat.sh` (run from `/root/nova`)
**Config**:
- depth=6, embed_dim=384, heads=3 (head_dim=128), context_length=2048, vocab_size=32768, window=L (full attention)
- device_batch_size=32 (4× H200), steps=10000
- optimizer=muon: matrices → Muon at lr=0.02; embeddings → AdamW at lr=0.3; unembedding → AdamW at lr=0.004; weight_decay=0.2
- warmdown_ratio=0.5 (last 50% of training), FP8=disabled
- dataset=FineWeb-Edu 100B shuffle

**Command**:
```bash
cd /root/nova
bash runs/run_nanochat.sh
```
**Sync after run**:
```bash
WANDB_API_KEY=wandb_v1_P44yHoBV4louYaIQ5dYJwS0Sdrd wandb sync --sync-all logs/wandb/
```
→ val_bpb = **0.9931** | time ≈ 9.7 min

## Implementation Notes

- `DistMuonAdamW` from `nanochat/optim.py`: 2D weight matrices → Muon, everything else (embeddings, biases, scalars) → AdamW
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
