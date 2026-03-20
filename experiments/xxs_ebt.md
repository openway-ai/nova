# EBT xxs Training Summary

## Model Architecture
- Model size: xxs
- Depth (layers): 6
- Embedding dim: 384
- Attention heads: 6
- Context length: 256
- Vocab size: 32768
- Total parameters: 31.4M
- EBT type: time_embed

## Training Configuration
- Steps: 10000
- Warmup steps: 1000
- Batch size per device: 2, accumulate_grad_batches: 2 (4x GPU) → effective batch: 16
- Peak LR: 0.0002, min LR scale: 10
- Weight decay: 0.01
- Gradient clip: 1.0
- Dataset: FineWeb-Edu (fineweb)
- Num workers: 12
- Eval interval: every 1000 steps (100 val batches)

## EBT-Specific Configuration
- MCMC num steps: 2
- MCMC step size: 500 (learnable, LR multiplier: 1500)
- Final alpha (MCMC step size): 426.5
- Denoising initial condition: random_noise
- Normalize initial condition: true

## Results
- valid_bpb_loss: **1.8016**
- valid_loss: 5.8108
- valid_perplexity: 349.1
- train_loss (final): 5.8149
- Training time: ~1674s (~28 minutes)

## Notes
- Wandb offline mode, logs at: nova/ebt/logs/wandb/offline-run-20260318_152834-rnuwai6q
- Checkpoints at: nova/ebt/logs/checkpoints/ebt-xxs-bs_256_s1_lr_0.0002_2026-03-18_15-28-35_/
