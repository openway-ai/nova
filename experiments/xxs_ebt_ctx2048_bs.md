# EBT xxs ctx2048 + Batch Size Training Summary

## Change vs Exp 1 (ctx2048)
- **Only change**: accumulate_grad_batches 2 → 16, + lr_scaling_rule
- Effective tokens/step: 2 × 16 (accum) × 4 GPUs × 2048 = 262,144 (matches nanochat d6)
- LR auto-scaled by lr_scaling_rule: 0.0002 × (128 / 256) = 0.0001

## Model Architecture
- Model size: xxs
- Depth (layers): 6
- Embedding dim: 384
- Attention heads: 6
- Context length: 2048
- Vocab size: 32768
- Total parameters: 31.4M
- EBT type: time_embed

## Training Configuration
- Steps: 10000
- Warmup steps: 1000
- Batch size per device: 2, accumulate_grad_batches: 16 (4x GPU) → effective batch: 128 sequences / 262,144 tokens
- Peak LR: 0.0002 (scaled by lr_scaling_rule), min LR scale: 10
- Weight decay: 0.01
- Gradient clip: 1.0
- Dataset: FineWeb-Edu (fineweb)
- Num workers: 12
- Eval interval: every 1000 steps (100 val batches)

## EBT-Specific Configuration
- MCMC num steps: 2
- MCMC step size: 500 (learnable, LR multiplier: 1500)
- Denoising initial condition: random_noise
- Normalize initial condition: true

## Results
- valid_bpb_loss: **TBD**
- valid_loss: TBD
- valid_perplexity: TBD

## Notes
- Run script: `nova/ebt/runs/run_ebt_ctx2048_bs.sh`
