# EBT xxs ctx2048 Training Summary

## Change vs Baseline
- **Only change**: context_length 256 → 2048
- Everything else identical to baseline (`xxs_ebt.md`)

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
- Batch size per device: 2, accumulate_grad_batches: 2 (4x GPU) → effective batch: 16 sequences / 32,768 tokens
- Peak LR: 0.0002, min LR scale: 10
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
- Run script: `nova/ebt/runs/run_ebt_ctx2048.sh`
