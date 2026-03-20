# EBT xxs ctx2048 + Batch Size + Muon Training Summary

## Change vs Exp 2 (ctx2048 + bs)
- **Only change**: optimizer AdamW → Muon (DistMuonAdamW), peak_learning_rate 0.0002 → 0.02
- 2D weight matrices use Muon at lr=0.02; embeddings/biases/scalars use AdamW at lr=0.3

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
- Batch size per device: 8, accumulate_grad_batches: 4 (4x GPU) → effective batch: 128 sequences / 262,144 tokens
- Peak LR: 0.02 (Muon matrix LR, matching nanochat), min LR scale: 10
- Weight decay: 0.01
- Gradient clip: 1.0
- Optimizer: Muon (DistMuonAdamW) — matrix params, AdamW — embeddings/biases/scalars
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
- Run script: `nova/ebt/runs/run_ebt_ctx2048_bs_muon.sh`
- At this point EBT and nanochat d6 share: context_length, tokens/step, optimizer — only EBT-specific training differs
