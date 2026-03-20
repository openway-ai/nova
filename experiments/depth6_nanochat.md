# Nanochat d6 Training Summary

## Model Architecture
- Depth (layers): 6
- Embedding dim: 384
- Attention heads: 3 (head dim: 128)
- Context length: 2048
- Vocab size: 32768
- Window pattern: L (full attention)

## Training Configuration
- Steps: 10000
- Device batch size: 32 (4x H200 GPUs)
- Total batch size: auto (Chinchilla-optimal scaling)
- Optimizer: Muon (matrices) + AdamW (embeddings/unembedding/scalars)
- Matrix LR: 0.02, Embedding LR: 0.3, Unembedding LR: 0.004
- Weight decay: 0.2
- Warmdown ratio: 0.5 (last 50% of training)
- FP8: disabled
- Dataset: FineWeb-Edu 100B shuffle

## Results
- val_bpb: **0.9931**
- Training time: ~581s (~9.7 minutes)
- Eval interval: every 1000 steps

## Notes
- Wandb disabled (--run dummy), no wandb logs saved
- Checkpoint saved at: ~/.cache/nanochat/base_checkpoints/d6/model_010000.pt
