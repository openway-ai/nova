# EBM-GRPO: Sudoku RL Post-Training Pipeline

## Overview

This directory implements **Group Relative Policy Optimization (GRPO)** for Energy-Based Transformers (EBT) on 9x9 Sudoku tasks. The implementation adapts the GRPO algorithm from the d1/diffu-GRPO reference codebase to work with EBT's energy-based paradigm.

## Key Differences from d1/diffu-GRPO

| Aspect | d1 (Masked Diffusion) | EBM-GRPO (This Implementation) |
|--------|----------------------|--------------------------------|
| **Log-prob estimation** | Mask-then-predict (mask completion tokens → forward → cross-entropy) | Standard next-token prediction through MCMC chain |
| **Model architecture** | Masked diffusion LLM (LLaDA) | Energy-Based Transformer with MCMC refinement |
| **Generation** | Iterative denoising with remasking | Autoregressive with EBT's energy-guided sampling |
| **Gradient flow** | Through masked prediction | Through full MCMC chain (optional) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RL Training Loop (ebm_grpo_trainer.py)                     │
│                                                             │
│  for each batch of prompts:                                 │
│    1. Generate N completions per prompt (rollout)           │
│    2. Compute rewards (multi-component)                     │
│    3. Compute advantages (group-relative normalization)     │
│    4. For K iterations:                                     │
│       a. Compute current log-probs via EBT forward          │
│       b. Compute PPO-clipped policy gradient loss           │
│       c. Add KL penalty (vs reference model)                │
│       d. Backprop and update                                │
└─────────────────────────────────────────────────────────────┘
```

## Files

### Core Implementation

1. **`ebm_grpo_config.py`** — Configuration dataclass with all RL hyperparameters
2. **`logprobs.py`** — EBT-specific log-probability estimation
3. **`rewards.py`** — Multi-component reward functions for Sudoku
4. **`sudoku_dataset_rl.py`** — Dataset that yields prompts for generation
5. **`rollout.py`** — Batch generation/rollout using EBT's autoregressive sampling
6. **`ebm_grpo_trainer.py`** — Main GRPO trainer (PyTorch Lightning module)
7. **`train_rl_sudoku.py`** — Entry point script

### Launch Script

8. **`../runs/run_ebt_sudoku_rl.sh`** — Shell script for multi-GPU training

## Log-Probability Estimation (Key Algorithm)

The core innovation is adapting log-prob computation for EBT:

```python
def get_per_token_logps(model, input_ids, prompt_length, learning=True):
    # 1. Standard next-token framing
    model_input = input_ids[:, :-1]   # (B, S-1)
    targets = input_ids[:, 1:]        # (B, S-1)

    # 2. Run EBT forward (MCMC chain)
    predicted_distributions, _ = model.forward(
        model_input,
        learning=learning,  # gradients through MCMC if True
        return_raw_logits=True,
    )

    # 3. Take final MCMC step logits
    final_logits = predicted_distributions[-1]  # (B, S-1, V)

    # 4. Compute per-token log-probs via cross-entropy
    per_token_logps = -F.cross_entropy(
        final_logits.reshape(-1, V),
        targets.reshape(-1),
        reduction='none'
    ).view(B, S-1)

    # 5. Return only completion positions
    return per_token_logps[:, prompt_length-1:]
```

**Why this works for EBT:**
- EBT's MCMC refinement produces a sequence of distributions over tokens
- The final distribution is the model's best prediction after energy minimization
- We treat this as the policy's output and compute log-probs via cross-entropy
- When `learning=True`, gradients flow back through the entire MCMC chain

## Reward Function

Multi-component reward for 9x9 Sudoku (total range: [0.0, 3.0]):

1. **Format reward** (0.0-0.5): Can we parse a 9x9 board from the output?
2. **Cell accuracy reward** (0.0-1.5): Fraction of blank cells filled correctly × 1.5
3. **Validity bonus** (0.0-0.5): Are all Sudoku constraints satisfied?
4. **Full solve bonus** (0.0-0.5): Perfect solution?

Implementation reuses existing `sudoku_evaluator.py` for parsing and validation.

## Usage

### 1. Prepare Data

Ensure Sudoku v2 data is prepared:

```bash
python openebm/elm/data/prepare_sudoku_data_v2.py \
    --data_dir openebm/elm/data/sudoku_cache_v2
```

### 2. Train SFT Model (Prerequisite)

RL training requires a pre-trained SFT checkpoint:

```bash
bash openebm/elm/runs/run_ebt_sudoku_sft_v3.sh
```

### 3. Run RL Training

**Single GPU:**
```bash
python -m openebm.elm.rl.train_rl_sudoku \
    --sft_checkpoint /path/to/sft.ckpt \
    --max_steps 1000 \
    --num_generations 8
```

**Multi-GPU (recommended):**
```bash
export SFT_CKPT=/path/to/best_sft.ckpt
bash openebm/elm/runs/run_ebt_sudoku_rl.sh
```

### 4. Monitor Training

Metrics logged to W&B:
- `train/reward_mean` — average total reward
- `train/reward_format` — format component
- `train/reward_accuracy` — accuracy component
- `train/reward_validity` — validity component
- `train/reward_full_solve` — full solve component
- `train/policy_loss` — PPO-clipped loss
- `train/kl` — KL divergence vs reference model
- `train/clip_ratio` — fraction of clipped ratios

## Hyperparameters

### Default Configuration

```python
num_generations = 8          # completions per prompt
max_completion_length = 512  # 9x9 sudoku needs ~200-400 tokens
temperature = 0.9
top_p = 0.95

num_iterations = 1           # GRPO inner loop (μ in paper)
epsilon = 0.2                # PPO clip range
beta = 0.04                  # KL penalty coefficient

learning_rate = 5e-6         # lower than SFT (5e-5)
max_steps = 1000
val_check_interval = 50
```

### Memory Optimization

- **`generation_batch_size`**: Sub-batch size for generation (default: 4)
  - Lower if OOM during generation
- **`use_learning_mode_for_logprobs`**: Enable gradients through MCMC (default: True)
  - Set to False for faster training with less signal
- **`num_iterations`**: GRPO inner loop iterations (default: 1)
  - Higher values = more updates per batch but slower

## Expected Results

Based on the plan and similar RL work:

| Metric | SFT Baseline | Expected RL Improvement |
|--------|--------------|------------------------|
| Board accuracy (SATNet test) | ~40-50% | +5-15% absolute |
| Cell accuracy | ~70-80% | +3-8% absolute |
| Validity rate | ~30-40% | +10-20% absolute |

## Verification Steps

### 1. Unit Test Log-Probs

```python
from openebm.elm.rl.logprobs import get_per_token_logps
import torch

# Load SFT model
model = ...  # load from checkpoint
input_ids = torch.randint(0, 32768, (2, 100))  # batch of 2, seq len 100
prompt_length = 50

logps = get_per_token_logps(model, input_ids, prompt_length, learning=False)
print(f"Log-probs shape: {logps.shape}")  # should be (2, 50)
print(f"Log-probs range: [{logps.min():.2f}, {logps.max():.2f}]")  # should be negative
assert not torch.isnan(logps).any()
assert not torch.isinf(logps).any()
```

### 2. Unit Test Rewards

```python
from openebm.elm.rl.rewards import compute_sudoku_rewards_detailed

# Perfect solution
result = compute_sudoku_rewards_detailed(
    completions=["5 3 4 6 7 8 9 1 2\n6 7 2 1 9 5 3 4 8\n..."],
    puzzles=["530070000600195000..."],
    solutions=[[[5,3,4,...], ...]],
)
print(result[0])  # should be {'total': 3.0, 'format': 0.5, ...}
```

### 3. Smoke Test Generation

```python
from openebm.elm.rl.rollout import generate_completions

completion_ids, completion_texts, masks = generate_completions(
    model=model,
    prompt_ids=prompt_ids,
    tokenizer=tokenizer,
    hparams=hparams,
    num_generations=4,
)
print(f"Generated {len(completion_texts)} completions")
print(f"First completion: {completion_texts[0][:100]}...")
```

### 4. Short Training Run

```bash
python -m openebm.elm.rl.train_rl_sudoku \
    --sft_checkpoint /path/to/sft.ckpt \
    --max_steps 10 \
    --num_generations 4 \
    --val_check_interval 5
```

Verify:
- Loss decreases or stays stable
- Rewards increase
- KL stays bounded (< 0.5)
- No NaN/Inf in gradients

## Troubleshooting

### OOM during generation

- Reduce `generation_batch_size` (default: 4 → 2 or 1)
- Reduce `num_generations` (default: 8 → 4)
- Reduce `max_completion_length` (default: 512 → 384)

### OOM during training

- Set `use_learning_mode_for_logprobs=False` (disables gradients through MCMC)
- Reduce `num_iterations` (default: 1, already minimal)
- Use gradient checkpointing (requires modifying `modeling_ebt.py`)

### Rewards not increasing

- Check SFT baseline quality (should solve ~40-50% of boards)
- Increase `temperature` (default: 0.9 → 1.0) for more exploration
- Reduce `beta` (default: 0.04 → 0.01) to allow more deviation from SFT
- Increase `epsilon` (default: 0.2 → 0.3) for larger policy updates

### KL divergence exploding

- Increase `beta` (default: 0.04 → 0.08)
- Reduce `learning_rate` (default: 5e-6 → 2e-6)
- Reduce `epsilon` (default: 0.2 → 0.1)

## Design Rationale

### Why not use LoRA?

OpenEBM doesn't use LoRA. We load two full copies of the model (policy + reference). This doubles memory but is simpler. If memory is tight, we can:
- Compute ref log-probs in a separate forward pass
- Offload the ref model to CPU between uses
- Set `beta=0.0` to disable KL (no reference model needed)

### Why PyTorch Lightning?

The RL trainer is a new `LightningModule` (not extending `ModelTrainer`) because:
- The training loop structure is fundamentally different from SFT
- We need custom generation + reward computation logic
- However, we reuse the same model class, tokenizer, and evaluation code

### Why group-relative advantages?

GRPO normalizes advantages within each group (N completions per prompt):
- Reduces variance compared to global normalization
- Makes the algorithm invariant to reward scale
- Works well when reward distributions vary across prompts

## Future Extensions

1. **Adaptive difficulty curriculum**: Start with easy puzzles, gradually increase difficulty
2. **Multi-task RL**: Mix Sudoku with other reasoning tasks
3. **Reward shaping**: Add intermediate rewards for partial progress
4. **Rejection sampling**: Use energy values to filter low-quality completions
5. **Iterative refinement**: Allow the model to revise its solution

## References

- **GRPO paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **d1/diffu-GRPO**: Reference implementation for masked diffusion models
- **EBT paper**: Energy-Based Transformers (OpenEBM)
- **RRN dataset**: Recurrent Relational Networks Sudoku dataset
