"""EBM-GRPO configuration dataclass.

Holds all RL-specific hyperparameters for Group Relative Policy Optimization
applied to Energy-Based Transformers.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EBMGRPOConfig:
    """Configuration for EBM-GRPO training."""

    # ── Generation / Rollout ──────────────────────────────────────────────────
    num_generations: int = 8
    """Number of completions per prompt for advantage estimation."""

    max_completion_length: int = 224
    """Max tokens to generate per completion (9x9 sudoku needs ~200-400)."""

    max_prompt_length: int = 192
    """Max prompt length (truncate left if exceeded)."""

    temperature: float = 0.9
    """Sampling temperature for generation."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    generation_batch_size: int = 8
    """Sub-batch size for generation (manages VRAM)."""

    # ── GRPO Algorithm ────────────────────────────────────────────────────────
    num_iterations: int = 1
    """GRPO inner loop iterations per batch (μ in paper). Set >1 for multi-step updates."""

    epsilon: float = 0.2
    """PPO clip range."""

    beta: float = 0.0
    """KL penalty coefficient. 0.0 disables KL (no reference model needed)."""

    reward_weights: Optional[List[float]] = None
    """Weights for each reward component. None = equal weighting."""

    # ── Optimization ──────────────────────────────────────────────────────────
    learning_rate: float = 1e-6
    """Peak learning rate for RL fine-tuning. Lower than SFT due to second-order
    gradients through the MCMC chain (Hessian-vector products amplified by alpha)."""

    weight_decay: float = 0.01
    """AdamW weight decay."""

    gradient_clip_val: float = 1.0
    """Gradient clipping norm (global)."""

    max_grad_per_param: float = 0.0
    """Per-parameter gradient clipping (0.0 = disabled). When > 0, each parameter's
    gradient is independently clipped to [-val, val]. More effective than global norm
    for EBT where alpha/transformer have vastly different gradient scales."""

    warmup_steps: int = 20
    """Linear warmup steps."""

    # ── Training Schedule ─────────────────────────────────────────────────────
    max_steps: int = 1000
    """Total RL training steps."""

    val_check_interval: int = 50
    """Validate every N steps."""

    log_interval: int = 10
    """Log metrics every N steps."""

    save_top_k: int = 3
    """Keep top-k checkpoints by reward."""

    seed: int = 42
    """Random seed."""

    # ── Model / Checkpoint ────────────────────────────────────────────────────
    sft_checkpoint_path: str = ""
    """Path to the SFT-trained checkpoint to initialize from."""

    # ── EBT-specific ──────────────────────────────────────────────────────────
    use_learning_mode_for_logprobs: bool = False
    """MCMC gradient mode for current_logps.

    False (Branch-A, default, d1-isomorphic): current_logps uses learning=False
      → MCMC chain has NO 2nd-order graph; alpha is frozen; only the last MCMC
      step's transformer params are differentiable. Numerator and denominator
      of the PPO ratio come from the same estimator → exp(new-old) well-defined.

    True (Branch-B): current_logps uses learning=True → create_graph=True in
      `_mcmc_step_excluded`; alpha receives gradient through the full MCMC
      chain. Numerically fragile under bf16; use only if Branch-A doesn't learn.

    Earlier comments claimed learning=False causes DDP hangs — that was only
    true when logprobs.py wrapped the forward in `set_grad_enabled(learning)`.
    After fixing logprobs.py to always wrap with `set_grad_enabled(True)`
    (Step 1.1), the last MCMC step's transformer params stay in the graph and
    cross_entropy → current_logps still has requires_grad=True, so all-reduce
    hooks fire correctly."""

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = ""
    """Sudoku data directory (defaults to SUDOKU_DATA_DIR_V2 env var)."""

    data_split: str = "train"
    """Which split to use for RL prompts."""

    augment: bool = True
    """Apply symmetry augmentation to puzzles."""

    # ── Distributed ───────────────────────────────────────────────────────────
    num_gpus: int = -1
    """Number of GPUs (-1 = all available)."""

    # ── Logging ───────────────────────────────────────────────────────────────
    wandb_project: str = "nlp_sudoku_rl"
    """W&B project name."""

    wandb_mode: str = "offline"
    """W&B mode (online/offline/disabled)."""

    run_name: str = ""
    """Run name for logging."""
