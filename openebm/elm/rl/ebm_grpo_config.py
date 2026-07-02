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
    """Legacy automatic-optimization loss recomputation count. Keep at 1 for
    energy_reinforce. Prefer gspo_update_epochs for true rollout reuse."""

    gspo_update_epochs: int = 1
    """True optimizer steps per rollout for energy_gspo, verl-style.
    Values >1 switch the trainer to manual optimization and reuse one rollout
    batch for multiple clipped GSPO updates."""

    epsilon: float = 0.2
    """Fallback symmetric PPO/GSPO clip range."""

    clip_ratio_low: Optional[float] = None
    """Lower GSPO/PPO clip range. If None, uses epsilon.

    For true GSPO mode, verl's validated recipes use much tighter sequence
    ratio clipping (for example 3e-4) than PPO/GRPO token-ratio defaults."""

    clip_ratio_high: Optional[float] = None
    """Upper GSPO/PPO clip range. If None, uses epsilon.

    For true GSPO mode, verl's validated recipes often use 4e-4 paired with
    clip_ratio_low=3e-4."""

    beta: float = 0.0
    """KL penalty coefficient. 0.0 disables KL (no reference model needed).

    For energy_reinforce, this scales a sequence-level energy-KL proxy:
    beta * mean((E_θ - E_ref)^2). For token_logprobs it scales the
    per-token KL via k1 estimator (legacy)."""

    energy_kl_mode: str = "symmetric_huber"
    """Sequence-energy anchor mode against the frozen SFT reference:
    - 'symmetric_huber' (default): smooth two-sided penalty on E_θ - E_ref.
      This is the safest RL default because both positive and negative energy
      drift can indicate policy collapse.
    - 'symmetric_l2': stronger two-sided squared penalty.
    - 'one_sided': legacy relu(E_θ - E_ref), only punishes worse-than-ref
      energy and allows large negative drift."""

    energy_kl_huber_delta: float = 0.5
    """Huber delta for energy_kl_mode='symmetric_huber'."""

    reward_weights: Optional[List[float]] = None
    """Weights for each reward component. None = equal weighting."""

    advantage_norm: str = "group"
    """Advantage normalization mode:
    - 'group' (default, legacy): per-prompt mean/std.
    - 'group_mean_global_std': subtract per-prompt mean, divide by batch-wide std.
      Prevents tiny intra-group std from amplifying noise into huge advantages."""

    global_std_min: float = 0.1
    """Lower clamp for the batch-wide reward std (only used when
    advantage_norm='group_mean_global_std')."""

    skip_degenerate_threshold: float = 0.9
    """If degenerate_group_rate exceeds this in a step, the optimizer step is
    skipped (training_step returns None). Prevents pretending to train while
    every advantage is zero. Set to 1.01 to disable."""

    min_reward_std_to_update: float = 1e-4
    """Skip optimizer update when rollout reward std is below this value.
    This catches all-zero/all-identical reward batches even when the per-rank
    degenerate-group statistic is diluted by DDP aggregation."""

    min_unique_completion_ratio_to_update: float = 0.0
    """Optional collapse guard. When >0, skip updates whose unique completion
    ratio is below this threshold. Keep 0.0 by default so low diversity is
    logged first and used as a manual diagnostic before enforcing the guard."""

    skip_consensus: str = "local"
    """DDP skip consensus mode:
    - 'all': skip only when every rank reports a bad rollout.
    - 'any': skip if any rank reports a bad rollout. More conservative and
      can be too strict with one prompt per rank.
    - 'local': skip the whole step only if every rank is bad; otherwise bad
      ranks contribute a zero full-graph loss while healthy ranks update."""

    # ── Optimization ──────────────────────────────────────────────────────────
    learning_rate: float = 1e-6
    """AdamW base/fallback peak LR for RL fine-tuning.

    With rl_optimizer='adamw', this is the transformer/other base peak LR and
    the per-family LRs are derived from it. With rl_optimizer='muon_adamw',
    this does NOT control transformer matrix parameters; Muon matrices use
    muon_lr. It only controls AdamW lanes unless the adamw_* absolute LR fields
    below are set. All optimizer groups are still scaled by the same warmup/cos
    scheduler, so each group's configured LR is its own peak LR."""

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
    rl_loss_type: str = "energy_gspo"
    """RL loss variant for EBT. Options:
    - 'energy_gspo' (default): Sequence-level energy ratio with PPO clipping.
      compute_sequence_energy returns completion-mean energy, so the ratio is
      exp(-(E_θ - E_θ_old)). First-order, no Hessian.
    - 'energy_reinforce': Pure on-policy, loss = -advantage * (-E_θ).
      Simplest, no importance ratio needed. No PPO stability guarantees.
    - 'token_logprobs': Original token-level logprobs via MCMC chain.
      Requires create_graph=True (Hessian). Known to produce NaN gradients
      with EBT's architecture — kept for research/debugging only.
    """

    use_learning_mode_for_logprobs: bool = True
    """Only used when rl_loss_type='token_logprobs'. Controls create_graph in MCMC."""

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

    # ── Trajectory logging ────────────────────────────────────────────────────
    traj_log_interval: int = 50
    """Print periodic completion dump every N steps (0 = disabled)."""

    traj_output_dir: str = ""
    """Root dir for trajectory JSONL files. Empty = auto-detect from trainer.log_dir."""

    traj_num_samples: int = 2
    """Number of completions to print per periodic dump."""

    collapse_check_window: int = 5
    """Number of consecutive degenerate steps before WARN-COLLAPSE fires."""

    # ── Optimizer ─────────────────────────────────────────────────────────────
    rl_optimizer: str = "adamw"
    """Optimizer kind: 'adamw' (default, v3 P0 multi-group lr) or
    'muon_adamw' (Muon for transformer matrices + AdamW for the rest, mirrors
    the SFT-stage optimizer used by openebm.elm.trainer)."""

    muon_lr: float = 2e-4
    """Learning rate for Muon group (transformer matrices). Only used when
    rl_optimizer='muon_adamw'. Default = SFT muon_lr (2e-3) / 10."""

    adamw_vocab_to_embed_lr: float = -1.0
    """Absolute AdamW LR for vocab_to_embed in muon_adamw mode. <=0 falls back
    to learning_rate * 0.5."""

    adamw_scalar_lr: float = -1.0
    """Absolute AdamW LR for transformer scalar/norm-like params in
    muon_adamw mode. <=0 falls back to learning_rate * 2.0."""

    adamw_other_lr: float = -1.0
    """Absolute AdamW LR for non-transformer catch-all params in muon_adamw
    mode. <=0 falls back to learning_rate."""

    muon_momentum: float = 0.95
    """Muon momentum. Inherits SFT default."""

    muon_ns_steps: int = 5
    """Newton-Schulz iteration count in Muon. Inherits SFT default."""

    muon_beta2: float = 0.95
    """Muon second-moment EMA coefficient. Inherits SFT default."""
