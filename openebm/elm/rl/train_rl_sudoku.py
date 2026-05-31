"""Entry point for EBM-GRPO Sudoku RL training.

Usage:
    python -m openebm.elm.rl.train_rl_sudoku \
        --sft_checkpoint /path/to/sft.ckpt \
        --max_steps 1000 \
        --num_generations 8

Or via torchrun for multi-GPU:
    torchrun --standalone --nproc_per_node=8 \
        -m openebm.elm.rl.train_rl_sudoku \
        --sft_checkpoint /path/to/sft.ckpt
"""

import argparse
import os
import sys

import torch

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.strategies import DDPStrategy

from openebm.elm.rl.ebm_grpo_config import EBMGRPOConfig
from openebm.elm.rl.ebm_grpo_trainer import EBMGRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="EBM-GRPO Sudoku RL Training")

    # Required
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Path to SFT-trained checkpoint")

    # GRPO config overrides
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=224)
    parser.add_argument("--max_prompt_length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--gspo_update_epochs", type=int, default=1,
                        help="True optimizer steps per rollout for energy_gspo. Values >1 enable verl-style rollout reuse.")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--clip_ratio_low", type=float, default=None,
                        help="Lower PPO/GSPO clip range; defaults to epsilon.")
    parser.add_argument("--clip_ratio_high", type=float, default=None,
                        help="Upper PPO/GSPO clip range; defaults to epsilon. verl GSPO recipes often use 0.28.")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--energy_kl_mode", type=str, default="symmetric_huber",
                        choices=["symmetric_huber", "symmetric_l2", "one_sided"],
                        help="Energy anchor mode against frozen SFT reference.")
    parser.add_argument("--energy_kl_huber_delta", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--max_grad_per_param", type=float, default=0.05,
                        help="Per-parameter gradient clipping value (0=disabled)")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--val_check_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--no_augment", action="store_true")

    # Training
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--float_precision", type=str, default="32-true")
    parser.add_argument(
        "--rl_loss_type",
        type=str,
        default="energy_gspo",
        choices=["energy_gspo", "energy_reinforce", "token_logprobs"],
        help="RL loss variant. energy_gspo (default): sequence-level energy ratio. "
             "energy_reinforce: pure on-policy energy loss. "
             "token_logprobs: legacy MCMC logprobs (Hessian-unstable).",
    )
    parser.add_argument(
        "--logp_mcmc_grad",
        type=str,
        default="full",
        choices=["none", "full"],
        help="Only for token_logprobs mode. "
             "'full': create_graph=True; 'none': create_graph=False.",
    )
    parser.add_argument(
        "--advantage_norm",
        type=str,
        default="group_mean_global_std",
        choices=["group", "group_mean_global_std"],
        help="Advantage normalization: per-group std (legacy) or "
             "per-group mean with batch-wide std (more stable).",
    )
    parser.add_argument(
        "--global_std_min",
        type=float,
        default=0.1,
        help="Lower clamp for batch-wide reward std (advantage_norm="
             "group_mean_global_std only).",
    )
    parser.add_argument(
        "--skip_degenerate_threshold",
        type=float,
        default=0.9,
        help="If degenerate_group_rate > threshold, skip the optimizer step. "
             "Set to 1.01 to disable.",
    )
    parser.add_argument("--min_reward_std_to_update", type=float, default=1e-4,
                        help="Skip update when rollout reward std is below this value.")
    parser.add_argument("--min_unique_completion_ratio_to_update", type=float, default=0.0,
                        help="Optional diversity guard; 0 disables update skipping by unique ratio.")
    parser.add_argument(
        "--skip_consensus",
        type=str,
        default="all",
        choices=["all", "any"],
        help="'all': skip only if every DDP rank reports a bad rollout; "
             "'any': skip if any rank reports a bad rollout.",
    )

    # Optimizer
    parser.add_argument("--rl_optimizer", type=str, default="muon_adamw",
                        choices=["adamw", "muon_adamw"],
                        help="adamw (default) or muon_adamw (Muon for transformer matrices).")
    parser.add_argument("--muon_lr", type=float, default=2e-4,
                        help="LR for Muon group (only used when rl_optimizer=muon_adamw).")
    parser.add_argument("--adamw_vocab_to_embed_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for vocab_to_embed in muon_adamw; <=0 uses learning_rate*0.5.")
    parser.add_argument("--adamw_scalar_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for scalar/norm-like params in muon_adamw; <=0 uses learning_rate*2.")
    parser.add_argument("--adamw_other_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for catch-all AdamW params in muon_adamw; <=0 uses learning_rate.")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_beta2", type=float, default=0.95)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="nlp_sudoku_rl")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--traj_log_interval", type=int, default=50,
                        help="Dump rollout samples every N steps (0 = disabled)")
    parser.add_argument("--traj_output_dir", type=str, default="",
                        help="Run/stage root for trajectory outputs; files go under <dir>/trajectories/")
    parser.add_argument("--traj_num_samples", type=int, default=2,
                        help="Number of rollout samples to print per periodic dump")
    parser.add_argument("--collapse_check_window", type=int, default=5,
                        help="Consecutive degenerate rollout windows before collapse dump")

    return parser.parse_args()


def load_sft_model_and_tokenizer(checkpoint_path):
    """Load the SFT-trained model from a Lightning checkpoint.

    Returns:
        model: EBT_NLP instance with loaded weights
        tokenizer: nanochat tokenizer
        hparams: the model's hyperparameters
    """
    from openebm.elm.trainer import ModelTrainer
    from nanochat.tokenizer import get_tokenizer

    print(f"Loading SFT checkpoint: {checkpoint_path}")

    # Load the full Lightning checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]

    # Get tokenizer
    tokenizer = get_tokenizer()
    hparams["tokenizer_obj"] = tokenizer

    # Backfill VE attributes for SFT ckpts saved before commit 2fe9418
    # ("Add VE support"), which made `use_ve` / `vocab_size` mandatory.
    if "use_ve" not in hparams:
        hparams["use_ve"] = False
    if "vocab_size" not in hparams:
        for attr in ("get_vocab_size", "vocab_size", "n_vocab"):
            obj = getattr(tokenizer, attr, None)
            if obj is None:
                continue
            hparams["vocab_size"] = obj() if callable(obj) else int(obj)
            break
        else:
            hparams["vocab_size"] = 32768   # nanochat default
        print(f"  Backfilled vocab_size={hparams['vocab_size']} (ckpt missing the field)")

    # Create model
    from openebm.elm.modeling_ebt import EBT_NLP

    class _HP:
        pass

    hp = _HP()
    for k, v in hparams.items():
        setattr(hp, k, v)

    model = EBT_NLP(hp)

    # Load state dict (handle "model." from Lightning + "_orig_mod." from torch.compile)
    state_dict = ckpt["state_dict"]
    model_state = {}
    n_renamed_eager = 0
    for key, val in state_dict.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]
        # Compat: SFT ckpts trained with use_sdpa_attention=True (older runs)
        # store transformer params under `transformer_eager.*`. Current
        # EBT_NLP build always names it `self.transformer`.
        if clean_key.startswith("transformer_eager."):
            clean_key = "transformer." + clean_key[len("transformer_eager."):]
            n_renamed_eager += 1
        if clean_key in model_state:
            continue
        model_state[clean_key] = val
    if n_renamed_eager > 0:
        print(f"  Renamed {n_renamed_eager} `transformer_eager.*` keys to `transformer.*`")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (ignored): {unexpected[:5]}...")
    if missing:
        # Hard-fail: missing keys mean part of the model is randomly initialised,
        # so any RL rollout produces gibberish and training is meaningless.
        sample = missing[:10]
        raise RuntimeError(
            f"SFT checkpoint loaded with {len(missing)} missing keys — these "
            f"parameters would remain randomly initialised, making RL rollouts "
            f"produce gibberish. Refusing to continue.\n"
            f"  ckpt: {checkpoint_path}\n"
            f"  first {len(sample)} missing keys: {sample}\n"
            f"Likely causes: (a) ckpt uses a key naming this loader doesn't "
            f"handle yet — extend the prefix-rewrite block above; (b) model "
            f"hparams (use_sdpa_attention, ebt_type, use_ve, etc.) don't match "
            f"the ckpt's training-time architecture."
        )
    print(f"  Model loaded successfully. Params: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, hparams


def main():
    args = parse_args()

    # Build config
    config = EBMGRPOConfig(
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_batch_size=args.generation_batch_size,
        num_iterations=args.num_iterations,
        gspo_update_epochs=args.gspo_update_epochs,
        epsilon=args.epsilon,
        clip_ratio_low=args.clip_ratio_low,
        clip_ratio_high=args.clip_ratio_high,
        beta=args.beta,
        energy_kl_mode=args.energy_kl_mode,
        energy_kl_huber_delta=args.energy_kl_huber_delta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        max_grad_per_param=args.max_grad_per_param,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        log_interval=args.log_interval,
        save_top_k=args.save_top_k,
        seed=args.seed,
        sft_checkpoint_path=args.sft_checkpoint,
        rl_loss_type=args.rl_loss_type,
        use_learning_mode_for_logprobs=(args.logp_mcmc_grad == "full"),
        advantage_norm=args.advantage_norm,
        global_std_min=args.global_std_min,
        skip_degenerate_threshold=args.skip_degenerate_threshold,
        min_reward_std_to_update=args.min_reward_std_to_update,
        min_unique_completion_ratio_to_update=args.min_unique_completion_ratio_to_update,
        skip_consensus=args.skip_consensus,
        rl_optimizer=args.rl_optimizer,
        muon_lr=args.muon_lr,
        adamw_vocab_to_embed_lr=args.adamw_vocab_to_embed_lr,
        adamw_scalar_lr=args.adamw_scalar_lr,
        adamw_other_lr=args.adamw_other_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_beta2=args.muon_beta2,
        data_dir=args.data_dir,
        augment=not args.no_augment,
        num_gpus=args.num_gpus,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        run_name=args.run_name,
        traj_log_interval=args.traj_log_interval,
        traj_output_dir=args.traj_output_dir,
        traj_num_samples=args.traj_num_samples,
        collapse_check_window=args.collapse_check_window,
    )

    # Seed
    seed_everything(config.seed)

    # Load model
    model, tokenizer, model_hparams = load_sft_model_and_tokenizer(args.sft_checkpoint)

    # Create GRPO trainer module
    grpo_module = EBMGRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        ckpt_dir = os.path.join("checkpoints", config.run_name or "ebm_grpo_sudoku")

    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="rl-step={step}-reward={val/reward_mean:.3f}",
            monitor="val/reward_mean",
            mode="max",
            save_top_k=config.save_top_k,
            every_n_train_steps=config.val_check_interval,
        )
    )

    # Determine GPUs
    if config.num_gpus == -1:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        gpus = config.num_gpus

    # Strategy
    strategy = "auto"
    if gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    # W&B logger
    logger = None
    try:
        from lightning.pytorch.loggers import WandbLogger
        if config.wandb_mode != "disabled":
            logger = WandbLogger(
                project=config.wandb_project,
                name=config.run_name or None,
                mode=config.wandb_mode,
            )
    except ImportError:
        pass

    # Lightning Trainer
    trainer = Trainer(
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        gradient_clip_val=None,  # per-param clipping in on_before_optimizer_step suffices
        precision=args.float_precision,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_interval,
        enable_progress_bar=True,
    )

    # Train
    print("\n" + "=" * 80)
    print("  EBM-GRPO Sudoku RL Training")
    print("=" * 80)
    print(f"  SFT checkpoint:     {args.sft_checkpoint}")
    print(f"  Num generations:    {config.num_generations}")
    print(f"  Max completion len: {config.max_completion_length}")
    print(f"  Temperature:        {config.temperature}")
    print(f"  Epsilon (clip):     {config.epsilon}")
    print(f"  Beta (KL):          {config.beta}")
    print(f"  Energy KL mode:     {config.energy_kl_mode}")
    print(f"  Optimizer:          {config.rl_optimizer}")
    print(f"  AdamW base LR:      {config.learning_rate}")
    if config.rl_optimizer == "muon_adamw":
        print(f"  Muon LR:            {config.muon_lr}")
        print(f"  AdamW v2e LR:       {config.adamw_vocab_to_embed_lr if config.adamw_vocab_to_embed_lr > 0 else config.learning_rate * 0.5}")
        print(f"  AdamW scalar LR:    {config.adamw_scalar_lr if config.adamw_scalar_lr > 0 else config.learning_rate * 2.0}")
        print(f"  AdamW other LR:     {config.adamw_other_lr if config.adamw_other_lr > 0 else config.learning_rate}")
    print(f"  Skip consensus:     {config.skip_consensus}")
    print(f"  Max steps:          {config.max_steps}")
    print(f"  GPUs:               {gpus}")
    print(f"  Precision:          {args.float_precision}")
    print(f"  Trajectory dir:     {config.traj_output_dir or os.environ.get('TRAJ_OUTPUT_DIR') or '<trainer.log_dir>'}/trajectories")
    print("=" * 80 + "\n")

    trainer.fit(grpo_module)


if __name__ == "__main__":
    main()
