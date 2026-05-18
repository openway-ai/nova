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
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--max_grad_per_param", type=float, default=0.1,
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
    parser.add_argument("--float_precision", type=str, default="bf16-mixed")
    parser.add_argument("--use_learning_mode", action="store_true", default=False,
                        help="Use learning=True for log-prob computation (gradients through MCMC)")
    parser.add_argument("--no_learning_mode", action="store_true", default=True,
                        help="Disable learning mode for log-probs (faster, less signal)")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="nlp_sudoku_rl")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")

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

    # Create model
    from openebm.elm.modeling_ebt import EBT_NLP

    class _HP:
        pass

    hp = _HP()
    for k, v in hparams.items():
        setattr(hp, k, v)

    model = EBT_NLP(hp)

    # Load state dict (handle the "model." prefix from Lightning)
    state_dict = ckpt["state_dict"]
    model_state = {}
    for key, val in state_dict.items():
        if key.startswith("model."):
            model_state[key[6:]] = val  # strip "model." prefix
        else:
            model_state[key] = val

    model.load_state_dict(model_state, strict=False)
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
        epsilon=args.epsilon,
        beta=args.beta,
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
        use_learning_mode_for_logprobs=not args.no_learning_mode,
        data_dir=args.data_dir,
        augment=not args.no_augment,
        num_gpus=args.num_gpus,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        run_name=args.run_name,
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
        gradient_clip_val=config.gradient_clip_val,
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
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Max steps:          {config.max_steps}")
    print(f"  GPUs:               {gpus}")
    print(f"  Precision:          {args.float_precision}")
    print("=" * 80 + "\n")

    trainer.fit(grpo_module)


if __name__ == "__main__":
    main()
