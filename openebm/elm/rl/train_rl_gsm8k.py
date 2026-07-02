"""Entry point for EBM-GRPO GSM8K RL training.

Usage:
    torchrun --standalone --nproc_per_node=6 \
        -m openebm.elm.rl.train_rl_gsm8k \
        --sft_checkpoint /path/to/sft.ckpt \
        --data_path /path/to/gsm8k.jsonl \
        --max_steps 1000
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

from torch.utils.data import DataLoader

from openebm.elm.rl.ebm_grpo_config import EBMGRPOConfig
from openebm.elm.rl.ebm_grpo_trainer import EBMGRPOTrainer
from openebm.elm.rl.gsm8k_dataset_rl import GSM8KRLPromptDataset, collate_gsm8k_prompts
from openebm.elm.rl.gsm8k_rewards import compute_gsm8k_rewards_detailed
from openebm.elm.rl.trajectory_logger import TrajLoggerConfig


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K-specific trainer subclass
# ─────────────────────────────────────────────────────────────────────────────

class EBMGRPOTrainerGSM8K(EBMGRPOTrainer):
    """GRPO trainer adapted for GSM8K math reasoning task.

    Replaces:
      - Dataset: SudokuRLPromptDataset → GSM8KRLPromptDataset
      - Reward:  compute_sudoku_rewards_detailed → compute_gsm8k_rewards_detailed
      - Reward components logged: format / exact_match / answer_proximity / partial_credit
    """

    def __init__(self, config: EBMGRPOConfig, model, tokenizer,
                 data_path: str, val_data_path: str = "",
                 ref_model=None):
        super().__init__(config=config, model=model, tokenizer=tokenizer,
                         ref_model=ref_model)
        self._gsm8k_data_path = data_path
        self._gsm8k_val_data_path = val_data_path or data_path  # fallback

    # ── Reward computation (override) ─────────────────────────────────────────

    def _compute_rewards(self, completion_texts, batch_meta):
        """GSM8K reward: exact_match + proximity + format + partial."""
        answers = batch_meta["answers"]
        # Expand answers to match num_generations
        num_prompts = len(answers)
        expanded_answers = []
        for i in range(num_prompts):
            for _ in range(self.config.num_generations):
                expanded_answers.append(answers[i])
        return compute_gsm8k_rewards_detailed(completion_texts, expanded_answers)

    # ── Override _generate_and_score to use GSM8K reward ─────────────────────

    @torch.no_grad()
    def _generate_and_score(self, batch):
        """Same as parent but calls GSM8K reward instead of sudoku reward."""
        from openebm.elm.rl.rollout import generate_completions
        from openebm.elm.rl.logprobs import compute_sequence_energy, get_per_token_logps
        import torch

        prompt_ids = batch["prompt_ids"].to(self.device)
        puzzles = batch.get("puzzles", [])   # not used for GSM8K but keep interface
        num_prompts = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # ── 1. Generate completions ───────────────────────────────────────────
        self.model.eval()
        self.model._dbg_logged_embed_stats = False
        self.model._dbg_logged_logps_stats = False
        with torch.amp.autocast('cuda', enabled=False):
            completion_ids, completion_texts, completion_masks = generate_completions(
                model=self.model,
                prompt_ids=prompt_ids,
                tokenizer=self.tokenizer,
                hparams=self._gen_hparams,
                num_generations=self.config.num_generations,
                max_completion_length=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                generation_batch_size=self.config.generation_batch_size,
            )
        self.model.train()
        comp_len = completion_ids.shape[1]

        # ── 2. Compute rewards (GSM8K) ────────────────────────────────────────
        expanded_answers = [
            batch["answers"][i // self.config.num_generations]
            for i in range(len(completion_texts))
        ]
        reward_details = compute_gsm8k_rewards_detailed(completion_texts, expanded_answers)
        rewards = torch.tensor(
            [d["total"] for d in reward_details],
            dtype=torch.float32, device=self.device,
        )

        # Debug first rollout
        if not getattr(self, '_dbg_logged_first_rollout', False):
            self._dbg_logged_first_rollout = True
            print(f"[DBG-RL] first GSM8K rollout: {repr(completion_texts[0][:300])}", flush=True)
            print(f"[DBG-RL] first GSM8K reward detail: {reward_details[0]}", flush=True)

        # ── 3. Compute energies / logprobs ────────────────────────────────────
        expanded_prompts = prompt_ids.repeat_interleave(self.config.num_generations, dim=0)
        full_ids = torch.cat([expanded_prompts, completion_ids], dim=1)

        # P0b: keep autocast disabled to match _loss_energy_*'s autocast(False)
        # block (avoids ratio drift between old/current energies).
        with torch.amp.autocast('cuda', enabled=False):
            old_energies = compute_sequence_energy(
                self.model, full_ids, prompt_len, completion_masks
            )

        old_per_token_logps = None
        if self.config.rl_loss_type == "token_logprobs":
            with torch.amp.autocast('cuda', enabled=False):
                old_per_token_logps, _ = get_per_token_logps(
                    self.model, full_ids, prompt_len, learning=False)
            old_per_token_logps = old_per_token_logps[:, :comp_len] * completion_masks.float()

        ref_per_token_logps = None
        if (self.config.rl_loss_type == "token_logprobs"
                and self.ref_model is not None and self.config.beta > 0.0):
            with torch.amp.autocast('cuda', enabled=False):
                ref_per_token_logps, _ = get_per_token_logps(
                    self.ref_model, full_ids, prompt_len, learning=False)
            ref_per_token_logps = ref_per_token_logps[:, :comp_len] * completion_masks.float()

        # ── 4. Advantages (group-relative) ────────────────────────────────────
        grouped_rewards = rewards.view(num_prompts, self.config.num_generations)
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        group_std_raw = grouped_rewards.std(dim=1, keepdim=True)
        degenerate = (group_std_raw.squeeze(-1) < 1e-6)
        if getattr(self.config, "advantage_norm", "group") == "group_mean_global_std":
            global_std = grouped_rewards.flatten().std().clamp(
                min=self.config.global_std_min)
            advantages = ((grouped_rewards - group_mean) / global_std).view(-1)
        else:
            group_std_safe = group_std_raw.clamp(min=1e-4)
            advantages = ((grouped_rewards - group_mean) / group_std_safe).view(-1)
        deg_mask = degenerate.repeat_interleave(self.config.num_generations)
        advantages = torch.where(deg_mask, torch.zeros_like(advantages), advantages)
        advantages = advantages.clamp(-5.0, 5.0)
        degenerate_rate = degenerate.float().mean().item()

        avg_comp_len = completion_masks.sum(dim=1).float().mean().item()
        reward_components = {}
        for key in ["format", "exact_match", "answer_proximity", "partial_credit", "length_penalty"]:
            vals = [d.get(key, 0.0) for d in reward_details]
            reward_components[key] = sum(vals) / max(len(vals), 1)
        parsed_vals = [1.0 if d.get("parsed_answer") is not None else 0.0 for d in reward_details]
        correct_vals = [1.0 if d.get("is_correct") else 0.0 for d in reward_details]
        reward_components["parse_rate"] = sum(parsed_vals) / max(len(parsed_vals), 1)
        reward_components["answer_acc"] = sum(correct_vals) / max(len(correct_vals), 1)

        advantage_var = advantages.var(unbiased=False).item() if advantages.numel() > 1 else 0.0
        reward_var = float(rewards.var(unbiased=False).item()) if rewards.numel() > 1 else 0.0

        # Parent training_step handles trajectory logging and collapse detection
        # using the raw arrays returned below. Keeping it in one place avoids
        # duplicate dumps and keeps Sudoku/GSM8K diagnostics comparable.
        return {
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "completion_masks": completion_masks,
            "completion_texts": completion_texts,
            "old_energies": old_energies.detach(),
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "reward_zero_frac": (rewards == 0).float().mean().item(),
            "reward_var": reward_var,
            "advantage_var": advantage_var,
            "avg_completion_length": avg_comp_len,
            "response_len_mean": avg_comp_len,
            "response_len_max": completion_masks.sum(dim=1).float().max().item(),
            "reward_components": reward_components,
            "degenerate_rate": degenerate_rate,
            # Raw arrays carried through for parent trajectory logging and collapse detection.
            "_completion_texts": completion_texts,
            "_rewards": rewards.detach().cpu().tolist(),
            "_old_energies": old_energies.detach().cpu().tolist(),
            "_advantages": advantages.detach().cpu().tolist(),
            "_completion_masks_lens": completion_masks.sum(dim=1).long().cpu().tolist(),
            "_puzzles": [batch["questions"][i // self.config.num_generations] for i in range(len(completion_texts))],
            "_solutions": expanded_answers,
            "_prompt_ids_per_seq": expanded_prompts.cpu().tolist(),
        }

    # ── Dataloaders (override) ────────────────────────────────────────────────

    def train_dataloader(self):
        dataset = GSM8KRLPromptDataset(
            tokenizer=self.tokenizer,
            data_path=self._gsm8k_data_path,
            max_prompt_length=self.config.max_prompt_length,
            split="train",
            seed=self.config.seed,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_gsm8k_prompts(
                batch, self.tokenizer, self.config.max_prompt_length
            ),
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = GSM8KRLPromptDataset(
            tokenizer=self.tokenizer,
            data_path=self._gsm8k_val_data_path,
            max_prompt_length=self.config.max_prompt_length,
            split="val",
            seed=self.config.seed + 1000,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_gsm8k_prompts(
                batch, self.tokenizer, self.config.max_prompt_length
            ),
            num_workers=1,
            pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        gen_data = self._generate_and_score(batch)
        self.log("val/reward_mean", gen_data["reward_mean"], prog_bar=True, sync_dist=True)
        self.log("val/reward_std", gen_data["reward_std"], sync_dist=True)
        self.log("val/completion_length", gen_data["avg_completion_length"], sync_dist=True)
        for key, val in gen_data.get("reward_components", {}).items():
            self.log(f"val/reward_{key}", val, sync_dist=True)
        return gen_data["reward_mean"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="EBM-GRPO GSM8K RL Training")

    parser.add_argument("--sft_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to GSM8K train JSONL file")
    parser.add_argument("--val_data_path", type=str, default="",
                        help="Path to GSM8K val JSONL (defaults to data_path)")

    # GRPO config
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=320)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.92)
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
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip_val", type=float, default=0.3)
    parser.add_argument("--max_grad_per_param", type=float, default=0.02)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--val_check_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rl_loss_type", type=str, default="energy_gspo",
                        choices=["energy_gspo", "energy_reinforce", "token_logprobs"])
    parser.add_argument("--advantage_norm", type=str, default="group_mean_global_std")
    parser.add_argument("--global_std_min", type=float, default=0.1)
    parser.add_argument("--skip_degenerate_threshold", type=float, default=0.9)
    parser.add_argument("--min_reward_std_to_update", type=float, default=1e-4,
                        help="Skip update when rollout reward std is below this value.")
    parser.add_argument("--min_unique_completion_ratio_to_update", type=float, default=0.0,
                        help="Optional diversity guard; 0 disables update skipping by unique ratio.")

    # Optimizer
    parser.add_argument("--rl_optimizer", type=str, default="muon_adamw",
                        choices=["adamw", "muon_adamw"])
    parser.add_argument("--muon_lr", type=float, default=2e-4)
    parser.add_argument("--adamw_vocab_to_embed_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for vocab_to_embed in muon_adamw; <=0 uses learning_rate*0.5.")
    parser.add_argument("--adamw_scalar_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for scalar/norm-like params in muon_adamw; <=0 uses learning_rate*2.")
    parser.add_argument("--adamw_other_lr", type=float, default=-1.0,
                        help="Absolute AdamW LR for catch-all AdamW params in muon_adamw; <=0 uses learning_rate.")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_beta2", type=float, default=0.95)

    # Logging / trajectory
    parser.add_argument("--traj_log_interval", type=int, default=50,
                        help="Print periodic completion every N steps")
    parser.add_argument("--traj_output_dir", type=str, default="",
                        help="Directory for trajectory JSONL files")
    parser.add_argument("--traj_num_samples", type=int, default=2)
    parser.add_argument("--collapse_check_window", type=int, default=5)

    # Training
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--float_precision", type=str, default="32-true")
    parser.add_argument("--wandb_project", type=str, default="nlp_gsm8k_rl")
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")

    return parser.parse_args()


def load_sft_model_and_tokenizer(checkpoint_path):
    """Reuse the same loading logic as sudoku RL entry point."""
    from nanochat.tokenizer import get_tokenizer
    from openebm.elm.modeling_ebt import EBT_NLP

    print(f"[INFO] Loading SFT checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]

    tokenizer = get_tokenizer()
    hparams["tokenizer_obj"] = tokenizer

    # Backfill VE attributes that may be missing from older SFT checkpoints.
    # commit 2fe9418 ("Add VE support") added `use_ve` / `vocab_size` to hparams,
    # but ckpts saved before that commit don't have them. Default use_ve=False
    # (no value-embedding) and read vocab size from the tokenizer.
    if "use_ve" not in hparams:
        hparams["use_ve"] = False
    if "vocab_size" not in hparams:
        # nanochat tokenizer exposes vocab size via get_vocab_size() or n_vocab
        for attr in ("get_vocab_size", "vocab_size", "n_vocab"):
            obj = getattr(tokenizer, attr, None)
            if obj is None:
                continue
            hparams["vocab_size"] = obj() if callable(obj) else int(obj)
            break
        else:
            hparams["vocab_size"] = 32768   # nanochat default
        print(f"[INFO] Backfilled vocab_size={hparams['vocab_size']} (ckpt missing the field)")

    class _HP:
        pass

    hp = _HP()
    for k, v in hparams.items():
        setattr(hp, k, v)

    model = EBT_NLP(hp)

    state_dict = ckpt["state_dict"]
    model_state = {}
    n_renamed_transformer_keys = 0
    for key, val in state_dict.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]
        if clean_key.startswith("transformer._orig_mod."):
            clean_key = "transformer." + clean_key[len("transformer._orig_mod."):]
            n_renamed_transformer_keys += 1
        # SFT ckpts trained with use_sdpa_attention=True (older luyudong runs)
        # store transformer params under `transformer_eager.*` while the
        # current EBT_NLP build always names it `self.transformer`. Rename
        # to keep the loader compatible across both naming conventions.
        if clean_key.startswith("transformer_eager."):
            clean_key = "transformer." + clean_key[len("transformer_eager."):]
            n_renamed_transformer_keys += 1
        # Some ckpts duplicate weights in BOTH transformer.* and
        # transformer_eager.*; in that case we keep the first occurrence.
        if clean_key in model_state:
            continue
        model_state[clean_key] = val
    if n_renamed_transformer_keys > 0:
        print(f"  [INFO] Renamed {n_renamed_transformer_keys} compiled/eager transformer keys to `transformer.*`")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        critical_prefixes = ("transformer.", "tf_head.", "vocab_to_embed.", "embeddings.", "alpha")
        critical_unexpected = [key for key in unexpected if key.startswith(critical_prefixes)]
        if critical_unexpected:
            sample = critical_unexpected[:10]
            raise RuntimeError(
                f"SFT checkpoint contains {len(critical_unexpected)} unexpected trainable keys "
                f"that would be ignored: {sample}. Refusing to continue with a partially "
                f"loaded model. Check architecture hparams and key rewriting."
            )
        print(f"  [WARN] {len(unexpected)} unexpected keys (ignored): {unexpected[:3]}...")
    if missing:
        # Hard-fail: missing keys mean part of the model is randomly initialised,
        # so any RL rollout would produce gibberish and training is meaningless.
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
    print(f"  [INFO] Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer, hparams


def main():
    args = parse_args()

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
        advantage_norm=args.advantage_norm,
        global_std_min=args.global_std_min,
        skip_degenerate_threshold=args.skip_degenerate_threshold,
        min_reward_std_to_update=args.min_reward_std_to_update,
        min_unique_completion_ratio_to_update=args.min_unique_completion_ratio_to_update,
        rl_optimizer=args.rl_optimizer,
        muon_lr=args.muon_lr,
        adamw_vocab_to_embed_lr=args.adamw_vocab_to_embed_lr,
        adamw_scalar_lr=args.adamw_scalar_lr,
        adamw_other_lr=args.adamw_other_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_beta2=args.muon_beta2,
        augment=False,           # no augmentation for GSM8K
        num_gpus=args.num_gpus,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        run_name=args.run_name,
        # trajectory logging
        traj_log_interval=args.traj_log_interval,
        traj_output_dir=args.traj_output_dir,
        traj_num_samples=args.traj_num_samples,
        collapse_check_window=args.collapse_check_window,
    )

    seed_everything(config.seed)
    model, tokenizer, _ = load_sft_model_and_tokenizer(args.sft_checkpoint)

    grpo_module = EBMGRPOTrainerGSM8K(
        config=config,
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
    )

    callbacks = [LearningRateMonitor(logging_interval="step")]
    ckpt_dir = args.checkpoint_dir or os.path.join("checkpoints", config.run_name or "ebm_grpo_gsm8k")
    callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="gsm8k-rl-step={step}-reward={val/reward_mean:.3f}",
        monitor="val/reward_mean",
        mode="max",
        save_top_k=config.save_top_k,
        every_n_train_steps=config.val_check_interval,
    ))

    gpus = torch.cuda.device_count() if config.num_gpus == -1 else config.num_gpus
    strategy = DDPStrategy(find_unused_parameters=True) if gpus > 1 else "auto"

    logger = None
    try:
        from lightning.pytorch.loggers import WandbLogger
        if config.wandb_mode != "disabled":
            logger = WandbLogger(project=config.wandb_project,
                                 name=config.run_name or None,
                                 mode=config.wandb_mode)
    except ImportError:
        pass

    trainer = Trainer(
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        gradient_clip_val=None,
        precision=args.float_precision,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_interval,
        enable_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("  EBM-GRPO GSM8K RL Training")
    print("=" * 80)
    print(f"  SFT checkpoint:     {args.sft_checkpoint}")
    print(f"  Data path:          {args.data_path}")
    print(f"  Num generations:    {config.num_generations}")
    print(f"  Max completion len: {config.max_completion_length}")
    print(f"  Temperature:        {config.temperature}")
    print(f"  Beta (KL anchor):   {config.beta}")
    print(f"  Energy KL mode:     {config.energy_kl_mode}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Max steps:          {config.max_steps}")
    print(f"  GPUs:               {gpus}")
    print(f"  Precision:          {args.float_precision}")
    print("=" * 80 + "\n")

    trainer.fit(grpo_module)


if __name__ == "__main__":
    main()
