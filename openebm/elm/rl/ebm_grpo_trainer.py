"""EBM-GRPO Trainer: Group Relative Policy Optimization for Energy-Based Transformers.

Implements the GRPO algorithm adapted for EBT:
  1. Generate N completions per prompt (rollout)
  2. Compute multi-component rewards
  3. Compute group-relative advantages
  4. For K iterations: PPO-clipped policy gradient + KL penalty

Key difference from d1's diffu-GRPO:
  - d1 uses mask-then-predict for log-probs (masked diffusion specific)
  - EBT uses standard next-token prediction through MCMC refinement chain
"""

import copy
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, Callback
except ImportError:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from openebm.elm.rl.ebm_grpo_config import EBMGRPOConfig
from openebm.elm.rl.logprobs import get_per_token_logps
from openebm.elm.rl.rewards import compute_sudoku_rewards, compute_sudoku_rewards_detailed
from openebm.elm.rl.rollout import generate_completions
from openebm.elm.rl.sudoku_dataset_rl import SudokuRLPromptDataset, collate_rl_prompts


class EBMGRPOTrainer(LightningModule):
    """GRPO trainer for EBT models on Sudoku tasks."""

    def __init__(self, config: EBMGRPOConfig, model, tokenizer, ref_model=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["model", "ref_model", "tokenizer"])

        # Policy model (trainable)
        self.model = model

        # RL stability: only preserve the 2nd-order graph for the LAST MCMC step.
        # With truncate_mcmc=False (SFT default), ALL steps use create_graph=True
        # when learning=True, which is expensive and numerically fragile under bf16.
        # truncate_mcmc=True limits the 2nd-order graph to the final step only —
        # sufficient gradient signal with much better stability.
        if hasattr(self.model, 'hparams'):
            if isinstance(self.model.hparams, dict):
                self.model.hparams['truncate_mcmc'] = True
            else:
                self.model.hparams.truncate_mcmc = True

        # Freeze MCMC step-size (alpha) and Langevin noise std during RL.
        # These were calibrated during SFT; the 2nd-order gradients through
        # the MCMC chain are numerically fragile and corrupt alpha via NaN
        # on the very first optimizer step (confirmed in v6 debug logs).
        if hasattr(self.model, 'alpha'):
            self.model.alpha.requires_grad_(False)
        if hasattr(self.model, 'langevin_dynamics_noise_std'):
            self.model.langevin_dynamics_noise_std.requires_grad_(False)

        self.tokenizer = tokenizer

        # Build hparams namespace for generate.py compatibility
        self._gen_hparams = self._build_gen_hparams()

        # Metrics tracking
        self._step_metrics = {}

        # Reference model (frozen) — for KL penalty
        if config.beta > 0.0:
            if ref_model is not None:
                self.ref_model = ref_model
            else:
                # Don't use copy.deepcopy on EBT — it has nn.Parameter alpha,
                # buffers, and may have compiled-state artifacts that don't
                # round-trip cleanly. Reconstruct from hparams and load state
                # dict instead.
                from openebm.elm.modeling_ebt import EBT_NLP
                self.ref_model = EBT_NLP(model.hparams)
                self.ref_model.load_state_dict(model.state_dict())
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        else:
            self.ref_model = None

    def _build_gen_hparams(self):
        """Build a hparams-like namespace for generate.py functions."""

        class _HParams:
            pass

        hp = _HParams()
        hp.model_name = "ebt"
        hp.infer_ebt_advanced = False
        hp.context_length = getattr(self.model.hparams, 'context_length', 2048)
        hp.tokenizer_obj = self.tokenizer
        return hp

    # ══════════════════════════════════════════════════════════════════════════
    # Core GRPO Logic
    # ══════════════════════════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        """One GRPO training step.

        Outer: generate completions + compute rewards + advantages (no grad)
        Inner: num_iterations of PPO-clipped updates (with grad)
        """
        # ── Phase 1: Generate and Score (no grad) ─────────────────────────────
        gen_data = self._generate_and_score(batch)

        # ── Phase 2: Policy Update (with grad) ───────────────────────────────
        total_loss = torch.tensor(0.0, device=self.device)
        total_metrics = {
            "policy_loss": 0.0,
            "kl": 0.0,
            "clip_ratio": 0.0,
        }

        for iteration in range(self.config.num_iterations):
            loss, metrics = self._compute_grpo_loss(gen_data, iteration)
            total_loss = total_loss + loss / self.config.num_iterations
            for k, v in metrics.items():
                total_metrics[k] += v / self.config.num_iterations

        # ── Logging ───────────────────────────────────────────────────────────
        self.log("train/loss", total_loss.item(), prog_bar=True)
        self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
        self.log("train/reward_std", gen_data["reward_std"])
        self.log("train/policy_loss", total_metrics["policy_loss"])
        self.log("train/kl", total_metrics["kl"])
        self.log("train/clip_ratio", total_metrics["clip_ratio"])
        self.log("train/completion_length", gen_data["avg_completion_length"])

        # Log reward components
        for key, val in gen_data.get("reward_components", {}).items():
            self.log(f"train/reward_{key}", val)

        # ── Stability diagnostics ────────────────────────────────────────────
        self.log("stability/degenerate_group_rate", gen_data.get("degenerate_rate", 0.0))
        self._log_stability_metrics()

        return total_loss

    def _log_stability_metrics(self):
        """Log EBT-specific stability metrics for monitoring divergence."""
        # Alpha (MCMC step size) — key indicator of energy landscape scale
        if hasattr(self.model, 'alpha'):
            self.log("stability/alpha", self.model.alpha.item())

        # Parameter norms for key components
        if hasattr(self.model, 'vocab_to_embed'):
            v2e_norm = sum(p.data.norm().item() ** 2 for p in self.model.vocab_to_embed.parameters()) ** 0.5
            self.log("stability/vocab_to_embed_norm", v2e_norm)

        # Transformer output layer norm (early warning of scale drift)
        if hasattr(self.model, 'transformer'):
            t_params = list(self.model.transformer.parameters())
            if t_params:
                last_norm = t_params[-1].data.norm().item()
                self.log("stability/transformer_last_param_norm", last_norm)

    @torch.no_grad()
    def _generate_and_score(self, batch):
        """Generate completions, compute rewards and advantages.

        All done without gradients to save memory.
        """
        prompt_ids = batch["prompt_ids"].to(self.device)
        prompt_lengths = batch["prompt_lengths"].to(self.device)
        puzzles = batch["puzzles"]
        solutions = batch["solutions"]

        num_prompts = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        # ── 1. Generate completions ──────────────────────────────────────────
        self.model.eval()
        # [DEBUG-RL] Reset debug flags for fresh logging in each phase.
        self.model._dbg_logged_embed_stats = False
        self.model._dbg_logged_logps_stats = False
        # Disable bf16 autocast during rollout: the EBT transformer's energy
        # head can overflow bf16 range, producing NaN/Inf that cascades to
        # corrupt logits and multinomial crashes. fp32 rollout is safe — the
        # generation batch is small and sequential (no backward graph).
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

        # completion_ids: (num_prompts * num_generations, comp_len)
        # completion_texts: list of str
        total_seqs = num_prompts * self.config.num_generations
        comp_len = completion_ids.shape[1]

        # ── 2. Compute rewards ───────────────────────────────────────────────
        # Expand puzzles/solutions to match num_generations
        expanded_puzzles = []
        expanded_solutions = []
        for i in range(num_prompts):
            for _ in range(self.config.num_generations):
                expanded_puzzles.append(puzzles[i])
                expanded_solutions.append(solutions[i])

        reward_details = compute_sudoku_rewards_detailed(
            completion_texts, expanded_puzzles, expanded_solutions
        )
        rewards = torch.tensor(
            [d["total"] for d in reward_details],
            dtype=torch.float32,
            device=self.device,
        )

        # ── 3. Compute old log-probs ─────────────────────────────────────────
        # Build full sequences (prompt + completion)
        expanded_prompts = prompt_ids.repeat_interleave(
            self.config.num_generations, dim=0
        )
        full_ids = torch.cat([expanded_prompts, completion_ids], dim=1)

        # [DEBUG-RL] Reset debug flag before old_logps computation.
        self.model._fwd_had_nan_energy = False
        self.model._dbg_logged_logps_stats = False
        with torch.amp.autocast('cuda', enabled=False):
            old_per_token_logps = get_per_token_logps(
                self.model, full_ids, prompt_len, learning=False
            )
        # [DEBUG-RL] Check if old_logps forward hit NaN.
        if getattr(self.model, '_fwd_had_nan_energy', False):
            import os as _os
            _rank = _os.environ.get('LOCAL_RANK', '?')
            print(
                f"[DBG-RL][rank={_rank}] old_logps forward hit NaN energy! "
                f"old_logps finite={torch.isfinite(old_per_token_logps).all().item()} "
                f"range=[{old_per_token_logps.min().item():.4e},{old_per_token_logps.max().item():.4e}]",
                flush=True,
            )
        # Mask out padding positions
        old_per_token_logps = old_per_token_logps[:, :comp_len] * completion_masks.float()

        # ── 4. Compute ref log-probs (for KL) ────────────────────────────────
        ref_per_token_logps = None
        if self.ref_model is not None and self.config.beta > 0.0:
            with torch.amp.autocast('cuda', enabled=False):
                ref_per_token_logps = get_per_token_logps(
                    self.ref_model, full_ids, prompt_len, learning=False
                )
            ref_per_token_logps = ref_per_token_logps[:, :comp_len] * completion_masks.float()
            # Defensive detach (we're already inside @torch.no_grad() in
            # _generate_and_score, but after Step 1.1 logprobs.py forces
            # set_grad_enabled(True) inside, which under the outer no_grad is
            # still safe — make it explicit anyway).
            ref_per_token_logps = ref_per_token_logps.detach()

        # ── 5. Compute advantages (group-relative) ───────────────────────────
        # Reshape rewards: (num_prompts, num_generations)
        grouped_rewards = rewards.view(num_prompts, self.config.num_generations)
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        group_std_raw = grouped_rewards.std(dim=1, keepdim=True)
        # Detect degenerate groups (all completions identical → std≈0). In the
        # sudoku task early-stage rollouts almost always degenerate because all
        # completions fail to parse → reward=0 for all → std=0.
        degenerate = (group_std_raw.squeeze(-1) < 1e-6)  # (num_prompts,)
        # Use a saner lower bound than 1e-8 to prevent advantage explosion.
        group_std_safe = group_std_raw.clamp(min=1e-4)
        advantages = ((grouped_rewards - group_mean) / group_std_safe).view(-1)
        # Zero-out degenerate groups (no learning signal anyway).
        deg_mask = degenerate.repeat_interleave(self.config.num_generations)
        advantages = torch.where(deg_mask, torch.zeros_like(advantages), advantages)
        # Absolute clip on advantage magnitude: with reward in [0, 3] a legit
        # advantage shouldn't exceed ±3; 5 leaves headroom.
        advantages = advantages.clamp(-5.0, 5.0)
        degenerate_rate = degenerate.float().mean().item()

        # ── Metrics ──────────────────────────────────────────────────────────
        avg_comp_len = completion_masks.sum(dim=1).float().mean().item()
        reward_components = {}
        for key in ["format", "clue_preservation", "blank_accuracy", "full_solve"]:
            reward_components[key] = sum(d[key] for d in reward_details) / len(reward_details)

        return {
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "completion_masks": completion_masks,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "avg_completion_length": avg_comp_len,
            "reward_components": reward_components,
            "degenerate_rate": degenerate_rate,
        }

    def _compute_grpo_loss(self, gen_data, iteration):
        """Compute PPO-clipped loss + KL penalty for one iteration."""
        full_ids = gen_data["full_ids"]
        prompt_len = gen_data["prompt_len"]
        completion_masks = gen_data["completion_masks"].float()
        old_logps = gen_data["old_per_token_logps"]
        ref_logps = gen_data["ref_per_token_logps"]
        advantages = gen_data["advantages"]

        comp_len = completion_masks.shape[1]

        # ── Compute current log-probs (with grad) ────────────────────────────
        learning_for_current = bool(self.config.use_learning_mode_for_logprobs)
        # [DEBUG-RL] Reset NaN flag and debug log flag before current_logps.
        self.model._fwd_had_nan_energy = False
        self.model._dbg_logged_logps_stats = False
        current_logps = get_per_token_logps(
            self.model,
            full_ids,
            prompt_len,
            learning=learning_for_current,
        )
        assert current_logps.requires_grad, "current_logps must require grad for DDP"
        current_logps = current_logps[:, :comp_len] * completion_masks

        # ── Cross-rank NaN sync: if ANY rank hit NaN, ALL ranks skip ─────────
        # This prevents DDP hang from graph divergence across ranks.
        import torch.distributed as dist
        local_bad = torch.tensor(
            1.0 if (
                getattr(self.model, "_fwd_had_nan_energy", False)
                or not torch.isfinite(current_logps).all()
                or not torch.isfinite(old_logps).all()
                or not torch.isfinite(advantages).all()
                or (completion_masks.sum().item() < 1.0)
            ) else 0.0,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        any_rank_bad = local_bad.item() > 0.5

        if any_rank_bad:
            import os as _os
            _rank = _os.environ.get('LOCAL_RANK', '?')
            _local_nan = getattr(self.model, "_fwd_had_nan_energy", False)
            _cur_finite = torch.isfinite(current_logps).all().item()
            _old_finite = torch.isfinite(old_logps).all().item()
            print(
                f"[DBG-RL][rank={_rank}] SKIPPING STEP (cross-rank sync): "
                f"local_nan_energy={_local_nan} "
                f"current_logps_finite={_cur_finite} "
                f"old_logps_finite={_old_finite} "
                f"current_logps_range=[{current_logps.min().item():.4e},{current_logps.max().item():.4e}] "
                f"old_logps_range=[{old_logps.min().item():.4e},{old_logps.max().item():.4e}]",
                flush=True,
            )
            placeholder = sum(
                (p * 0.0).sum() for p in self.model.parameters() if p.requires_grad
            )
            self.log("stability/skipped_step", 1.0)
            return placeholder, {"policy_loss": 0.0, "kl": 0.0, "clip_ratio": 0.0}
        self.log("stability/skipped_step", 0.0)

        # PPO ratio
        ratio = torch.exp(current_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)

        # Per-token loss
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL penalty
        kl_value = 0.0
        if ref_logps is not None and self.config.beta > 0.0:
            # Approximate KL: exp(ref - cur) - (ref - cur) - 1.
            # Clamp the difference to prevent exp blowup when cur - ref is huge.
            kl_diff = (ref_logps - current_logps).clamp(-10.0, 10.0)
            per_token_kl = torch.exp(kl_diff) - kl_diff - 1.0
            # Mask padding positions so they don't contribute to KL.
            per_token_kl = per_token_kl * completion_masks
            per_token_loss = per_token_loss + self.config.beta * per_token_kl
            kl_value = per_token_kl.sum() / completion_masks.sum().clamp(min=1.0)
            kl_value = kl_value.item()

        # Aggregate loss (mean over valid tokens)
        loss = (per_token_loss * completion_masks).sum() / completion_masks.sum().clamp(min=1.0)

        # Metrics
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_masks).sum() / completion_masks.sum().clamp(min=1.0)

        metrics = {
            "policy_loss": loss.item(),
            "kl": kl_value,
            "clip_ratio": clip_ratio.item(),
        }

        return loss, metrics

    # ══════════════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        """Validate by generating and scoring without updates."""
        gen_data = self._generate_and_score(batch)
        self.log("val/reward_mean", gen_data["reward_mean"], prog_bar=True, sync_dist=True)
        self.log("val/reward_std", gen_data["reward_std"], sync_dist=True)
        self.log("val/completion_length", gen_data["avg_completion_length"], sync_dist=True)
        for key, val in gen_data.get("reward_components", {}).items():
            self.log(f"val/reward_{key}", val, sync_dist=True)
        return gen_data["reward_mean"]

    # ══════════════════════════════════════════════════════════════════════════
    # Optimizer
    # ══════════════════════════════════════════════════════════════════════════

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms, sanitize NaN grads, and apply per-parameter clipping."""
        total_norm = 0.0
        alpha_grad = None
        max_param_grad_norm = 0.0
        nan_grad_params = 0

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                # Sanitize NaN/Inf gradients BEFORE clipping — clamp(NaN) is still NaN.
                if not torch.isfinite(p.grad.data).all():
                    nan_grad_params += 1
                    p.grad.data = torch.where(
                        torch.isfinite(p.grad.data), p.grad.data,
                        torch.zeros_like(p.grad.data)
                    )
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_param_grad_norm = max(max_param_grad_norm, param_norm)
                if name == "alpha":
                    alpha_grad = p.grad.data.item()
        total_norm = total_norm ** 0.5
        self.log("stability/grad_norm_before_clip", total_norm)
        self.log("stability/max_param_grad_norm", max_param_grad_norm)
        self.log("stability/nan_grad_params", float(nan_grad_params))
        if alpha_grad is not None:
            self.log("stability/alpha_grad", alpha_grad)

        if nan_grad_params > 0:
            import os as _os
            _rank = _os.environ.get('LOCAL_RANK', '?')
            print(
                f"[DBG-RL][rank={_rank}] Sanitized NaN grads in {nan_grad_params} params "
                f"(total_norm_after={total_norm:.4e})",
                flush=True,
            )

        # Per-parameter gradient clipping
        if self.config.max_grad_per_param > 0.0:
            clip_val = self.config.max_grad_per_param
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)

    def configure_optimizers(self):
        """AdamW optimizer with linear warmup + cosine decay."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Linear warmup then cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Data
    # ══════════════════════════════════════════════════════════════════════════

    def train_dataloader(self):
        dataset = SudokuRLPromptDataset(
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            split="train",
            data_dir=self.config.data_dir or None,
            augment=self.config.augment,
            seed=self.config.seed,
        )
        return DataLoader(
            dataset,
            batch_size=1,  # Each "batch" is one prompt; we generate N completions
            collate_fn=lambda batch: collate_rl_prompts(
                batch, self.tokenizer, self.config.max_prompt_length
            ),
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = SudokuRLPromptDataset(
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_prompt_length,
            split="val",
            data_dir=self.config.data_dir or None,
            augment=False,
            seed=self.config.seed + 1000,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=lambda batch: collate_rl_prompts(
                batch, self.tokenizer, self.config.max_prompt_length
            ),
            num_workers=1,
            pin_memory=True,
        )
