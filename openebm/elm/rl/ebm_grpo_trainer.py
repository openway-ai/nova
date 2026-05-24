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
from openebm.elm.rl.logprobs import get_per_token_logps, compute_sequence_energy
from openebm.elm.rl.rewards import compute_sudoku_rewards, compute_sudoku_rewards_detailed
from openebm.elm.rl.rollout import generate_completions
from openebm.elm.rl.sudoku_dataset_rl import SudokuRLPromptDataset, collate_rl_prompts
from openebm.elm.rl.trajectory_logger import (
    AlignedMetricsPrinter,
    TrajectoryLogger,
    TrajLoggerConfig,
    logp,
)


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

        # v3 fix C1: explicitly freeze input embeddings during RL.
        # `compute_sequence_energy` already detaches `real_embeddings` from the
        # graph (logprobs.py:39), so embeddings.weight receives no task signal.
        # However AdamW + weight_decay still drift it on every step. Freezing
        # eliminates that silent drift (and saves DDP all_reduce on a 32k×1664
        # matrix). Output projection `vocab_to_embed` stays trainable.
        if hasattr(self.model, 'embeddings'):
            for p in self.model.embeddings.parameters():
                p.requires_grad_(False)

        self.tokenizer = tokenizer

        # Build hparams namespace for generate.py compatibility
        self._gen_hparams = self._build_gen_hparams()

        # Metrics tracking
        self._step_metrics = {}
        self._skip_optim_step = False

        # Trajectory logger — initialised lazily on first use so the output
        # dir is available. Set output_dir via TRAJ_OUTPUT_DIR env var or
        # falls back to <checkpoint_dir>/../trajectories.
        self._traj_logger: TrajectoryLogger | None = None
        self._metrics_printer = AlignedMetricsPrinter()

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

        # ── Skip-degenerate (DDP-safe) ─────────────────────────────────────
        # Lightning forbids returning None under DDP. Instead we:
        #   1. all-reduce the per-rank skip decision so every rank takes the
        #      same branch (avoids DDP collective mismatch / hang),
        #   2. on agreement, return a graph-attached placeholder loss whose
        #      gradient is identically zero,
        #   3. set self._skip_optim_step so on_before_optimizer_step clears
        #      every p.grad to None — AdamW's per-param branch is
        #      `if p.grad is None: continue`, which truly skips the update
        #      (including weight_decay).
        skip_thr = getattr(self.config, "skip_degenerate_threshold", 1.01)
        local_skip = float(gen_data["degenerate_rate"] > skip_thr)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([local_skip], device=self.device)
            # MIN: skip only when every rank agrees the batch is degenerate.
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            global_skip = flag.item() > 0.5
        else:
            global_skip = local_skip > 0.5

        self._skip_optim_step = global_skip

        if global_skip:
            self.log("stability/skipped_step", 1.0)
            self.log("stability/degenerate_group_rate", gen_data["degenerate_rate"])
            self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
            step = self.global_step
            if step % self.config.log_interval == 0:
                import os as _os
                if _os.environ.get('LOCAL_RANK', '0') == '0':
                    print(
                        f"[GRPO] step={step} SKIPPED degen={gen_data['degenerate_rate']:.2f} "
                        f"reward={gen_data['reward_mean']:.3f}",
                        flush=True,
                    )
            # Placeholder loss: every trainable parameter contributes a
            # zero-valued term so DDP's backward all-reduce covers them all
            # (required when find_unused_parameters=True with no real grad).
            placeholder = sum(
                (p * 0.0).sum()
                for p in self.model.parameters()
                if p.requires_grad
            )
            return placeholder
        self.log("stability/skipped_step", 0.0)

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
        loss_val = total_loss.item()
        self.log("train/loss", loss_val, prog_bar=True)
        self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
        self.log("train/reward_std", gen_data["reward_std"])
        self.log("train/reward_var", gen_data["reward_var"])
        self.log("train/advantage_var", gen_data["advantage_var"])
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

        # ── Stdout metrics (visible in train.log via pipe) ───────────────────
        step = self.global_step
        if step % self.config.log_interval == 0:
            rc = gen_data.get("reward_components", {})
            print(
                f"[GRPO] step={step} | "
                f"loss={loss_val:.4f} | "
                f"reward={gen_data['reward_mean']:.3f}±{gen_data['reward_std']:.3f} | "
                f"fmt={rc.get('format', 0):.2f} clue={rc.get('clue_preservation', 0):.2f} "
                f"blank_acc={rc.get('blank_accuracy', 0):.3f} solve={rc.get('full_solve', 0):.2f} | "
                f"comp_len={gen_data['avg_completion_length']:.0f} | "
                f"clip={total_metrics['clip_ratio']:.2f} degen={gen_data.get('degenerate_rate', 0):.2f}",
                flush=True,
            )

        # ── Trajectory logging & collapse detection ──────────────────────────
        self._maybe_init_traj_logger()
        if self._traj_logger is not None:
            completions = gen_data.get("_completion_texts", [])
            rewards_list = gen_data.get("_rewards", [])
            energies_list = gen_data.get("_old_energies", [])
            adv_list = gen_data.get("_advantages", [])
            lens_list = gen_data.get("_completion_masks_lens", [])
            puzzles_list = gen_data.get("_puzzles", [])

            # Periodic stdout + JSONL dump
            self._traj_logger.maybe_log(
                step=step,
                prompts=puzzles_list,
                completions=completions,
                rewards=rewards_list,
                ground_truths=puzzles_list,
                energies=energies_list,
                advantages=adv_list,
                kls=[total_metrics["kl"]] * len(completions),
                response_lens=lens_list,
            )

            # Collapse detection
            unique_ratio = (
                len(set(completions)) / max(1, len(completions))
                if completions else 0.0
            )
            self._traj_logger.detect_collapse(
                step=step,
                rewards=rewards_list,
                unique_completion_ratio=unique_ratio,
                prompts=puzzles_list,
                completions=completions,
                ground_truths=puzzles_list,
                energies=energies_list,
            )
            self.log("train/unique_completion_ratio", unique_ratio)

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

    def _maybe_init_traj_logger(self):
        """Lazily initialise TrajectoryLogger on first call (rank-0 only).

        Output dir resolution order:
          1. config.traj_output_dir (CLI flag)
          2. TRAJ_OUTPUT_DIR env var
          3. <trainer.log_dir>/trajectories (Lightning sets log_dir after fit starts)
          4. fallback to ./trajectories

        Returns the logger instance (may be None on non-rank-0).
        """
        if self._traj_logger is not None:
            return self._traj_logger
        traj_root = (
            getattr(self.config, "traj_output_dir", "")
            or os.environ.get("TRAJ_OUTPUT_DIR")
            or (str(self.trainer.log_dir) if self.trainer and hasattr(self.trainer, "log_dir") and self.trainer.log_dir else None)
            or "."
        )
        overrides = {}
        if getattr(self.config, "traj_log_interval", 0) > 0:
            overrides["sample_every_n_steps"] = self.config.traj_log_interval
        if getattr(self.config, "traj_num_samples", 0) > 0:
            overrides["samples_per_dump"] = self.config.traj_num_samples
        if getattr(self.config, "collapse_check_window", 0) > 0:
            overrides["collapse_window"] = self.config.collapse_check_window
        cfg = TrajLoggerConfig.from_env(output_dir=traj_root, **overrides)
        self._traj_logger = TrajectoryLogger(cfg)
        return self._traj_logger

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

        # One-time debug: print first completion and reward to catch format regressions early.
        if not getattr(self, '_dbg_logged_first_rollout', False):
            self._dbg_logged_first_rollout = True
            sample_text = completion_texts[0][:300] if completion_texts else "(empty)"
            print(f"[DBG-RL] first rollout sample: {repr(sample_text)}", flush=True)
            print(f"[DBG-RL] first rollout reward detail: {reward_details[0]}", flush=True)

        # ── 3. Compute old energies / logprobs ───────────────────────────────
        expanded_prompts = prompt_ids.repeat_interleave(
            self.config.num_generations, dim=0
        )
        full_ids = torch.cat([expanded_prompts, completion_ids], dim=1)

        # Energy-based path (for energy_gspo and energy_reinforce).
        # P0b: keep autocast disabled to match _loss_energy_*'s autocast(False)
        # block. Mismatched precision causes systematic bias in (curr - old).
        with torch.amp.autocast('cuda', enabled=False):
            old_energies = compute_sequence_energy(self.model, full_ids, prompt_len)

        # Token-level logprobs path (for token_logprobs variant; KL is now
        # handled via sequence-energy proxy in _loss_energy_reinforce when
        # beta>0, so we don't need per-token ref logprobs there).
        old_per_token_logps = None
        if self.config.rl_loss_type == "token_logprobs":
            with torch.amp.autocast('cuda', enabled=False):
                old_per_token_logps, _ = get_per_token_logps(
                    self.model, full_ids, prompt_len, learning=False
                )
            old_per_token_logps = old_per_token_logps[:, :comp_len] * completion_masks.float()

        # ── 4. Compute ref log-probs (for token_logprobs KL) ─────────────────
        ref_per_token_logps = None
        if self.config.rl_loss_type == "token_logprobs" and self.ref_model is not None and self.config.beta > 0.0:
            with torch.amp.autocast('cuda', enabled=False):
                ref_per_token_logps, _ = get_per_token_logps(
                    self.ref_model, full_ids, prompt_len, learning=False
                )
            ref_per_token_logps = ref_per_token_logps[:, :comp_len] * completion_masks.float()

        # ── 5. Compute advantages (group-relative) ───────────────────────────
        # Reshape rewards: (num_prompts, num_generations)
        grouped_rewards = rewards.view(num_prompts, self.config.num_generations)
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        group_std_raw = grouped_rewards.std(dim=1, keepdim=True)
        # Detect degenerate groups (all completions identical → std≈0). In the
        # sudoku task early-stage rollouts almost always degenerate because all
        # completions fail to parse → reward=0 for all → std=0.
        degenerate = (group_std_raw.squeeze(-1) < 1e-6)  # (num_prompts,)
        # Choose normalization: legacy per-group std vs. batch-wide std.
        # Global std prevents tiny intra-group std from amplifying noise.
        if getattr(self.config, "advantage_norm", "group") == "group_mean_global_std":
            global_std = grouped_rewards.flatten().std().clamp(
                min=self.config.global_std_min
            )
            advantages = ((grouped_rewards - group_mean) / global_std).view(-1)
        else:
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
        advantage_var = advantages.var(unbiased=False).item() if advantages.numel() > 1 else 0.0
        reward_var = float(rewards.var(unbiased=False).item()) if rewards.numel() > 1 else 0.0

        return {
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "completion_masks": completion_masks,
            "old_energies": old_energies.detach(),
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "reward_var": reward_var,
            "advantage_var": advantage_var,
            "avg_completion_length": avg_comp_len,
            "reward_components": reward_components,
            "degenerate_rate": degenerate_rate,
            # Raw arrays carried through for trajectory logging in training_step.
            "_completion_texts": completion_texts,
            "_rewards": rewards.detach().cpu().tolist(),
            "_old_energies": old_energies.detach().cpu().tolist(),
            "_advantages": advantages.detach().cpu().tolist(),
            "_completion_masks_lens": completion_masks.sum(dim=1).long().cpu().tolist(),
            "_puzzles": list(expanded_puzzles),
            "_solutions": list(expanded_solutions),
            "_prompt_ids_per_seq": expanded_prompts.cpu().tolist(),
        }

    def _compute_grpo_loss(self, gen_data, iteration):
        """Compute GRPO loss. Supports three variants via config.rl_loss_type."""
        full_ids = gen_data["full_ids"]
        prompt_len = gen_data["prompt_len"]
        completion_masks = gen_data["completion_masks"].float()
        advantages = gen_data["advantages"]

        loss_type = self.config.rl_loss_type

        # P0d: energy_reinforce assumes strict on-policy. With num_iterations>1
        # the policy has drifted but no importance ratio compensates. Warn once.
        if (loss_type == "energy_reinforce" and iteration > 0
                and not getattr(self, "_warned_reinforce_multi_iter", False)):
            self._warned_reinforce_multi_iter = True
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    "[WARN] energy_reinforce with num_iterations>1 is biased "
                    "(no importance ratio). Consider energy_gspo for multi-iter PPO.",
                    flush=True,
                )

        if loss_type == "energy_gspo":
            loss, metrics = self._loss_energy_gspo(gen_data)
        elif loss_type == "energy_reinforce":
            loss, metrics = self._loss_energy_reinforce(gen_data)
        elif loss_type == "token_logprobs":
            loss, metrics = self._loss_token_logprobs(gen_data, iteration)
        else:
            raise ValueError(f"Unknown rl_loss_type: {loss_type}")

        return loss, metrics

    def _energy_kl_anchor(self, full_ids, prompt_len, current_energies):
        """One-sided energy-KL anchor against ref_model.

        Returns (kl_penalty_term, kl_value, energy_drift). The penalty is
        ``relu(E_θ - E_ref).mean()`` — only punishes drift to higher energy
        than the SFT reference (= drift to lower likelihood than ref).
        Returns zeros + None when no ref_model / beta=0.

        Shared by `_loss_energy_gspo` and `_loss_energy_reinforce` so both
        loss types get the same SFT-anchoring behavior on top of their
        respective stability mechanisms (PPO clip vs. pure on-policy).
        """
        if self.ref_model is None or self.config.beta <= 0.0:
            return None, 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            ref_energies = compute_sequence_energy(self.ref_model, full_ids, prompt_len)
        energy_diff = current_energies - ref_energies.detach()
        energy_kl = energy_diff.clamp(min=0.0).mean()
        return energy_kl, energy_kl.item(), energy_diff.mean().item()

    def _loss_energy_gspo(self, gen_data):
        """Energy-GSPO: sequence-level energy ratio with PPO clipping.

        v3 fix: `compute_sequence_energy` already returns per-token MEAN
        energy (logprobs.py:94 does `comp_energy.mean(dim=1)`). Therefore
        `log_ratio = -(curr - old)` is already length-normalized; dividing
        by `comp_lengths` again is double-normalization, shrinking log_ratio
        to O(1/L²) and effectively disabling PPO clipping (sudoku v2 logs
        showed clip=0.00 throughout). Removing the extra `/comp_lengths`
        also brings GSPO into mathematical agreement with energy_reinforce.
        """
        full_ids = gen_data["full_ids"]
        prompt_len = gen_data["prompt_len"]
        completion_masks = gen_data["completion_masks"].float()
        old_energies = gen_data["old_energies"]
        advantages = gen_data["advantages"]

        # P0b: keep both energy computations under the same autocast context
        # (rollout-time `old_energies` runs under autocast(enabled=False)).
        with torch.amp.autocast('cuda', enabled=False):
            current_energies = compute_sequence_energy(self.model, full_ids, prompt_len)

        # Sequence-level importance ratio. Energies are already per-token mean.
        raw_log_ratio = -(current_energies - old_energies)
        log_ratio = raw_log_ratio.clamp(-10.0, 10.0)
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(loss1, loss2).mean()

        # KL anchor on top of PPO clip. PPO clip bounds per-step ratio drift,
        # the KL term bounds cumulative drift from SFT ref. They're orthogonal.
        kl_term, kl_value, energy_drift = self._energy_kl_anchor(
            full_ids, prompt_len, current_energies)
        if kl_term is not None:
            loss = policy_loss + self.config.beta * kl_term
            self.log("train/energy_kl_proxy", kl_value)
            self.log("train/energy_drift", energy_drift)
        else:
            loss = policy_loss

        # P1a: log_ratio clamp visibility.
        log_ratio_abs_max = raw_log_ratio.abs().max().item()
        log_ratio_clamp_rate = (raw_log_ratio.abs() > 10.0).float().mean().item()
        # P1b: "effective clip rate" = how often PPO `min` actually selected
        # the clipped branch (matches PPO paper's definition).
        effective_clip_rate = (loss2 < loss1).float().mean().item()

        self.log("train/energy_mean", current_energies.mean().item())
        self.log("train/ratio_mean", ratio.mean().item())
        self.log("stability/log_ratio_abs_max", log_ratio_abs_max)
        self.log("stability/log_ratio_clamp_rate", log_ratio_clamp_rate)
        self.log("train/effective_clip_rate", effective_clip_rate)

        if self.global_step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO-EXT] step={self.global_step} "
                    f"ratio_mean={ratio.mean().item():.4f} "
                    f"log_ratio_abs_max={log_ratio_abs_max:.4f} "
                    f"clamp_rate={log_ratio_clamp_rate:.3f} "
                    f"effective_clip={effective_clip_rate:.3f}",
                    flush=True,
                )

        return loss, {
            "policy_loss": policy_loss.item(),
            "kl": kl_value,
            "clip_ratio": effective_clip_rate,
        }

    def _loss_energy_reinforce(self, gen_data):
        """Energy-REINFORCE with one-sided energy-KL anchor.

        v3 fixes:
          - P0b: keep `current_energies` under autocast(enabled=False) to
            match the rollout-time `old_energies` (avoids ratio drift between
            energy snapshots; also matters when ref_model energies are taken).
          - P0c: replace symmetric (E_θ - E_ref)^2 with one-sided ReLU
            penalty `relu(E_θ - E_ref)`. The squared form penalises *both*
            directions, including the desired "E_θ < E_ref on high-reward
            samples" — this directly explains the "reward stuck oscillating"
            symptom in v2 (analysis_output_20260524). One-sided penalises
            only the unwanted direction (drift to higher energy than ref).
        """
        full_ids = gen_data["full_ids"]
        prompt_len = gen_data["prompt_len"]
        advantages = gen_data["advantages"]

        with torch.amp.autocast('cuda', enabled=False):
            current_energies = compute_sequence_energy(self.model, full_ids, prompt_len)

        # Energies are already per-token mean (logprobs.py:94), so no length
        # normalisation needed here.
        policy_loss = (advantages * current_energies).mean()

        # One-sided energy-KL anchor (shared with _loss_energy_gspo).
        kl_term, kl_value, energy_drift = self._energy_kl_anchor(
            full_ids, prompt_len, current_energies)
        if kl_term is not None:
            loss = policy_loss + self.config.beta * kl_term
            self.log("train/energy_kl_proxy", kl_value)
            self.log("train/energy_drift", energy_drift)
        else:
            loss = policy_loss

        cur_energy_mean = current_energies.mean().item()
        self.log("train/energy_mean", cur_energy_mean)

        if self.global_step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO-EXT] step={self.global_step} "
                    f"kl_proxy={kl_value:.4f} energy_drift={energy_drift:+.4f} "
                    f"energy_mean={cur_energy_mean:+.4f} policy_loss={policy_loss.item():+.4f}",
                    flush=True,
                )

        return loss, {"policy_loss": policy_loss.item(), "kl": kl_value, "clip_ratio": 0.0}

    def _loss_token_logprobs(self, gen_data, iteration):
        """Token-level logprobs ratio (legacy). Known to produce NaN with EBT."""
        full_ids = gen_data["full_ids"]
        prompt_len = gen_data["prompt_len"]
        completion_masks = gen_data["completion_masks"].float()
        old_logps = gen_data["old_per_token_logps"]
        ref_logps = gen_data["ref_per_token_logps"]
        advantages = gen_data["advantages"]
        comp_len = completion_masks.shape[1]

        learning_for_current = bool(self.config.use_learning_mode_for_logprobs)
        current_logps, _ = get_per_token_logps(
            self.model, full_ids, prompt_len, learning=learning_for_current,
        )
        current_logps = current_logps[:, :comp_len] * completion_masks

        # Cross-rank NaN sync
        import torch.distributed as dist
        local_bad = torch.tensor(
            1.0 if (
                not torch.isfinite(current_logps).all()
                or not torch.isfinite(old_logps).all()
            ) else 0.0,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
        if local_bad.item() > 0.5:
            placeholder = sum(
                (p * 0.0).sum() for p in self.model.parameters() if p.requires_grad
            )
            self.log("stability/skipped_step", 1.0)
            return placeholder, {"policy_loss": 0.0, "kl": 0.0, "clip_ratio": 0.0}
        self.log("stability/skipped_step", 0.0)

        ratio = torch.exp(current_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.epsilon, 1.0 + self.config.epsilon)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        kl_value = 0.0
        if ref_logps is not None and self.config.beta > 0.0:
            kl_diff = (ref_logps - current_logps).clamp(-10.0, 10.0)
            per_token_kl = torch.exp(kl_diff) - kl_diff - 1.0
            per_token_kl = per_token_kl * completion_masks
            per_token_loss = per_token_loss + self.config.beta * per_token_kl
            kl_value = (per_token_kl.sum() / completion_masks.sum().clamp(min=1.0)).item()

        loss = (per_token_loss * completion_masks).sum() / completion_masks.sum().clamp(min=1.0)
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_masks).sum() / completion_masks.sum().clamp(min=1.0)

        return loss, {"policy_loss": loss.item(), "kl": kl_value, "clip_ratio": clip_ratio.item()}

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
        # If training_step flagged this batch as skip-degen, drop every grad
        # to None so AdamW (and any stateful optimizer) leaves params untouched.
        # Lightning still steps the LR scheduler — that's a small acceptable cost.
        if getattr(self, "_skip_optim_step", False):
            for p in self.model.parameters():
                p.grad = None
            self._skip_optim_step = False
            self.log("stability/grad_norm_before_clip", 0.0)
            self.log("stability/max_param_grad_norm", 0.0)
            self.log("stability/nan_grad_params", 0.0)
            return

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

        # Stdout grad stats (rank 0 only, every log_interval steps)
        step = self.global_step
        if step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO] step={step} grad_norm={total_norm:.4e} "
                    f"max_param_grad={max_param_grad_norm:.4e} nan_params={nan_grad_params}",
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
