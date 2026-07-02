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
import json
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
from openebm.elm.rl.optimizer_utils import make_skip_missing_grad_muon_adamw
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
        self._manual_gspo_updates = (
            getattr(config, "gspo_update_epochs", 1) > 1
            and config.rl_loss_type == "energy_gspo"
        )
        if getattr(config, "gspo_update_epochs", 1) > 1 and config.rl_loss_type != "energy_gspo":
            raise ValueError("gspo_update_epochs > 1 is only supported for rl_loss_type='energy_gspo'")
        if self._manual_gspo_updates:
            self.automatic_optimization = False

        # Trajectory logger — initialised lazily on first use so the output
        # dir is available. Set output_dir via TRAJ_OUTPUT_DIR env var or
        # falls back to <checkpoint_dir>/../trajectories.
        self._traj_logger: TrajectoryLogger | None = None
        self._metrics_printer = AlignedMetricsPrinter()
        self._phase_log_enabled = os.environ.get("RL_PHASE_LOG", "1").lower() in ("1", "true", "yes")

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

    def _is_rank0(self):
        return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("RANK", "0") == "0"

    def _json_safe(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return self._json_safe(value.detach().item())
            return [self._json_safe(v) for v in value.detach().cpu().tolist()]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _log_json_event(self, event, step, payload):
        if not self._is_rank0():
            return
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "step": int(step),
            **self._json_safe(payload),
        }
        print("[GRPO-JSON] " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    def _heartbeat_dir(self) -> str:
        return (
            getattr(self.config, "traj_output_dir", "")
            or os.environ.get("TRAJ_OUTPUT_DIR")
            or (str(self.trainer.log_dir) if self.trainer and getattr(self.trainer, "log_dir", None) else ".")
        )

    def _log_phase(self, phase: str, **fields) -> None:
        """Write a tiny rank-0 heartbeat and optional concise phase line."""
        if not self._is_rank0():
            return
        step = int(getattr(self, "global_step", 0))
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "phase": phase,
            **self._json_safe(fields),
        }
        try:
            out_dir = self._heartbeat_dir()
            os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
            path = os.path.join(out_dir, "logs", "heartbeat.json")
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(record, f, ensure_ascii=False, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)
        except Exception:
            pass
        if self._phase_log_enabled and (step == 0 or step % self.config.log_interval == 0):
            details = " ".join(f"{k}={v}" for k, v in record.items() if k not in ("ts", "phase"))
            print(f"[RL-PHASE] {phase} {details}", flush=True)

    def _zero_placeholder_loss(self):
        """Cheap graph-attached zero loss for skipped DDP steps.

        DDP runs with find_unused_parameters=True in the RL scripts, so a single
        used parameter is enough to keep the graph attached while unused
        parameters are marked ready by DDP. Touching every trainable parameter
        makes a skipped step as expensive as a full backward/all-reduce.
        """
        for p in self.model.parameters():
            if p.requires_grad:
                return p.reshape(-1)[0] * 0.0
        return torch.zeros((), device=self.device, requires_grad=True)

    def _zero_full_graph_loss(self, gen_data):
        """Zero loss that touches the normal energy graph.

        Used when only the local rank has a bad rollout. Healthy ranks still
        contribute real gradients through DDP all-reduce; the bad rank
        contributes zeros for the same graph instead of a KL-only update.
        """
        with torch.amp.autocast('cuda', enabled=False):
            energies = compute_sequence_energy(
                self.model,
                gen_data["full_ids"],
                gen_data["prompt_len"],
                gen_data["completion_masks"].float(),
            )
        return energies.sum() * 0.0

    def _reward_component_short_names(self, components):
        short = {
            "format": "fmt",
            "clue_preservation": "clue",
            "blank_accuracy": "blank_acc",
            "constraint_validity": "validity",
            "full_solve": "solve",
            "exact_match": "exact",
            "partial_credit": "partial",
            "length_penalty": "len_pen",
        }
        return {short.get(k, k): v for k, v in components.items()}

    def _component_text(self, components):
        short_components = self._reward_component_short_names(components)
        return " ".join(f"{k}={float(v):.3f}" for k, v in short_components.items())

    def _completion_unique_ratio(self, completions):
        if not completions:
            return 0.0
        return len(set(completions)) / max(1, len(completions))

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
        step_start = time.time()
        self._log_phase("training_step_start", batch_idx=int(batch_idx))
        gen_data = self._generate_and_score(batch)
        self._log_phase(
            "rollout_ready",
            elapsed_s=round(time.time() - step_start, 3),
            reward_mean=round(float(gen_data.get("reward_mean", 0.0)), 4),
            reward_std=round(float(gen_data.get("reward_std", 0.0)), 4),
        )

        # ── Skip-degenerate (DDP-safe) ─────────────────────────────────────
        # Lightning forbids returning None under DDP. Instead we:
        #   1. all-reduce the per-rank skip decision so every rank takes the
        #      same branch (avoids DDP collective mismatch / hang),
        #   2. on agreement, return a cheap graph-attached placeholder loss
        #      whose gradient is identically zero,
        #   3. set self._skip_optim_step so on_before_optimizer_step clears
        #      every p.grad to None — AdamW's per-param branch is
        #      `if p.grad is None: continue`, which truly skips the update
        #      (including weight_decay).
        completions_for_skip = gen_data.get("_completion_texts", [])
        unique_ratio_for_skip = self._completion_unique_ratio(completions_for_skip)
        skip_thr = getattr(self.config, "skip_degenerate_threshold", 1.01)
        min_reward_std = getattr(self.config, "min_reward_std_to_update", 0.0)
        min_unique_ratio = getattr(self.config, "min_unique_completion_ratio_to_update", 0.0)
        skip_reasons = []
        if gen_data["degenerate_rate"] > skip_thr:
            skip_reasons.append("degenerate_group_rate")
        if min_reward_std > 0.0 and gen_data.get("reward_std", 0.0) < min_reward_std:
            skip_reasons.append("low_reward_std")
        if min_unique_ratio > 0.0 and unique_ratio_for_skip < min_unique_ratio:
            skip_reasons.append("low_unique_completion_ratio")
        local_skip = float(bool(skip_reasons))
        import torch.distributed as dist
        consensus_t0 = time.time()
        reason_keys = ("degenerate_group_rate", "low_reward_std", "low_unique_completion_ratio")
        reason_counts = torch.tensor(
            [1.0 if reason in skip_reasons else 0.0 for reason in reason_keys],
            device=self.device,
        )
        skip_rank_count = torch.tensor([local_skip], device=self.device)
        skip_consensus = getattr(self.config, "skip_consensus", "any")
        if skip_consensus not in ("all", "any", "local"):
            raise ValueError(f"Unknown skip_consensus: {skip_consensus}")
        world_size = 1
        local_zero_update = False
        self._log_phase(
            "skip_consensus_start",
            local_skip=local_skip,
            skip_reasons=skip_reasons,
            unique_ratio=round(float(unique_ratio_for_skip), 4),
        )
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(reason_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(skip_rank_count, op=dist.ReduceOp.SUM)
            skip_count = int(round(float(skip_rank_count.detach().cpu().item())))
            any_rank_bad = skip_count > 0
            all_ranks_bad = skip_count >= world_size
            if skip_consensus == "any":
                # Stability-first mode: if any rank sees a bad rollout, skip the
                # optimizer step on all ranks to avoid one poisoned shard moving
                # the shared policy.
                global_skip = any_rank_bad
            elif skip_consensus == "local":
                # Hybrid mode: if every rank is bad, skip the whole step. If
                # only this rank is bad, contribute a zero full-graph loss while
                # healthy ranks contribute real gradients. This avoids KL-only
                # updates from bad ranks without discarding healthy rank signal.
                global_skip = all_ranks_bad
                local_zero_update = local_skip > 0.5 and not global_skip
            else:
                # Throughput-first mode: skip only when every rank agrees the
                # batch has no useful signal.
                global_skip = all_ranks_bad
        else:
            global_skip = local_skip > 0.5
        global_skip_reasons = [
            reason for reason, count in zip(reason_keys, reason_counts.detach().cpu().tolist())
            if count > 0.5
        ]
        self._log_phase(
            "skip_consensus_done",
            elapsed_s=round(time.time() - consensus_t0, 3),
            global_skip=bool(global_skip),
            skip_rank_count=int(skip_rank_count.detach().cpu().item()),
            global_skip_reasons=global_skip_reasons,
            skip_consensus=skip_consensus,
            world_size=world_size,
            local_zero_update=bool(local_zero_update),
        )

        self.log("stability/reward_std_too_low", float("low_reward_std" in skip_reasons))
        self.log("stability/unique_completion_ratio", unique_ratio_for_skip)
        self.log("stability/local_skip", local_skip)
        self._skip_optim_step = global_skip

        if global_skip:
            reported_skip_reasons = global_skip_reasons or skip_reasons
            self.log("stability/skipped_step", 1.0)
            self.log("stability/local_zero_update", 0.0)
            self.log("stability/degenerate_group_rate", gen_data["degenerate_rate"])
            self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
            step = self.global_step
            if step % self.config.log_interval == 0:
                if self._is_rank0():
                    print(
                        f"[GRPO] step={step} SKIPPED reason={','.join(reported_skip_reasons) or 'ddp_consensus'} "
                        f"degen={gen_data['degenerate_rate']:.2f} "
                        f"reward={gen_data['reward_mean']:.3f}±{gen_data['reward_std']:.3f} "
                        f"uniq={unique_ratio_for_skip:.2f}",
                        flush=True,
                    )
                self._log_json_event("rl_step", step, {
                    "loss": {
                        "total": 0.0,
                        "policy": 0.0,
                        "ref_energy_kl": 0.0,
                        "clip_ratio": 0.0,
                    },
                    "reward": {
                        "mean": gen_data["reward_mean"],
                        "std": gen_data["reward_std"],
                        "var": gen_data.get("reward_var", 0.0),
                        "advantage_var": gen_data.get("advantage_var", 0.0),
                        "components": gen_data.get("reward_components", {}),
                    },
                    "rollout": {
                        "completion_len_mean": gen_data.get("avg_completion_length", 0.0),
                        "unique_completion_ratio": unique_ratio_for_skip,
                        "degenerate_group_rate": gen_data.get("degenerate_rate", 0.0),
                        "skipped_step": 1.0,
                        "skip_reasons": reported_skip_reasons,
                        "local_skip_reasons": skip_reasons,
                        "global_skip_rank_count": int(skip_rank_count.detach().cpu().item()),
                    },
                })
            return self._zero_placeholder_loss()
        self.log("stability/skipped_step", 0.0)

        if local_zero_update:
            self.log("stability/local_zero_update", 1.0)
            self.log("stability/degenerate_group_rate", gen_data["degenerate_rate"])
            self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
            step = self.global_step
            self._log_phase(
                "local_zero_update",
                local_skip_reasons=skip_reasons,
                reward_mean=round(float(gen_data["reward_mean"]), 4),
                reward_std=round(float(gen_data["reward_std"]), 4),
            )
            if step % self.config.log_interval == 0:
                if self._is_rank0():
                    print(
                        f"[GRPO] step={step} LOCAL_ZERO reason={','.join(skip_reasons) or 'bad_local_rollout'} "
                        f"degen={gen_data['degenerate_rate']:.2f} "
                        f"reward={gen_data['reward_mean']:.3f}±{gen_data['reward_std']:.3f} "
                        f"uniq={unique_ratio_for_skip:.2f}",
                        flush=True,
                    )
                self._log_json_event("rl_step", step, {
                    "loss": {
                        "total": 0.0,
                        "policy": 0.0,
                        "ref_energy_kl": 0.0,
                        "clip_ratio": 0.0,
                    },
                    "reward": {
                        "mean": gen_data["reward_mean"],
                        "std": gen_data["reward_std"],
                        "var": gen_data.get("reward_var", 0.0),
                        "advantage_var": gen_data.get("advantage_var", 0.0),
                        "components": gen_data.get("reward_components", {}),
                    },
                    "rollout": {
                        "completion_len_mean": gen_data.get("avg_completion_length", 0.0),
                        "unique_completion_ratio": unique_ratio_for_skip,
                        "degenerate_group_rate": gen_data.get("degenerate_rate", 0.0),
                        "skipped_step": 0.0,
                        "local_zero_update": 1.0,
                        "skip_reasons": skip_reasons,
                        "global_skip_rank_count": int(skip_rank_count.detach().cpu().item()),
                    },
                })
            return self._zero_full_graph_loss(gen_data)
        self.log("stability/local_zero_update", 0.0)

        if self._manual_gspo_updates:
            total_loss, total_metrics = self._manual_gspo_rollout_updates(gen_data)
            if total_loss is None:
                return self._zero_placeholder_loss()
            self.log("train/gspo_update_epochs", float(self.config.gspo_update_epochs))
        else:
            # ── Phase 2: Policy Update (with grad) ───────────────────────────
            total_loss = torch.tensor(0.0, device=self.device)
            total_metrics = {
                "policy_loss": 0.0,
                "ref_energy_kl": 0.0,
                "kl": 0.0,  # legacy alias
                "clip_ratio": 0.0,
            }

            for iteration in range(self.config.num_iterations):
                loss, metrics = self._compute_grpo_loss(gen_data, iteration)
                total_loss = total_loss + loss / self.config.num_iterations
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v / self.config.num_iterations
        self._log_phase(
            "loss_ready",
            elapsed_s=round(time.time() - step_start, 3),
            loss=float(total_loss.detach().cpu().item()) if torch.isfinite(total_loss.detach()).item() else str(total_loss.detach().cpu().item()),
        )

        # Automatic optimization path returns this tensor for Lightning backward.
        # Manual GSPO path has already stepped the optimizer and returns a
        # detached scalar only for logging/progress display.
        if not torch.isfinite(total_loss):
            self._skip_optim_step = True
            step = self.global_step
            self.log("stability/nonfinite_loss", 1.0)
            self._log_json_event("rl_nonfinite_loss", step, {
                "loss": {"total": str(total_loss.detach().cpu().item())},
                "reward": {
                    "mean": gen_data.get("reward_mean"),
                    "std": gen_data.get("reward_std"),
                    "components": gen_data.get("reward_components", {}),
                },
                "rollout": {
                    "degenerate_group_rate": gen_data.get("degenerate_rate", 0.0),
                    "unique_completion_ratio": unique_ratio_for_skip,
                },
            })
            return self._zero_placeholder_loss()
        self.log("stability/nonfinite_loss", 0.0)

        # ── Logging ───────────────────────────────────────────────────────────
        loss_val = total_loss.item()
        self.log("train/loss", loss_val, prog_bar=True)
        self.log("train/reward_mean", gen_data["reward_mean"], prog_bar=True)
        self.log("train/reward_std", gen_data["reward_std"])
        self.log("train/reward_min", gen_data.get("reward_min", 0.0))
        self.log("train/reward_max", gen_data.get("reward_max", 0.0))
        self.log("train/reward_zero_frac", gen_data.get("reward_zero_frac", 0.0))
        self.log("train/reward_var", gen_data["reward_var"])
        self.log("train/advantage_var", gen_data["advantage_var"])
        self.log("train/policy_loss", total_metrics["policy_loss"])
        ref_energy_kl = total_metrics.get("ref_energy_kl", total_metrics.get("kl", 0.0))
        self.log("train/ref_energy_kl", ref_energy_kl)
        self.log("train/clip_ratio", total_metrics["clip_ratio"])
        self.log("train/completion_length", gen_data["avg_completion_length"])
        self.log("train/response_len_mean", gen_data.get("response_len_mean", gen_data["avg_completion_length"]))
        self.log("train/response_len_max", gen_data.get("response_len_max", 0.0))
        self.log("train/entropy_mean", gen_data.get("entropy_mean", 0.0))

        # Log reward components
        for key, val in gen_data.get("reward_components", {}).items():
            self.log(f"train/reward_{key}", val)

        # ── Stability diagnostics ────────────────────────────────────────────
        self.log("stability/degenerate_group_rate", gen_data.get("degenerate_rate", 0.0))
        self._log_stability_metrics()

        # ── Stdout/JSON metrics (visible in train.log via pipe) ──────────────
        step = self.global_step
        rc = gen_data.get("reward_components", {})
        completions = gen_data.get("_completion_texts", [])
        unique_ratio = self._completion_unique_ratio(completions)
        self.log("train/unique_completion_ratio", unique_ratio)

        if step % self.config.log_interval == 0:
            component_text = self._component_text(rc)
            if self._is_rank0():
                print(
                    f"[GRPO] step={step} | "
                    f"loss={loss_val:.4f} | "
                    f"reward={gen_data['reward_mean']:.3f}±{gen_data['reward_std']:.3f} | "
                    f"range=[{gen_data.get('reward_min', 0.0):.3f},{gen_data.get('reward_max', 0.0):.3f}] | "
                    f"{component_text} | "
                    f"comp_len={gen_data['avg_completion_length']:.0f} | "
                    f"H={gen_data.get('entropy_mean', 0.0):.2f} | "
                    f"clip={total_metrics.get('clip_ratio', 0.0):.2f} "
                    f"degen={gen_data.get('degenerate_rate', 0):.2f} "
                    f"uniq={unique_ratio:.2f}",
                    flush=True,
                )
            self._log_json_event("rl_step", step, {
                "loss": {
                    "total": loss_val,
                    "policy": total_metrics.get("policy_loss", 0.0),
                    "kl_raw": total_metrics.get("kl_raw", ref_energy_kl),
                    "kl_term": total_metrics.get("kl_term"),
                    "ref_energy_kl": ref_energy_kl,
                    "clip_ratio": total_metrics.get("clip_ratio", 0.0),
                    "reinforce_surrogate": total_metrics.get("reinforce_surrogate"),
                },
                "reward": {
                    "mean": gen_data["reward_mean"],
                    "std": gen_data["reward_std"],
                    "min": gen_data.get("reward_min"),
                    "max": gen_data.get("reward_max"),
                    "zero_frac": gen_data.get("reward_zero_frac"),
                    "var": gen_data["reward_var"],
                    "advantage_var": gen_data["advantage_var"],
                    "components": rc,
                },
                "rollout": {
                    "completion_len_mean": gen_data["avg_completion_length"],
                    "response_len_mean": gen_data.get("response_len_mean"),
                    "response_len_max": gen_data.get("response_len_max"),
                    "entropy_mean": gen_data.get("entropy_mean"),
                    "unique_completion_ratio": unique_ratio,
                    "degenerate_group_rate": gen_data.get("degenerate_rate", 0.0),
                    "skipped_step": 0.0,
                },
                "energy": {
                    "current_mean": total_metrics.get("energy_mean"),
                    "old_mean": total_metrics.get("old_energy_mean"),
                    "ref_energy_drift": total_metrics.get("energy_drift"),
                    "ref_energy_kl": ref_energy_kl,
                    "ratio_mean": total_metrics.get("ratio_mean"),
                    "ratio_std": total_metrics.get("ratio_std"),
                    "ratio_min": total_metrics.get("ratio_min"),
                    "ratio_max": total_metrics.get("ratio_max"),
                    "log_ratio_mean": total_metrics.get("log_ratio_mean"),
                    "log_ratio_std": total_metrics.get("log_ratio_std"),
                    "clip_frac": total_metrics.get("clip_frac"),
                    "approx_kl_k1": total_metrics.get("approx_kl_k1"),
                    "approx_kl_k3": total_metrics.get("approx_kl_k3"),
                    "log_ratio_abs_max": total_metrics.get("log_ratio_abs_max"),
                    "log_ratio_clamp_rate": total_metrics.get("log_ratio_clamp_rate"),
                    "effective_clip_rate": total_metrics.get("effective_clip_rate"),
                    "old_policy_energy_drift": total_metrics.get("old_policy_energy_drift"),
                    "clip_ratio_low": total_metrics.get("clip_ratio_low"),
                    "clip_ratio_high": total_metrics.get("clip_ratio_high"),
                },
            })

        # ── Trajectory logging & collapse detection ──────────────────────────
        self._maybe_init_traj_logger()
        if self._traj_logger is not None:
            rewards_list = gen_data.get("_rewards", [])
            energies_list = gen_data.get("_old_energies", [])
            adv_list = gen_data.get("_advantages", [])
            lens_list = gen_data.get("_completion_masks_lens", [])
            prompts_list = gen_data.get("_puzzles", [])
            ground_truths_list = gen_data.get("_solutions", prompts_list)

            # Periodic stdout + JSONL dump
            self._traj_logger.maybe_log(
                step=step,
                prompts=prompts_list,
                completions=completions,
                rewards=rewards_list,
                ground_truths=ground_truths_list,
                energies=energies_list,
                advantages=adv_list,
                ref_energy_kls=[ref_energy_kl] * len(completions),
                response_lens=lens_list,
                per_sample_metrics=gen_data.get("_reward_details"),
            )

            # Collapse detection
            self._traj_logger.detect_collapse(
                step=step,
                rewards=rewards_list,
                unique_completion_ratio=unique_ratio,
                prompts=prompts_list,
                completions=completions,
                ground_truths=ground_truths_list,
                energies=energies_list,
            )
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
        t0 = time.time()
        self._log_phase(
            "generate_start",
            num_prompts=int(num_prompts),
            num_generations=int(self.config.num_generations),
        )
        self.model.eval()
        # [DEBUG-RL] Reset debug flags for fresh logging in each phase.
        self.model._dbg_logged_embed_stats = False
        self.model._dbg_logged_logps_stats = False
        # Disable bf16 autocast during rollout: the EBT transformer's energy
        # head can overflow bf16 range, producing NaN/Inf that cascades to
        # corrupt logits and multinomial crashes. fp32 rollout is safe — the
        # generation batch is small and sequential (no backward graph).
        with torch.amp.autocast('cuda', enabled=False):
            rollout_out = generate_completions(
                model=self.model,
                prompt_ids=prompt_ids,
                tokenizer=self.tokenizer,
                hparams=self._gen_hparams,
                num_generations=self.config.num_generations,
                max_completion_length=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                generation_batch_size=self.config.generation_batch_size,
                return_stats=True,
            )
            completion_ids, completion_texts, completion_masks, rollout_stats = rollout_out
        self.model.train()
        self._log_phase(
            "generate_done",
            elapsed_s=round(time.time() - t0, 3),
            num_completions=len(completion_texts),
        )

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
        if self._is_rank0() and not getattr(self, '_dbg_logged_first_rollout', False):
            self._dbg_logged_first_rollout = True
            sample_text = completion_texts[0][:300] if completion_texts else "(empty)"
            print(f"[RL-SAMPLE] first_completion={repr(sample_text)}", flush=True)
            print(f"[RL-SAMPLE] first_metrics={reward_details[0]}", flush=True)

        # ── 3. Compute old energies / logprobs ───────────────────────────────
        expanded_prompts = prompt_ids.repeat_interleave(
            self.config.num_generations, dim=0
        )
        full_ids = torch.cat([expanded_prompts, completion_ids], dim=1)

        # Energy-based path (for energy_gspo and energy_reinforce).
        # P0b: keep autocast disabled to match _loss_energy_*'s autocast(False)
        # block. Mismatched precision causes systematic bias in (curr - old).
        energy_t0 = time.time()
        self._log_phase("old_energy_start", seqs=int(full_ids.shape[0]), seq_len=int(full_ids.shape[1]))
        with torch.amp.autocast('cuda', enabled=False):
            old_energies = compute_sequence_energy(
                self.model, full_ids, prompt_len, completion_masks
            )
        self._log_phase(
            "old_energy_done",
            elapsed_s=round(time.time() - energy_t0, 3),
            energy_mean=round(float(old_energies.mean().detach().cpu().item()), 5),
        )

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
        for key in ["format", "clue_preservation", "blank_accuracy", "constraint_validity", "full_solve"]:
            reward_components[key] = sum(d[key] for d in reward_details) / len(reward_details)
        advantage_var = advantages.var(unbiased=False).item() if advantages.numel() > 1 else 0.0
        reward_var = float(rewards.var(unbiased=False).item()) if rewards.numel() > 1 else 0.0
        reward_std = rewards.std().item()
        reward_zero_frac = float((rewards == 0).float().mean().item())
        response_lengths = completion_masks.sum(dim=1).float()

        return {
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "completion_masks": completion_masks,
            "old_energies": old_energies.detach(),
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "reward_mean": rewards.mean().item(),
            "reward_std": reward_std,
            "reward_min": rewards.min().item(),
            "reward_max": rewards.max().item(),
            "reward_zero_frac": reward_zero_frac,
            "reward_var": reward_var,
            "advantage_var": advantage_var,
            "avg_completion_length": avg_comp_len,
            "response_len_mean": response_lengths.mean().item(),
            "response_len_max": response_lengths.max().item(),
            "entropy_mean": float(rollout_stats.get("entropy_mean", 0.0)),
            "reward_components": reward_components,
            "degenerate_rate": degenerate_rate,
            # Raw arrays carried through for trajectory logging in training_step.
            "_completion_texts": completion_texts,
            "_rewards": rewards.detach().cpu().tolist(),
            "_old_energies": old_energies.detach().cpu().tolist(),
            "_advantages": advantages.detach().cpu().tolist(),
            "_completion_masks_lens": completion_masks.sum(dim=1).long().cpu().tolist(),
            "_reward_details": reward_details,
            "_puzzles": list(expanded_puzzles),
            "_solutions": list(expanded_solutions),
            "_prompt_ids_per_seq": expanded_prompts.cpu().tolist(),
        }

    def _manual_gspo_rollout_updates(self, gen_data):
        """Reuse one rollout for multiple clipped GSPO optimizer updates.

        This mirrors verl's ppo_epochs idea but is restricted to energy_gspo,
        where old_energies provide a sequence-level proximal anchor.
        """
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        total_loss = torch.tensor(0.0, device=self.device)
        total_metrics = {"policy_loss": 0.0, "kl": 0.0, "clip_ratio": 0.0}
        completed = 0

        for update_epoch in range(self.config.gspo_update_epochs):
            optimizer.zero_grad()
            loss, metrics = self._compute_grpo_loss(gen_data, update_epoch)
            if not torch.isfinite(loss):
                self.log("stability/nonfinite_loss", 1.0)
                self._log_json_event("gspo_update_skipped", self.global_step, {
                    "update_epoch": update_epoch,
                    "reason": "nonfinite_loss",
                    "loss": str(loss.detach().cpu().item()),
                    "reward": {
                        "mean": gen_data.get("reward_mean"),
                        "std": gen_data.get("reward_std"),
                        "components": gen_data.get("reward_components", {}),
                    },
                })
                optimizer.zero_grad()
                break

            self.manual_backward(loss)
            grad_metrics = self._sanitize_log_and_clip_gradients()
            optimizer.step()
            if scheduler is not None:
                if isinstance(scheduler, (list, tuple)):
                    for sch in scheduler:
                        sch.step()
                else:
                    scheduler.step()

            completed += 1
            total_loss = total_loss + loss.detach()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

            if self.global_step % self.config.log_interval == 0:
                self._log_json_event("gspo_update_epoch", self.global_step, {
                    "update_epoch": update_epoch + 1,
                    "update_epochs": self.config.gspo_update_epochs,
                    "loss": {"total": loss.detach().item(), **metrics},
                    "grad": grad_metrics,
                })

        if completed == 0:
            return None, total_metrics
        total_loss = total_loss / completed
        for k in list(total_metrics.keys()):
            total_metrics[k] = total_metrics[k] / completed
        total_metrics["gspo_completed_update_epochs"] = float(completed)
        return total_loss, total_metrics

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

    def _energy_kl_anchor(self, full_ids, prompt_len, current_energies, completion_masks=None):
        """Sequence-energy anchor against the frozen SFT reference.

        The earlier one-sided anchor only punished ``E_theta > E_ref`` and
        allowed large negative drift. The analyzed Sudoku/GSM8K runs collapsed
        exactly in that unconstrained direction, so the default is now a smooth
        two-sided Huber penalty. This keeps the policy close to the SFT energy
        scale while GSPO clipping handles per-rollout policy drift.
        """
        if self.ref_model is None or self.config.beta <= 0.0:
            return None, 0.0, 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            ref_energies = compute_sequence_energy(
                self.ref_model, full_ids, prompt_len, completion_masks
            )
        energy_diff = current_energies - ref_energies.detach()
        mode = getattr(self.config, "energy_kl_mode", "symmetric_huber")
        if mode == "one_sided":
            energy_kl = energy_diff.clamp(min=0.0).mean()
        elif mode == "symmetric_l2":
            energy_kl = energy_diff.pow(2).mean()
        elif mode == "symmetric_huber":
            target = torch.zeros_like(energy_diff)
            energy_kl = F.smooth_l1_loss(
                energy_diff,
                target,
                beta=getattr(self.config, "energy_kl_huber_delta", 0.5),
                reduction="mean",
            )
        else:
            raise ValueError(f"Unknown energy_kl_mode: {mode}")
        return energy_kl, energy_kl.item(), energy_diff.mean().item()

    def _loss_energy_gspo(self, gen_data):
        """Energy-GSPO: sequence-level energy ratio with PPO clipping.

        Logged health metrics:
        - train/gspo_policy_loss: unclipped surrogate after PPO max branch. Near
          zero on the first update epoch is expected because group advantages
          have mean zero; gradients still flow through the straight-through
          ratio.
        - train/gspo_kl_raw: unweighted E_theta-vs-reference energy anchor.
          This is not token-probability KL; it is a sequence-energy drift
          penalty. Keep beta*gspo_kl_raw below the policy-loss scale.
        - train/gspo_approx_kl_k1/k3: old-policy drift estimates from the
          sequence ratio. Sustained k3 above ~1e-3 with tight GSPO clips means
          rollout reuse is drifting too far.
        - train/gspo_ratio_* and train/gspo_clip_frac: sequence-level ratio
          diagnostics. With clip low/high around 3e-4/4e-4, clip_frac should be
          nonzero on reused update epochs and near zero only on exactly
          on-policy first epochs.

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
            current_energies = compute_sequence_energy(
                self.model, full_ids, prompt_len, completion_masks
            )

        # Sequence-level importance ratio, adapted from verl's GSPO loss.
        # Treat -E as a sequence log-prob surrogate. The forward ratio is the
        # rollout-vs-current sequence ratio, while the gradient flows through
        # `-current_energies - sg(-current_energies)`, matching verl's
        # `log_prob - log_prob.detach() + seq_ratio.detach()` pattern.
        raw_log_ratio = -(current_energies - old_energies)
        log_ratio_for_grad = (
            -current_energies + current_energies.detach() + raw_log_ratio.detach()
        )
        log_ratio = torch.clamp(log_ratio_for_grad, min=-20.0, max=10.0)
        ratio = torch.exp(log_ratio)

        clip_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.epsilon
        clip_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.epsilon
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * clipped_ratio
        policy_loss = torch.maximum(pg_losses1, pg_losses2).mean()

        # KL anchor on top of PPO clip. PPO clip bounds per-step ratio drift,
        # the KL term bounds cumulative drift from SFT ref. They're orthogonal.
        kl_term, kl_value, energy_drift = self._energy_kl_anchor(
            full_ids, prompt_len, current_energies, completion_masks)
        if kl_term is not None:
            kl_weighted = self.config.beta * kl_term
            loss = policy_loss + kl_weighted
            self.log("train/ref_energy_kl", kl_value)
            self.log("train/ref_energy_drift", energy_drift)
        else:
            kl_weighted = torch.zeros_like(policy_loss)
            loss = policy_loss

        # P1a: log_ratio clamp visibility.
        raw_log_ratio_det = raw_log_ratio.detach()
        ratio_det = ratio.detach()
        advantages_det = advantages.detach()
        log_ratio_abs_max = raw_log_ratio_det.abs().max().item()
        log_ratio_clamp_rate = ((raw_log_ratio_det < -20.0) | (raw_log_ratio_det > 10.0)).float().mean().item()
        # P1b: effective clip rate = how often clipped branch is selected.
        effective_clip_rate = (pg_losses2 > pg_losses1).float().mean().item()
        clip_frac = ((ratio_det < (1.0 - clip_low)) | (ratio_det > (1.0 + clip_high))).float().mean().item()
        old_policy_energy_drift = (current_energies - old_energies).mean().item()
        approx_kl_k1 = (-raw_log_ratio_det).mean().item()
        approx_kl_k3 = ((ratio_det - 1.0) - raw_log_ratio_det).mean().item()

        # Diagnostic: equivalent REINFORCE loss surrogate.
        # When num_iterations=1 the on-policy ratio is ≡1 → policy_loss is
        # mathematically -mean(A) = 0 (A is group-relative z-score with
        # mean=0). The displayed `loss=0` is a numerical artefact; actual
        # gradients are non-zero. Logging `mean(A * E)` gives a visible,
        # non-zero signal whose magnitude tracks the true update direction
        # — same form as the REINFORCE branch — so users can monitor
        # learning progress in GSPO mode despite the misleading zero loss.
        # `.detach()` avoids touching the autograd graph.
        reinforce_surrogate = (
            advantages.detach() * current_energies.detach()
        ).mean().item()

        total_loss_value = loss.detach().item()
        policy_loss_value = policy_loss.detach().item()
        kl_weighted_value = kl_weighted.detach().item()

        self.log("train/energy_mean", current_energies.mean().item())
        self.log("train/gspo_policy_loss", policy_loss_value)
        self.log("train/gspo_kl_raw", kl_value)
        self.log("train/gspo_kl_term", kl_weighted_value)
        self.log("train/gspo_total_loss", total_loss_value)
        self.log("train/gspo_ratio_mean", ratio_det.mean().item())
        self.log("train/gspo_ratio_std", ratio_det.std(unbiased=False).item() if ratio_det.numel() > 1 else 0.0)
        self.log("train/gspo_ratio_min", ratio_det.min().item())
        self.log("train/gspo_ratio_max", ratio_det.max().item())
        self.log("train/gspo_log_ratio_mean", raw_log_ratio_det.mean().item())
        self.log("train/gspo_log_ratio_std", raw_log_ratio_det.std(unbiased=False).item() if raw_log_ratio_det.numel() > 1 else 0.0)
        self.log("train/gspo_clip_frac", clip_frac)
        self.log("train/gspo_effective_clip_frac", effective_clip_rate)
        self.log("train/gspo_approx_kl_k1", approx_kl_k1)
        self.log("train/gspo_approx_kl_k3", approx_kl_k3)
        self.log("train/gspo_adv_mean", advantages_det.mean().item())
        self.log("train/gspo_adv_std", advantages_det.std(unbiased=False).item() if advantages_det.numel() > 1 else 0.0)
        self.log("train/gspo_adv_abs_mean", advantages_det.abs().mean().item())
        self.log("train/ratio_mean", ratio_det.mean().item())
        self.log("train/reinforce_surrogate", reinforce_surrogate)
        self.log("stability/log_ratio_abs_max", log_ratio_abs_max)
        self.log("stability/log_ratio_clamp_rate", log_ratio_clamp_rate)
        self.log("train/effective_clip_rate", effective_clip_rate)

        if self.global_step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO-EXT] step={self.global_step} "
                    f"loss={total_loss_value:+.6f} "
                    f"policy={policy_loss_value:+.6f} "
                    f"kl_term={kl_weighted_value:.6f} "
                    f"ratio_mean={ratio_det.mean().item():.4f} "
                    f"ratio_std={ratio_det.std(unbiased=False).item() if ratio_det.numel() > 1 else 0.0:.4f} "
                    f"log_ratio_abs_max={log_ratio_abs_max:.4f} "
                    f"clip_frac={clip_frac:.3f} "
                    f"effective_clip={effective_clip_rate:.3f} "
                    f"approx_kl_k3={approx_kl_k3:.6e} "
                    f"old_policy_energy_drift={old_policy_energy_drift:+.4f} "
                    f"ref_energy_kl={kl_value:.4f} "
                    f"reinforce_surrogate={reinforce_surrogate:+.6f}",
                    flush=True,
                )

        return loss, {
            "policy_loss": policy_loss.item(),
            "kl_raw": kl_value,
            "kl_term": kl_weighted_value,
            "total_loss": total_loss_value,
            "ref_energy_kl": kl_value,
            "clip_ratio": effective_clip_rate,
            "effective_clip_rate": effective_clip_rate,
            "clip_frac": clip_frac,
            "energy_mean": current_energies.mean().item(),
            "old_energy_mean": old_energies.mean().item(),
            "ref_energy_drift": energy_drift,
            "ratio_mean": ratio_det.mean().item(),
            "ratio_std": ratio_det.std(unbiased=False).item() if ratio_det.numel() > 1 else 0.0,
            "ratio_min": ratio_det.min().item(),
            "ratio_max": ratio_det.max().item(),
            "log_ratio_mean": raw_log_ratio_det.mean().item(),
            "log_ratio_std": raw_log_ratio_det.std(unbiased=False).item() if raw_log_ratio_det.numel() > 1 else 0.0,
            "approx_kl_k1": approx_kl_k1,
            "approx_kl_k3": approx_kl_k3,
            "log_ratio_abs_max": log_ratio_abs_max,
            "log_ratio_clamp_rate": log_ratio_clamp_rate,
            "old_policy_energy_drift": old_policy_energy_drift,
            "clip_ratio_low": clip_low,
            "clip_ratio_high": clip_high,
            "reinforce_surrogate": reinforce_surrogate,
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
        completion_masks = gen_data["completion_masks"].float()
        advantages = gen_data["advantages"]

        with torch.amp.autocast('cuda', enabled=False):
            current_energies = compute_sequence_energy(
                self.model, full_ids, prompt_len, completion_masks
            )

        # Energies are already per-token mean (logprobs.py:94), so no length
        # normalisation needed here.
        policy_loss = (advantages * current_energies).mean()

        # One-sided energy-KL anchor (shared with _loss_energy_gspo).
        kl_term, kl_value, energy_drift = self._energy_kl_anchor(
            full_ids, prompt_len, current_energies, completion_masks)
        if kl_term is not None:
            loss = policy_loss + self.config.beta * kl_term
            self.log("train/ref_energy_kl", kl_value)
            self.log("train/ref_energy_drift", energy_drift)
        else:
            loss = policy_loss

        cur_energy_mean = current_energies.mean().item()
        self.log("train/energy_mean", cur_energy_mean)

        if self.global_step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO-EXT] step={self.global_step} "
                    f"ref_energy_kl={kl_value:.4f} ref_energy_drift={energy_drift:+.4f} "
                    f"energy_mean={cur_energy_mean:+.4f} policy_loss={policy_loss.item():+.4f}",
                    flush=True,
                )

        return loss, {
            "policy_loss": policy_loss.item(),
            "ref_energy_kl": kl_value,
            "clip_ratio": 0.0,
            "energy_mean": cur_energy_mean,
            "old_energy_mean": gen_data["old_energies"].mean().item(),
            "ref_energy_drift": energy_drift,
        }

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
            self.log("stability/skipped_step", 1.0)
            return self._zero_placeholder_loss(), {"policy_loss": 0.0, "kl": 0.0, "clip_ratio": 0.0}
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

    def _sanitize_log_and_clip_gradients(self):
        total_norm = 0.0
        alpha_grad = None
        max_param_grad_norm = 0.0
        nan_grad_params = 0

        for name, p in self.model.named_parameters():
            if p.grad is not None:
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

        step = self.global_step
        if nan_grad_params > 0:
            import os as _os
            try:
                nan_debug_interval = int(_os.environ.get("NAN_GRAD_DEBUG_INTERVAL", "0"))
            except ValueError:
                nan_debug_interval = 0
            if nan_debug_interval > 0 and step % nan_debug_interval == 0:
                _rank = _os.environ.get('LOCAL_RANK', '?')
                print(
                    f"[DBG-RL][rank={_rank}] Sanitized NaN grads in {nan_grad_params} params "
                    f"(total_norm_after={total_norm:.4e})",
                    flush=True,
                )

        if step % self.config.log_interval == 0:
            import os as _os
            if _os.environ.get('LOCAL_RANK', '0') == '0':
                print(
                    f"[GRPO] step={step} grad_norm={total_norm:.4e} "
                    f"max_param_grad={max_param_grad_norm:.4e} nan_params={nan_grad_params}",
                    flush=True,
                )

        if self.config.max_grad_per_param > 0.0:
            clip_val = self.config.max_grad_per_param
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-clip_val, clip_val)

        return {
            "grad_norm_before_clip": total_norm,
            "max_param_grad_norm": max_param_grad_norm,
            "nan_grad_params": float(nan_grad_params),
        }

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms, sanitize NaN grads, and apply per-parameter clipping."""
        if getattr(self, "_skip_optim_step", False):
            for p in self.model.parameters():
                p.grad = None
            self._skip_optim_step = False
            self.log("stability/grad_norm_before_clip", 0.0)
            self.log("stability/max_param_grad_norm", 0.0)
            self.log("stability/nan_grad_params", 0.0)
            self._log_phase("optimizer_step_skipped")
            return
        metrics = self._sanitize_log_and_clip_gradients()
        self._log_phase(
            "backward_done",
            grad_norm=round(float(metrics.get("grad_norm_before_clip", 0.0)), 6),
            max_grad=round(float(metrics.get("max_param_grad_norm", 0.0)), 6),
            nan_grad_params=int(metrics.get("nan_grad_params", 0.0)),
        )

    def configure_optimizers(self):
        """Dispatch to the configured optimizer kind."""
        if getattr(self.config, "rl_optimizer", "adamw") == "muon_adamw":
            return self._configure_muon_adamw_optimizer()
        return self._configure_adamw_optimizer()

    def _build_lr_lambda(self):
        """Linear warmup → cosine decay schedule (shared by both optimizer paths)."""
        import math

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    def _configure_adamw_optimizer(self):
        """AdamW optimizer with param-group LRs + linear warmup + cosine decay.

        v3 P0: Multi-group LR inherits SFT design philosophy (different
        parameter families need different effective LRs):
          - transformer   : base_lr        (transformer body)
          - vocab_to_embed: base_lr * 0.5  (54M output projection; halve to
                                            slow hidden drift seen in v2)
          - scalar/alpha  : base_lr * 2.0  (alpha is frozen, but kept for any
                                            future scalar params)
          - other         : base_lr        (catch-all)

        Embedding is frozen in __init__ (v3 C1) so receives no entry here.
        """
        base_lr = self.config.learning_rate
        transformer_lr    = base_lr
        vocab_to_embed_lr = base_lr * 0.5
        scalar_lr         = base_lr * 2.0

        transformer_params, v2e_params, scalar_params, other_params = [], [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("vocab_to_embed.") or ".vocab_to_embed." in n:
                v2e_params.append(p)
            elif p.ndim == 0 or n.endswith(".alpha") or n == "alpha":
                scalar_params.append(p)
            elif n.startswith("transformer.") or ".transformer." in n:
                transformer_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if transformer_params:
            param_groups.append({"params": transformer_params, "lr": transformer_lr,
                                 "group_name": "transformer"})
        if v2e_params:
            param_groups.append({"params": v2e_params, "lr": vocab_to_embed_lr,
                                 "group_name": "vocab_to_embed"})
        if scalar_params:
            param_groups.append({"params": scalar_params, "lr": scalar_lr,
                                 "group_name": "scalar"})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr,
                                 "group_name": "other"})

        import os as _os
        if _os.environ.get("LOCAL_RANK", "0") == "0":
            print("[INFO] Optimizer param-groups:", flush=True)
            for g in param_groups:
                n_params = sum(p.numel() for p in g["params"])
                print(f"  {g['group_name']:<16} lr={g['lr']:.2e}  n_params={n_params:,}",
                      flush=True)

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Linear warmup then cosine decay (shared with _configure_muon_adamw_optimizer).
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._build_lr_lambda())

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _configure_muon_adamw_optimizer(self):
        """Muon (transformer matrices) + AdamW (everything else) — RL variant.

        Mirrors openebm.elm.trainer._configure_muon_adamw_optimizer (used in
        SFT) but is simplified for the RL setting:
          - alpha / embeddings / langevin_noise are already frozen in __init__
            (requires_grad=False), so they auto-skip; no group is created.
          - vocab_to_embed → AdamW, lr = base_lr * 0.5 (matches v3 P0 ratio).
          - transformer matrices (ndim>=2) → Muon, grouped by shape.
          - transformer scalars (ndim<2) + any other trainable params → AdamW,
            lr = base_lr * 2.0 (matches v3 P0 ratio).
        """
        from nanochat.optim import MuonAdamW

        base_lr = self.config.learning_rate
        v2e_lr = (
            self.config.adamw_vocab_to_embed_lr
            if self.config.adamw_vocab_to_embed_lr > 0
            else base_lr * 0.5
        )
        scalar_lr = (
            self.config.adamw_scalar_lr
            if self.config.adamw_scalar_lr > 0
            else base_lr * 2.0
        )
        other_lr = (
            self.config.adamw_other_lr
            if self.config.adamw_other_lr > 0
            else base_lr
        )
        muon_lr = self.config.muon_lr
        adam_betas = (0.9, 0.95)

        v2e_params, scalar_params, matrix_params, other_params = [], [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("vocab_to_embed.") or ".vocab_to_embed." in n:
                v2e_params.append(p)
            elif n.startswith("transformer.") or ".transformer." in n:
                if p.ndim >= 2:
                    matrix_params.append(p)
                else:
                    scalar_params.append(p)
            else:
                # alpha / scalars outside transformer → AdamW scalar lane
                other_params.append(p)

        param_groups = []
        if v2e_params:
            param_groups.append(dict(
                kind='adamw', params=v2e_params,
                lr=v2e_lr, betas=adam_betas, eps=1e-10,
                weight_decay=0.0,
            ))
        if scalar_params:
            param_groups.append(dict(
                kind='adamw', params=scalar_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10,
                weight_decay=0.0,
            ))
        if other_params:
            param_groups.append(dict(
                kind='adamw', params=other_params,
                lr=other_lr, betas=adam_betas, eps=1e-10,
                weight_decay=self.config.weight_decay,
            ))

        # Muon needs same-shape stacking — group transformer matrices by shape.
        shape_groups = {}
        for p in matrix_params:
            shape_groups.setdefault(p.shape, []).append(p)
        for shape in sorted(shape_groups.keys(), key=lambda s: tuple(s)):
            param_groups.append(dict(
                kind='muon', params=shape_groups[shape],
                lr=muon_lr,
                momentum=self.config.muon_momentum,
                ns_steps=self.config.muon_ns_steps,
                beta2=self.config.muon_beta2,
                weight_decay=self.config.weight_decay,
            ))

        # Muon stacks grads by shape, so missing-grad groups must be skipped as
        # groups. Zero-fill is not a no-op under Muon momentum/weight decay.
        PLMuonAdamW = make_skip_missing_grad_muon_adamw(MuonAdamW)
        optimizer = PLMuonAdamW(param_groups)
        for g in optimizer.param_groups:
            g['initial_lr'] = g['lr']

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._build_lr_lambda())

        import os as _os
        if _os.environ.get("LOCAL_RANK", "0") == "0":
            n_muon = sum(p.numel() for p in matrix_params)
            n_v2e = sum(p.numel() for p in v2e_params)
            n_scalar = sum(p.numel() for p in scalar_params)
            n_other = sum(p.numel() for p in other_params)
            n_total = n_muon + n_v2e + n_scalar + n_other
            print("=" * 80, flush=True)
            print("[Muon+AdamW] RL hybrid optimizer enabled:", flush=True)
            print(f"  AdamW base/fallback LR: {base_lr:.2e} (Muon matrices use separate muon_lr)", flush=True)
            print(f"  Muon  groups (by shape): {len(shape_groups)}", flush=True)
            print(f"  Muon  params: {n_muon:,} ({n_muon/max(1,n_total)*100:.1f}%) lr={muon_lr:.2e}",
                  flush=True)
            print(f"  AdamW vocab_to_embed:    {n_v2e:,} lr={v2e_lr:.2e}", flush=True)
            print(f"  AdamW transformer scalar:{n_scalar:,} lr={scalar_lr:.2e}", flush=True)
            print(f"  AdamW other (catch-all): {n_other:,} lr={other_lr:.2e}", flush=True)
            print(f"  Muon  momentum={self.config.muon_momentum} "
                  f"ns_steps={self.config.muon_ns_steps} beta2={self.config.muon_beta2}",
                  flush=True)
            print("=" * 80, flush=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

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
