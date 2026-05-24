"""Trajectory logging & collapse-warning utilities for EBM-GRPO RL.

Designed to be imported and used from `EBMGRPOTrainer._generate_and_score`.

Features (controlled by env vars / config):
  * **Periodic stdout sample dump** every N steps (`DBG-RL-PERIODIC`).
  * **JSONL trajectory persistence** under `${OUTPUT_DIR}/trajectories/step_<step>/rollout_samples.jsonl`.
  * **summary.json** every K periodic dumps with reward / length / unique-completion stats.
  * **Collapse warning + dump**: when `unique_completion_ratio < THRESH` or
    `reward_std < EPS` for `K` consecutive steps, write the entire batch's
    rollouts to `collapse_dump/step_<step>.jsonl` and emit `WARN-COLLAPSE`.

Kept dependency-free (only stdlib + torch numerics) so it can run on rank-0
without perturbing DDP collectives.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


# ── log-level helpers ────────────────────────────────────────────────────────
_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _level() -> int:
    return _LEVELS.get(os.environ.get("LOG_LEVEL", "INFO").upper(), 20)


def _is_rank0() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("RANK", "0") == "0"


def logp(prefix: str, msg: str, min_level: int = 20) -> None:
    """Rank-0-only printer with [PREFIX] tagging and LOG_LEVEL filter."""
    if not _is_rank0():
        return
    if _level() > min_level:
        return
    print(f"[{prefix}] {msg}", flush=True)


# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class TrajLoggerConfig:
    output_dir: str
    sample_every_n_steps: int = 50
    samples_per_dump: int = 3
    summary_every_n_steps: int = 500
    persist_jsonl: bool = True
    # Collapse detection
    unique_ratio_threshold: float = 0.3
    reward_std_eps: float = 1e-3
    collapse_window: int = 3   # K consecutive bad steps
    enabled: bool = True

    @classmethod
    def from_env(cls, output_dir: str, **overrides) -> "TrajLoggerConfig":
        """Build from env vars (TRAJ_*). Explicit kwargs take precedence."""
        def _envf(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v is not None and v != "" else default
        def _envi(name: str, default: int) -> int:
            v = os.environ.get(name)
            return int(v) if v is not None and v != "" else default
        def _envb(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.lower() in ("1", "true", "yes")
        cfg = cls(
            output_dir=output_dir,
            sample_every_n_steps=_envi("TRAJ_SAMPLE_EVERY", 50),
            samples_per_dump=_envi("TRAJ_SAMPLES_PER_DUMP", 3),
            summary_every_n_steps=_envi("TRAJ_SUMMARY_EVERY", 500),
            persist_jsonl=_envb("TRAJ_PERSIST_JSONL", True),
            unique_ratio_threshold=_envf("TRAJ_UNIQUE_RATIO_THRESH", 0.3),
            reward_std_eps=_envf("TRAJ_REWARD_STD_EPS", 1e-3),
            collapse_window=_envi("TRAJ_COLLAPSE_WINDOW", 3),
            enabled=_envb("TRAJ_ENABLED", True),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ── Logger ──────────────────────────────────────────────────────────────────
class TrajectoryLogger:
    """Per-trainer instance trajectory logger. All disk IO is rank-0 only."""

    def __init__(self, cfg: TrajLoggerConfig):
        self.cfg = cfg
        self._collapse_history = deque(maxlen=cfg.collapse_window)

        if cfg.enabled and _is_rank0() and cfg.persist_jsonl:
            os.makedirs(os.path.join(cfg.output_dir, "trajectories"), exist_ok=True)
            os.makedirs(os.path.join(cfg.output_dir, "collapse_dump"), exist_ok=True)

    # ─── Public API ──────────────────────────────────────────────────────

    def should_dump(self, step: int) -> bool:
        if not self.cfg.enabled:
            return False
        return step == 0 or (step % self.cfg.sample_every_n_steps == 0)

    def maybe_log(
        self,
        step: int,
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Sequence[float],
        ground_truths: Sequence[Any],
        *,
        energies: Optional[Sequence[float]] = None,
        advantages: Optional[Sequence[float]] = None,
        kls: Optional[Sequence[float]] = None,
        response_lens: Optional[Sequence[int]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Periodic stdout + JSONL dump. No-op for non-rank-0 ranks."""
        if not (self.cfg.enabled and _is_rank0()):
            return
        if not self.should_dump(step):
            return

        records = self._build_records(
            step, prompts, completions, rewards, ground_truths,
            energies=energies, advantages=advantages, kls=kls,
            response_lens=response_lens, extra=extra,
        )

        # 1) stdout sample dump (3 samples by default)
        self._print_samples(step, records, k=self.cfg.samples_per_dump)

        # 2) JSONL persistence
        if self.cfg.persist_jsonl:
            self._write_jsonl(step, records)

        # 3) periodic summary
        if step > 0 and step % self.cfg.summary_every_n_steps == 0:
            self._write_summary(step, records)

    def detect_collapse(
        self,
        step: int,
        rewards: Sequence[float],
        unique_completion_ratio: float,
        prompts: Optional[Sequence[str]] = None,
        completions: Optional[Sequence[str]] = None,
        ground_truths: Optional[Sequence[Any]] = None,
        energies: Optional[Sequence[float]] = None,
    ) -> bool:
        """Track collapse signals; on K consecutive hits, emit warning + dump.

        Returns True iff a collapse warning was just fired (this step).
        """
        if not (self.cfg.enabled and _is_rank0()):
            return False

        if rewards:
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / max(1, len(rewards))
            std = var ** 0.5
        else:
            std = 0.0

        bad = (unique_completion_ratio < self.cfg.unique_ratio_threshold) or (
            std < self.cfg.reward_std_eps
        )
        self._collapse_history.append(bad)

        if (
            len(self._collapse_history) >= self.cfg.collapse_window
            and all(self._collapse_history)
        ):
            logp(
                "WARN-COLLAPSE",
                f"step={step} unique_ratio={unique_completion_ratio:.3f} "
                f"reward_std={std:.4g} window={self.cfg.collapse_window} → DUMP",
                min_level=_LEVELS["WARN"],
            )
            self._dump_collapse(step, prompts, completions, rewards, ground_truths, energies)
            self._collapse_history.clear()
            return True
        return False

    # ─── Internals ──────────────────────────────────────────────────────

    def _build_records(
        self, step, prompts, completions, rewards, ground_truths,
        *, energies, advantages, kls, response_lens, extra,
    ) -> List[Dict[str, Any]]:
        n = len(completions)
        def _get(seq, i, default=None):
            return seq[i] if seq is not None and i < len(seq) else default
        recs = []
        for i in range(n):
            rec = {
                "step": int(step),
                "sample_idx": i,
                "prompt": _get(prompts, i, ""),
                "completion": _get(completions, i, ""),
                "reward": float(_get(rewards, i, 0.0)),
                "ground_truth": _get(ground_truths, i, None),
                "energy": _get(energies, i, None),
                "advantage": _get(advantages, i, None),
                "kl": _get(kls, i, None),
                "response_len": _get(response_lens, i, None),
            }
            if extra:
                rec.update(extra)
            recs.append(rec)
        return recs

    def _print_samples(self, step: int, records: List[Dict[str, Any]], k: int) -> None:
        if not records:
            return
        for i, rec in enumerate(records[:k]):
            comp = (rec["completion"] or "").replace("\n", "\\n")
            prompt = (rec["prompt"] or "").replace("\n", "\\n")
            gt = rec.get("ground_truth")
            energy = rec.get("energy")
            energy_s = f"{energy:.4f}" if isinstance(energy, (int, float)) else str(energy)
            logp(
                f"DBG-RL-PERIODIC step={step:>5} idx={i}",
                f"reward={rec['reward']:+.3f} energy={energy_s} "
                f"len={rec.get('response_len')} | gt={gt}",
            )
            logp(
                f"DBG-RL-PERIODIC step={step:>5} idx={i}",
                f"prompt={prompt[:160]}…" if len(prompt) > 160 else f"prompt={prompt}",
            )
            logp(
                f"DBG-RL-PERIODIC step={step:>5} idx={i}",
                f"compl ={comp[:300]}…" if len(comp) > 300 else f"compl ={comp}",
            )

    def _write_jsonl(self, step: int, records: List[Dict[str, Any]]) -> None:
        sub = os.path.join(self.cfg.output_dir, "trajectories", f"step_{step:06d}")
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, "rollout_samples.jsonl")
        with open(path, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def _write_summary(self, step: int, records: List[Dict[str, Any]]) -> None:
        rewards = [r["reward"] for r in records]
        lens = [r["response_len"] for r in records if r.get("response_len") is not None]
        completions = [r["completion"] for r in records]
        unique_ratio = (
            len(set(completions)) / max(1, len(completions)) if completions else 0.0
        )
        rmean = (sum(rewards) / len(rewards)) if rewards else 0.0
        rvar = (
            sum((x - rmean) ** 2 for x in rewards) / len(rewards) if rewards else 0.0
        )
        summary = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": len(records),
            "reward_mean": rmean,
            "reward_std": rvar ** 0.5,
            "reward_min": min(rewards) if rewards else None,
            "reward_max": max(rewards) if rewards else None,
            "len_mean": (sum(lens) / len(lens)) if lens else None,
            "len_max": max(lens) if lens else None,
            "unique_completion_ratio": unique_ratio,
        }
        path = os.path.join(
            self.cfg.output_dir, "trajectories", f"step_{step:06d}", "summary.json"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logp(
            "INFO",
            f"trajectory summary @ step={step}: "
            f"reward={summary['reward_mean']:.3f}±{summary['reward_std']:.3f} "
            f"unique={summary['unique_completion_ratio']:.2f} "
            f"len={summary['len_mean']}",
        )

    def _dump_collapse(
        self, step, prompts, completions, rewards, ground_truths, energies,
    ) -> None:
        path = os.path.join(self.cfg.output_dir, "collapse_dump", f"step_{step:06d}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = max(
            len(completions or []),
            len(rewards or []),
        )
        with open(path, "w") as f:
            for i in range(n):
                rec = {
                    "step": int(step),
                    "sample_idx": i,
                    "prompt": prompts[i] if prompts and i < len(prompts) else None,
                    "completion": completions[i] if completions and i < len(completions) else None,
                    "reward": float(rewards[i]) if rewards and i < len(rewards) else None,
                    "ground_truth": ground_truths[i] if ground_truths and i < len(ground_truths) else None,
                    "energy": energies[i] if energies and i < len(energies) else None,
                }
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        logp("WARN-COLLAPSE", f"dumped full batch to {path}", min_level=_LEVELS["WARN"])


# ── Aligned column logger for per-step metrics ──────────────────────────────
_HEADER = (
    f"{'step':>6} {'loss':>10} {'reward':>10} {'r_std':>8} {'energy_gap':>11} "
    f"{'kl':>10} {'lr':>10} {'gnorm':>9} {'degen':>6}"
)
_HEADER_INTERVAL = 20


class AlignedMetricsPrinter:
    """Stable, fixed-width columnar metrics output for the train.log."""

    def __init__(self):
        self._row_count = 0

    def emit(self, step, loss, reward, reward_std, energy_gap, kl, lr, gnorm, degen):
        if not _is_rank0():
            return
        if self._row_count % _HEADER_INTERVAL == 0:
            print("[INFO]    " + _HEADER, flush=True)
        line = (
            f"{step:>6d} {loss:>10.4f} {reward:>10.4f} {reward_std:>8.3f} "
            f"{energy_gap:>11.4f} {kl:>10.4f} {lr:>10.2e} {gnorm:>9.4f} "
            f"{degen:>6.2f}"
        )
        print("[METRIC] " + line, flush=True)
        self._row_count += 1
