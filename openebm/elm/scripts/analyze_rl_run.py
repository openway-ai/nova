#!/usr/bin/env python3
"""Analyze EBM RL train logs and generate reusable plots/reports.

Usage:
    python openebm/elm/scripts/analyze_rl_run.py /path/to/run_dir
    python openebm/elm/scripts/analyze_rl_run.py /path/to/train.log --out-dir /tmp/report

Outputs:
    analysis_report.md
    analysis_metrics.csv
    analysis_overview.png
    analysis_metrics.png
    analysis_stability.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
GRPO_RE = re.compile(
    r"\[GRPO\]\s+step=(?P<step>\d+)\s+\|\s+"
    r"loss=(?P<loss>-?[0-9.eE+-]+)\s+\|\s+"
    r"reward=(?P<reward>-?[0-9.eE+-]+)[+\-]?±(?P<reward_std>-?[0-9.eE+-]+)\s+\|\s+"
    r"(?P<comps>.*?)\s+\|\s+comp_len=(?P<comp_len>-?[0-9.eE+-]+)\s+\|\s+"
    r"clip=(?P<clip>-?[0-9.eE+-]+)\s+degen=(?P<degen>-?[0-9.eE+-]+)(?:\s+uniq=(?P<uniq>-?[0-9.eE+-]+))?"
)
GRAD_RE = re.compile(
    r"\[GRPO\]\s+step=(?P<step>\d+)\s+grad_norm=(?P<grad_norm>-?[0-9.eE+-]+)\s+"
    r"max_param_grad=(?P<max_param_grad>-?[0-9.eE+-]+)\s+nan_params=(?P<nan_params>-?[0-9.eE+-]+)"
)
EXT_RE = re.compile(
    r"\[GRPO-EXT\]\s+step=(?P<step>\d+).*?ratio_mean=(?P<ratio_mean>-?[0-9.eE+-]+).*?"
    r"log_ratio_abs_max=(?P<log_ratio_abs_max>-?[0-9.eE+-]+).*?clamp_rate=(?P<clamp_rate>-?[0-9.eE+-]+).*?"
    r"effective_clip=(?P<effective_clip>-?[0-9.eE+-]+).*?energy_ppo_kl=(?P<energy_ppo_kl>-?[0-9.eE+-]+).*?"
    r"reinforce_surrogate=(?P<reinforce_surrogate>-?[0-9.eE+-]+)"
)
CKPT_RE = re.compile(r"step=(?:step=)?(?P<step>\d+).*?reward=(?:val/reward_mean=)?(?P<reward>-?[0-9.]+)")

ERROR_PATTERNS = (
    "Traceback", "RuntimeError", "ValueError", "TypeError", "Exception", "ERROR",
    "out of memory", "CUDA error", "Killed", "SIGTERM", "SIGKILL", "ChildFailedError",
)


def parse_ts(line: str) -> Optional[datetime]:
    m = TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def flatten_components(components: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mapping = {
        "format": "format",
        "clue_preservation": "clue",
        "blank_accuracy": "blank_acc",
        "blank_acc": "blank_acc",
        "answer_acc": "answer_acc",
        "exact_match": "exact_match",
        "partial_credit": "partial_credit",
        "parse_rate": "parse_rate",
        "full_solve": "solve",
        "solve": "solve",
        "length_penalty": "length_penalty",
    }
    for k, v in components.items():
        fv = to_float(v)
        if fv is not None:
            out[mapping.get(k, k)] = fv
    return out


def parse_component_text(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, val in re.findall(r"([A-Za-z_/-]+)=(-?[0-9.eE+-]+)", text):
        key = key.replace("blank/answer", "blank_acc")
        if key == "blank":
            key = "blank_acc"
        fv = to_float(val)
        if fv is not None:
            out[key] = fv
    return out


@dataclass
class Analysis:
    log_path: Path
    run_dir: Path
    records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Tuple[int, str]] = field(default_factory=list)
    warnings: List[Tuple[int, str]] = field(default_factory=list)
    checkpoints: List[Tuple[int, float, str]] = field(default_factory=list)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    raw_lines: int = 0


def resolve_paths(input_path: Path, out_dir: Optional[Path]) -> Tuple[Path, Path]:
    if input_path.is_dir():
        run_dir = input_path
        log_path = run_dir / "logs" / "train.log"
    else:
        log_path = input_path
        run_dir = log_path.parent.parent if log_path.parent.name == "logs" else log_path.parent
    if out_dir is not None:
        run_dir = out_dir
    return log_path, run_dir


def parse_log(log_path: Path, run_dir: Path) -> Analysis:
    a = Analysis(log_path=log_path, run_dir=run_dir)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    with log_path.open(errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            a.raw_lines += 1
            ts = parse_ts(line)
            if ts is not None:
                a.first_ts = a.first_ts or ts
                a.last_ts = ts
            if any(p in line for p in ERROR_PATTERNS):
                a.errors.append((lineno, line.strip()))
            if "WARN" in line or "COLLAPSE" in line or "SKIPPED" in line or "nonfinite" in line:
                a.warnings.append((lineno, line.strip()))

            rec: Optional[Dict[str, Any]] = None
            if "[GRPO-JSON]" in line:
                payload = line.split("[GRPO-JSON]", 1)[1].strip()
                try:
                    e = json.loads(payload)
                except json.JSONDecodeError:
                    e = None
                if e is not None and e.get("event") in {"rl_step", "gspo_update_epoch", "rl_nonfinite_loss"}:
                    rec = {"line": lineno, "event": e.get("event"), "step": int(e.get("step", 0))}
                    if ts is not None:
                        rec["ts"] = ts.isoformat(sep=" ")
                    loss = e.get("loss") or {}
                    reward = e.get("reward") or {}
                    rollout = e.get("rollout") or {}
                    energy = e.get("energy") or loss
                    grad = e.get("grad") or {}
                    rec.update({
                        "loss": to_float(loss.get("total")),
                        "policy_loss": to_float(loss.get("policy", loss.get("policy_loss"))),
                        "kl_proxy": to_float(loss.get("kl", energy.get("kl_proxy"))),
                        "reward_mean": to_float(reward.get("mean")),
                        "reward_std": to_float(reward.get("std")),
                        "reward_var": to_float(reward.get("var")),
                        "advantage_var": to_float(reward.get("advantage_var")),
                        "comp_len": to_float(rollout.get("completion_len_mean")),
                        "degen": to_float(rollout.get("degenerate_group_rate")),
                        "unique_ratio": to_float(rollout.get("unique_completion_ratio")),
                        "skipped": to_float(rollout.get("skipped_step")),
                        "energy_mean": to_float(energy.get("current_mean", energy.get("energy_mean"))),
                        "old_energy_mean": to_float(energy.get("old_mean", energy.get("old_energy_mean"))),
                        "energy_drift": to_float(energy.get("drift", energy.get("energy_drift"))),
                        "ratio_mean": to_float(energy.get("ratio_mean")),
                        "log_ratio_abs_max": to_float(energy.get("log_ratio_abs_max")),
                        "clamp_rate": to_float(energy.get("log_ratio_clamp_rate")),
                        "effective_clip": to_float(energy.get("effective_clip_rate", energy.get("clip_ratio"))),
                        "reinforce_surrogate": to_float(loss.get("reinforce_surrogate")),
                        "grad_norm": to_float(grad.get("grad_norm_before_clip")),
                        "max_param_grad": to_float(grad.get("max_param_grad_norm")),
                        "nan_params": to_float(grad.get("nan_grad_params")),
                        "update_epoch": to_float(e.get("update_epoch")),
                    })
                    rec.update(flatten_components(reward.get("components") or {}))
            if rec is None:
                m = GRPO_RE.search(line)
                if m:
                    rec = {
                        "line": lineno, "event": "rl_step_text", "step": int(m.group("step")),
                        "loss": to_float(m.group("loss")), "reward_mean": to_float(m.group("reward")),
                        "reward_std": to_float(m.group("reward_std")), "comp_len": to_float(m.group("comp_len")),
                        "effective_clip": to_float(m.group("clip")), "degen": to_float(m.group("degen")),
                        "unique_ratio": to_float(m.group("uniq")),
                    }
                    if ts is not None:
                        rec["ts"] = ts.isoformat(sep=" ")
                    rec.update(parse_component_text(m.group("comps")))
                else:
                    m = GRAD_RE.search(line)
                    if m:
                        rec = {"line": lineno, "event": "grad", "step": int(m.group("step"))}
                        if ts is not None:
                            rec["ts"] = ts.isoformat(sep=" ")
                        for k, v in m.groupdict().items():
                            if k != "step":
                                rec[k] = to_float(v)
                    else:
                        m = EXT_RE.search(line)
                        if m:
                            rec = {"line": lineno, "event": "energy_ext", "step": int(m.group("step"))}
                            if ts is not None:
                                rec["ts"] = ts.isoformat(sep=" ")
                            for k, v in m.groupdict().items():
                                if k != "step":
                                    rec[k] = to_float(v)
            if rec is not None:
                a.records.append({k: v for k, v in rec.items() if v is not None})

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        for p in ckpt_dir.iterdir():
            if not p.is_file():
                continue
            m = CKPT_RE.search(p.name)
            if m:
                a.checkpoints.append((int(m.group("step")), float(m.group("reward")), p.name))
    a.checkpoints.sort()
    return a


def aggregate(records: List[Dict[str, Any]]) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    numeric: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        step = int(r.get("step", 0))
        for k, v in r.items():
            if k in {"line", "event", "step", "ts"}:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric[step][k].append(float(v))
    steps = sorted(numeric)
    keys = sorted({k for by in numeric.values() for k in by})
    series = {k: [] for k in keys}
    for s in steps:
        for k in keys:
            vals = numeric[s].get(k)
            series[k].append(mean(vals) if vals else None)
    return steps, series


def last_value(series: Dict[str, List[Optional[float]]], key: str) -> Optional[float]:
    vals = series.get(key, [])
    for v in reversed(vals):
        if v is not None:
            return v
    return None


def window_mean(series: Dict[str, List[Optional[float]]], key: str, first: bool = False, n: int = 5) -> Optional[float]:
    vals = [v for v in series.get(key, []) if v is not None]
    if not vals:
        return None
    part = vals[:n] if first else vals[-n:]
    return mean(part) if part else None


def best_at(steps: List[int], series: Dict[str, List[Optional[float]]], key: str) -> Tuple[Optional[float], Optional[int]]:
    best_v: Optional[float] = None
    best_s: Optional[int] = None
    for s, v in zip(steps, series.get(key, [])):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_s = v, s
    return best_v, best_s


def first_collapse_step(steps: List[int], series: Dict[str, List[Optional[float]]]) -> Optional[int]:
    for s, r, std, degen in zip(
        steps,
        series.get("reward_mean", [None] * len(steps)),
        series.get("reward_std", [None] * len(steps)),
        series.get("degen", [None] * len(steps)),
    ):
        if (r is not None and r <= 1e-6 and std is not None and std <= 1e-6) or (degen is not None and degen >= 0.9):
            return s
    return None


def write_csv(path: Path, steps: List[int], series: Dict[str, List[Optional[float]]]) -> None:
    keys = list(series)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys)
        for i, s in enumerate(steps):
            w.writerow([s] + ["" if series[k][i] is None else f"{series[k][i]:.10g}" for k in keys])


def non_null_xy(steps: List[int], values: List[Optional[float]]) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for s, v in zip(steps, values):
        if v is None:
            continue
        xs.append(s)
        ys.append(v)
    return xs, ys


def plot_line(ax: Any, steps: List[int], series: Dict[str, List[Optional[float]]], key: str, label: Optional[str] = None, *args: Any, **kwargs: Any) -> bool:
    values = series.get(key)
    if values is None:
        return False
    xs, ys = non_null_xy(steps, values)
    if not xs:
        return False
    ax.plot(xs, ys, label=label or key, *args, **kwargs)
    return True


def setup_axis(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def plot_metrics(path: Path, title: str, steps: List[int], series: Dict[str, List[Optional[float]]], checkpoints: List[Tuple[int, float, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=14)

    ax = axes[0][0]
    for k, label in [("reward_mean", "train reward mean"), ("reward_std", "train reward std")]:
        plot_line(ax, steps, series, k, label)
    if checkpoints:
        ax.scatter([s for s, _, _ in checkpoints], [v for _, v, _ in checkpoints], marker="D", label="checkpoint val reward")
    ax.set_title("Reward")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    ax.legend()

    ax = axes[0][1]
    component_keys = [k for k in ["format", "blank_acc", "answer_acc", "exact_match", "partial_credit", "parse_rate", "clue", "solve"] if k in series]
    for k in component_keys:
        label = "blank/answer acc" if k in {"blank_acc", "answer_acc"} else k
        plot_line(ax, steps, series, k, label)
    ax.set_title("Task components")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    ax.legend()

    ax = axes[1][0]
    for k, label in [("degen", "degen"), ("effective_clip", "clip"), ("skipped", "skipped")]:
        plot_line(ax, steps, series, k, label)
    collapse = [0.0] * len(steps)
    for i, s in enumerate(steps):
        d = series.get("degen", [None] * len(steps))[i]
        r = series.get("reward_mean", [None] * len(steps))[i]
        if (d is not None and d >= 0.9) or (r is not None and r <= 1e-6):
            collapse[i] = 1.0
    if any(collapse):
        ax.fill_between(steps, collapse, alpha=0.2, label="collapse_count")
    ax.set_title("Degeneration / skipped / collapse")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    ax.legend()

    ax = axes[1][1]
    for k, label in [("grad_norm", "grad_norm"), ("max_param_grad", "max_param_grad")]:
        plot_line(ax, steps, series, k, label)
    ax2 = ax.twinx()
    for k, label, style in [("ratio_mean", "ratio_mean", "--"), ("clamp_rate", "clamp_rate", "--")]:
        plot_line(ax2, steps, series, k, label, linestyle="--")
    ax.set_title("Optimization / energy")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_stability(path: Path, title: str, steps: List[int], series: Dict[str, List[Optional[float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(title + " stability", fontsize=14)
    specs = [
        ("Energy drift", ["energy_drift", "energy_mean", "old_energy_mean"]),
        ("Ratio / clipping", ["ratio_mean", "log_ratio_abs_max", "clamp_rate", "effective_clip"]),
        ("Grad health", ["grad_norm", "max_param_grad", "nan_params"]),
        ("Rollout health", ["comp_len", "unique_ratio", "degen", "advantage_var"]),
    ]
    for ax, (name, keys) in zip(axes.flat, specs):
        for k in keys:
            plot_line(ax, steps, series, k)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overview(path: Path, title: str, steps: List[int], series: Dict[str, List[Optional[float]]], checkpoints: List[Tuple[int, float, str]]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(title + " RL overview", fontsize=16)

    ax = axes[0][0]
    plot_line(ax, steps, series, "reward_mean", "train reward mean")
    plot_line(ax, steps, series, "reward_std", "train reward std")
    if checkpoints:
        ax.scatter([s for s, _, _ in checkpoints], [v for _, v, _ in checkpoints], marker="D", label="checkpoint val reward")
    setup_axis(ax, "Reward")

    ax = axes[0][1]
    for k in ["format", "blank_acc", "answer_acc", "exact_match", "partial_credit", "parse_rate", "clue", "solve"]:
        label = "blank/answer acc" if k in {"blank_acc", "answer_acc"} else k
        plot_line(ax, steps, series, k, label)
    setup_axis(ax, "Task components")

    ax = axes[1][0]
    for k, label in [("degen", "degen"), ("effective_clip", "clip"), ("skipped", "skipped")]:
        plot_line(ax, steps, series, k, label)
    collapse = [0.0] * len(steps)
    for i, _ in enumerate(steps):
        d = series.get("degen", [None] * len(steps))[i]
        r = series.get("reward_mean", [None] * len(steps))[i]
        std = series.get("reward_std", [None] * len(steps))[i]
        if (d is not None and d >= 0.9) or (r is not None and r <= 1e-6 and (std is None or std <= 1e-6)):
            collapse[i] = 1.0
    if any(collapse):
        ax.fill_between(steps, collapse, alpha=0.2, label="collapse_count")
    setup_axis(ax, "Degeneration / skipped / collapse")

    ax = axes[1][1]
    plot_line(ax, steps, series, "grad_norm", "grad_norm")
    plot_line(ax, steps, series, "max_param_grad", "max_param_grad")
    ax2 = ax.twinx()
    plot_line(ax2, steps, series, "ratio_mean", "ratio_mean", linestyle="--")
    plot_line(ax2, steps, series, "clamp_rate", "clamp_rate", linestyle="--")
    ax.set_title("Optimization / energy")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2)

    ax = axes[2][0]
    for k in ["energy_drift", "energy_mean", "old_energy_mean"]:
        plot_line(ax, steps, series, k)
    setup_axis(ax, "Energy drift")

    ax = axes[2][1]
    for k in ["ratio_mean", "log_ratio_abs_max", "clamp_rate", "effective_clip"]:
        plot_line(ax, steps, series, k)
    setup_axis(ax, "Ratio / clipping")

    ax = axes[3][0]
    for k in ["grad_norm", "max_param_grad", "nan_params"]:
        plot_line(ax, steps, series, k)
    setup_axis(ax, "Grad health")

    ax = axes[3][1]
    for k in ["comp_len", "unique_ratio", "degen", "advantage_var"]:
        plot_line(ax, steps, series, k)
    setup_axis(ax, "Rollout health")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def status_and_assessment(a: Analysis, steps: List[int], series: Dict[str, List[Optional[float]]]) -> Tuple[str, str, List[str]]:
    notes: List[str] = []
    status = "unknown"
    if a.errors:
        status = "failed"
        notes.append(a.errors[-1][1])
    elif any("训练成功" in x for _, x in a.warnings):
        status = "completed"
    else:
        status = "stopped_or_incomplete"
    first_reward = window_mean(series, "reward_mean", first=True)
    last_reward = window_mean(series, "reward_mean", first=False)
    best_reward, best_step = best_at(steps, series, "reward_mean")
    collapse_step = first_collapse_step(steps, series)
    last_degen = last_value(series, "degen")
    last_grad = last_value(series, "grad_norm")
    last_drift = last_value(series, "energy_drift")
    if collapse_step is not None:
        notes.append(f"collapse/degenerate signal starts around step {collapse_step}")
    if first_reward is not None and last_reward is not None:
        if last_reward < 0.25 * max(first_reward, 1e-9):
            notes.append(f"train reward regressed from {first_reward:.4f} to {last_reward:.4f}")
        elif last_reward > first_reward * 1.2:
            notes.append(f"train reward improved from {first_reward:.4f} to {last_reward:.4f}")
    if last_degen is not None and last_degen >= 0.9:
        notes.append(f"last degeneration rate is high ({last_degen:.2f})")
    if last_grad is not None and last_grad < 1e-2:
        notes.append(f"last grad_norm is very small ({last_grad:.3g})")
    if last_drift is not None and abs(last_drift) > 0.5:
        notes.append(f"large energy drift at the end ({last_drift:.3f})")
    if best_reward is not None:
        notes.append(f"best parsed train reward {best_reward:.4f} at step {best_step}")
    return status, "; ".join(notes) if notes else "no obvious issue from parsed metrics", notes


def write_report(path: Path, a: Analysis, steps: List[int], series: Dict[str, List[Optional[float]]], title: str) -> None:
    status, reason, notes = status_and_assessment(a, steps, series)
    first_reward = window_mean(series, "reward_mean", first=True)
    last_reward = window_mean(series, "reward_mean")
    best_reward, best_step = best_at(steps, series, "reward_mean")
    best_ckpt = max(a.checkpoints, key=lambda x: x[1], default=None)
    last_ckpt = a.checkpoints[-1] if a.checkpoints else None
    collapse_step = first_collapse_step(steps, series)
    max_step = max(steps) if steps else None
    lines = [
        f"# Run Analysis: {title}", "",
        "## Status", "",
        f"- Run directory: `{a.run_dir}`",
        f"- Log file: `{a.log_path}`",
        f"- Parsed log lines: {a.raw_lines}",
        f"- Time range: {a.first_ts} to {a.last_ts}",
        f"- Last parsed metric step: {max_step}",
        f"- Status: **{status}**",
        f"- Status reason: {reason}",
        "", "## Key Metrics", "",
        "| Metric | Value |", "| --- | ---: |",
    ]
    metric_rows = [
        ("first-window train reward", first_reward),
        ("last-window train reward", last_reward),
        ("best train reward", None if best_reward is None else f"{best_reward:.4f} at step {best_step}"),
        ("last reward std", last_value(series, "reward_std")),
        ("last fmt", last_value(series, "format")),
        ("last blank/answer acc", last_value(series, "blank_acc") or last_value(series, "answer_acc")),
        ("last solve", last_value(series, "solve")),
        ("last degen", last_value(series, "degen")),
        ("last skipped", last_value(series, "skipped")),
        ("collapse step", collapse_step),
        ("last grad_norm", last_value(series, "grad_norm")),
        ("last ratio_mean", last_value(series, "ratio_mean")),
        ("last clamp_rate", last_value(series, "clamp_rate")),
        ("last energy_drift", last_value(series, "energy_drift")),
    ]
    for k, v in metric_rows:
        if v is None:
            s = "n/a"
        elif isinstance(v, float):
            s = f"{v:.4f}"
        else:
            s = str(v)
        lines.append(f"| {k} | {s} |")
    if best_ckpt:
        lines.append(f"| best checkpoint val reward | {best_ckpt[1]:.4f} at step {best_ckpt[0]} |")
    if last_ckpt:
        lines.append(f"| last checkpoint val reward | {last_ckpt[1]:.4f} at step {last_ckpt[0]} |")
    lines += ["", "## Checkpoints", "", "| Step | val/reward_mean | File |", "| ---: | ---: | --- |"]
    if a.checkpoints:
        for s, v, name in a.checkpoints:
            lines.append(f"| {s} | {v:.4f} | `{name}` |")
    else:
        lines.append("| n/a | n/a | n/a |")
    lines += ["", "## Interpretation", ""]
    if notes:
        for n in notes:
            lines.append(f"- {n}.")
    else:
        lines.append("- No major parsed anomaly found.")
    if collapse_step is not None:
        lines.append("- The run should not be continued from late checkpoints after collapse; prefer the best early checkpoint or restart from SFT with safer hyperparameters.")
    if last_value(series, "effective_clip") == 0 or (last_value(series, "clamp_rate") == 0):
        lines.append("- GSPO/PPO clipping rarely activates; ratio-based trust region is not constraining this run.")
    if (last_value(series, "energy_drift") or 0) < -0.5:
        lines.append("- Energy drift is strongly negative, suggesting the policy can become over-confident relative to the reference without being penalized by one-sided KL.")
    lines += ["", "## Recommended Next Actions", ""]
    if collapse_step is not None:
        lines += [
            "- Restart with lower learning rate and/or `GSPO_UPDATE_EPOCHS=1` rather than continuing this collapsed run.",
            "- Add symmetric or two-sided energy KL to prevent large negative energy drift from the SFT reference.",
            "- Reduce debug logging and run a shorter smoke test before a long training job.",
        ]
    else:
        lines += [
            "- Continue monitoring reward, degeneration, energy drift, and validation checkpoints.",
            "- If ratio/clip stay near zero, inspect whether the GSPO surrogate is too conservative for this model.",
        ]
    if a.errors:
        lines += ["", "## Error Snippets", ""]
        for line_no, text in a.errors[-12:]:
            lines.append(f"- line {line_no}: `{text[:240]}`")
    if a.warnings:
        lines += ["", "## Warning / Collapse Snippets", ""]
        for line_no, text in a.warnings[-12:]:
            lines.append(f"- line {line_no}: `{text[:240]}`")
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- `analysis_overview.png`",
        "- `analysis_metrics.png`",
        "- `analysis_stability.png`",
        "- `analysis_metrics.csv`",
        "",
    ]
    path.write_text("\n".join(lines))


def analyze(input_path: Path, out_dir: Optional[Path]) -> None:
    log_path, run_dir = resolve_paths(input_path, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    a = parse_log(log_path, run_dir)
    steps, series = aggregate(a.records)
    title = run_dir.parent.name if run_dir.name.startswith("logs") else run_dir.name
    write_csv(run_dir / "analysis_metrics.csv", steps, series)
    plot_overview(run_dir / "analysis_overview.png", title, steps, series, a.checkpoints)
    plot_metrics(run_dir / "analysis_metrics.png", title, steps, series, a.checkpoints)
    plot_stability(run_dir / "analysis_stability.png", title, steps, series)
    write_report(run_dir / "analysis_report.md", a, steps, series, title)
    print(f"Wrote {run_dir / 'analysis_report.md'}")
    print(f"Wrote {run_dir / 'analysis_overview.png'}")
    print(f"Wrote {run_dir / 'analysis_metrics.png'}")
    print(f"Wrote {run_dir / 'analysis_stability.png'}")
    print(f"Wrote {run_dir / 'analysis_metrics.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="Run directory or train.log path")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory; defaults to run directory")
    args = ap.parse_args()
    analyze(args.input, args.out_dir)


if __name__ == "__main__":
    main()
