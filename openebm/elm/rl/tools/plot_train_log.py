"""Robust training log parser & visualizer for EBM-GRPO Sudoku RL.

Parses GRPO-style training logs and produces:
  - PNG: multi-subplot summary figure (grouped by metric category)
  - CSV: per-step (per-rank) raw records + a per-step aggregated CSV
  - stdout: final/min/max + first-step a metric becomes constant (e.g. 0)

Usage:
    python -m openebm.elm.rl.tools.plot_train_log <train.log> [--out-dir DIR]

The parser is regex-driven and tolerant: lines that cannot be parsed are
silently skipped. Metric extraction is keyword-based, so adding new metrics
only requires extending the `_METRIC_PATTERNS` table.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Regexes that extract structured records from log lines. Each pattern returns
# a dict of {metric_name: float} when it matches.
_GRPO_LINE = re.compile(
    r"\[GRPO\]\s+step=(?P<step>\d+)\s+\|\s+"
    r"loss=(?P<loss>-?[0-9.eE+-]+)\s+\|\s+"
    r"reward=(?P<reward>-?[0-9.eE+-]+)\xb1(?P<reward_std>-?[0-9.eE+-]+)\s+\|\s+"
    r"fmt=(?P<fmt>-?[0-9.eE+-]+)\s+"
    r"clue=(?P<clue>-?[0-9.eE+-]+)\s+"
    r"blank_acc=(?P<blank_acc>-?[0-9.eE+-]+)\s+"
    r"solve=(?P<solve>-?[0-9.eE+-]+)\s+\|\s+"
    r"comp_len=(?P<comp_len>-?[0-9.eE+-]+)\s+\|\s+"
    r"clip=(?P<clip>-?[0-9.eE+-]+)\s+"
    r"degen=(?P<degen>-?[0-9.eE+-]+)"
)

_GRPO_GRAD = re.compile(
    r"\[GRPO\]\s+step=(?P<step>\d+)\s+"
    r"grad_norm=(?P<grad_norm>-?[0-9.eE+-]+)\s+"
    r"max_param_grad=(?P<max_param_grad>-?[0-9.eE+-]+)\s+"
    r"nan_params=(?P<nan_params>-?[0-9.eE+-]+)"
)

# Generic "key=value" sweep — captures stability/* / val/* lines (Lightning print).
_KV_PATTERNS = [
    re.compile(r"alpha=(?P<alpha>-?[0-9.eE+-]+)"),
    re.compile(r"energy_mean=(?P<energy_mean>-?[0-9.eE+-]+)"),
    re.compile(r"ratio_mean=(?P<ratio_mean>-?[0-9.eE+-]+)"),
    re.compile(r"val/reward_mean=(?P<val_reward_mean>-?[0-9.eE+-]+)"),
    re.compile(r"lr=(?P<lr>-?[0-9.eE+-]+)"),
    re.compile(r"learning_rate=(?P<lr>-?[0-9.eE+-]+)"),
    re.compile(r"kl=(?P<kl>-?[0-9.eE+-]+)"),
    re.compile(r"sft_loss=(?P<sft_loss>-?[0-9.eE+-]+)"),
    re.compile(r"reward_loss=(?P<reward_loss>-?[0-9.eE+-]+)"),
    re.compile(r"policy_loss=(?P<policy_loss>-?[0-9.eE+-]+)"),
]

# Metric → group (subplot category)
_METRIC_GROUP: Dict[str, str] = {
    "loss": "loss",
    "reward_loss": "loss",
    "policy_loss": "loss",
    "sft_loss": "loss",
    "ref_energy_kl": "loss",
    "kl": "loss",
    "ref_energy_drift": "energy",
    "old_policy_energy_drift": "energy",
    "reward": "reward",
    "reward_std": "reward",
    "val_reward_mean": "reward",
    "fmt": "reward_components",
    "clue": "reward_components",
    "blank_acc": "reward_components",
    "validity": "reward_components",
    "solve": "reward_components",
    "comp_len": "generation",
    "clip": "generation",
    "degen": "generation",
    "grad_norm": "optim",
    "max_param_grad": "optim",
    "nan_params": "optim",
    "alpha": "optim",
    "lr": "optim",
    "energy_mean": "energy",
    "ratio_mean": "energy",
}


def parse_log(log_path: str) -> List[Dict[str, float]]:
    """Parse log, return list of records. Each record has a `step` key.

    Multiple records per step (one per rank/sub-batch) are kept as-is.
    Use `aggregate_by_step` to fold them.
    """
    records: List[Dict[str, float]] = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _GRPO_LINE.search(line)
            if m:
                rec = {k: float(v) for k, v in m.groupdict().items()}
                rec["step"] = int(rec["step"])
                records.append(rec)
                continue
            m = _GRPO_GRAD.search(line)
            if m:
                rec = {k: float(v) for k, v in m.groupdict().items()}
                rec["step"] = int(rec["step"])
                records.append(rec)
                continue
            # Generic key=value sweep (only kept when accompanied by a step
            # somewhere on the line so the time axis is well-defined).
            step_match = re.search(r"step[=:](\d+)", line)
            if step_match is None:
                continue
            extras: Dict[str, float] = {}
            for pat in _KV_PATTERNS:
                m = pat.search(line)
                if m:
                    for k, v in m.groupdict().items():
                        try:
                            extras[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            if extras:
                extras["step"] = int(step_match.group(1))
                records.append(extras)
    return records


def aggregate_by_step(
    records: List[Dict[str, float]],
) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    """Average records sharing the same step. Missing metrics yield None."""
    by_step: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        step = int(rec["step"])
        for k, v in rec.items():
            if k == "step":
                continue
            by_step[step][k].append(float(v))

    steps = sorted(by_step)
    all_keys = sorted({k for s in by_step.values() for k in s})
    series: Dict[str, List[Optional[float]]] = {k: [] for k in all_keys}
    for s in steps:
        for k in all_keys:
            vals = by_step[s].get(k)
            series[k].append(mean(vals) if vals else None)
    return steps, series


def first_constant_step(
    steps: List[int], values: List[Optional[float]], target: float = 0.0,
    min_run: int = 3, atol: float = 1e-9,
) -> Optional[int]:
    """Return the first step at which `values` becomes `target` and stays so
    for at least `min_run` consecutive recorded steps."""
    run_start: Optional[int] = None
    run_len = 0
    for s, v in zip(steps, values):
        if v is None:
            run_start, run_len = None, 0
            continue
        if abs(v - target) <= atol:
            if run_start is None:
                run_start, run_len = s, 1
            else:
                run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_start, run_len = None, 0
    return None


def write_csv(steps, series, raw_records, out_dir):
    raw_path = os.path.join(out_dir, "train_log_raw.csv")
    agg_path = os.path.join(out_dir, "train_log_agg.csv")

    all_keys = sorted({k for r in raw_records for k in r if k != "step"})
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + all_keys)
        for r in raw_records:
            w.writerow([r.get("step", "")] + [r.get(k, "") for k in all_keys])

    with open(agg_path, "w", newline="") as f:
        w = csv.writer(f)
        keys = list(series)
        w.writerow(["step"] + keys)
        for i, s in enumerate(steps):
            w.writerow([s] + [
                "" if series[k][i] is None else f"{series[k][i]:.6g}"
                for k in keys
            ])
    return raw_path, agg_path


def plot_summary(steps, series, out_path, title=""):
    groups: Dict[str, List[str]] = defaultdict(list)
    for metric in series:
        groups[_METRIC_GROUP.get(metric, "other")].append(metric)

    group_order = ["loss", "reward", "reward_components", "generation", "optim", "energy", "other"]
    groups_present = [g for g in group_order if g in groups]

    n = len(groups_present)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, gname in enumerate(groups_present):
        ax = axes[idx // cols][idx % cols]
        for m in sorted(groups[gname]):
            ys = series[m]
            xs = [s for s, y in zip(steps, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            if not yv:
                continue
            ax.plot(xs, yv, label=m, marker=".", markersize=3, linewidth=1)
        ax.set_title(gname)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if gname == "optim":
            try:
                ax.set_yscale("symlog", linthresh=1e-6)
            except Exception:
                pass

    # Hide unused axes
    for k in range(len(groups_present), rows * cols):
        axes[k // cols][k % cols].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_summary(steps, series):
    print(f"\n{'metric':<22} {'final':>12} {'min':>12} {'max':>12} {'first_zero_step':>16}")
    print("-" * 80)
    for k in sorted(series):
        vals = [v for v in series[k] if v is not None]
        if not vals:
            continue
        final = series[k][-1] if series[k][-1] is not None else next(
            (v for v in reversed(series[k]) if v is not None), None
        )
        first0 = first_constant_step(steps, series[k], target=0.0, min_run=3)
        print(
            f"{k:<22} {final if final is not None else 'na':>12} "
            f"{min(vals):>12.6g} {max(vals):>12.6g} "
            f"{(str(first0) if first0 is not None else '-'):>16}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path to train.log")
    parser.add_argument("--out-dir", default="./analysis_output",
                        help="Directory to write png/csv into")
    parser.add_argument("--title", default="", help="Figure title")
    args = parser.parse_args()

    if not os.path.isfile(args.log_path):
        print(f"error: log not found: {args.log_path}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out_dir, exist_ok=True)

    records = parse_log(args.log_path)
    if not records:
        print("warning: no records parsed", file=sys.stderr)
        sys.exit(1)

    steps, series = aggregate_by_step(records)
    png_path = os.path.join(args.out_dir, "train_log_summary.png")
    plot_summary(steps, series, png_path,
                 title=args.title or os.path.basename(args.log_path))
    raw_csv, agg_csv = write_csv(steps, series, records, args.out_dir)

    print(f"parsed records:  {len(records)}")
    print(f"distinct steps:  {len(steps)}  (range {steps[0]}..{steps[-1]})")
    print(f"png:  {png_path}")
    print(f"raw csv:  {raw_csv}")
    print(f"agg csv:  {agg_csv}")
    print_summary(steps, series)


if __name__ == "__main__":
    main()
