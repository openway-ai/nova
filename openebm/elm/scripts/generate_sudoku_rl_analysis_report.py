#!/usr/bin/env python3
"""Generate a compact Sudoku RL comparison report from analyzer artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MetricRow = Dict[str, Optional[float]]


def _to_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_key_value(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _metrics_path(run_dir: Path) -> Path:
    return run_dir / "analysis_metrics.csv"


def _load_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[Optional[float]]]]:
    path = _metrics_path(run_dir)
    if not path.exists():
        return [], {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        steps: List[int] = []
        series: Dict[str, List[Optional[float]]] = {}
        for row in reader:
            step_value = row.get("step")
            if step_value is None:
                continue
            try:
                steps.append(int(float(step_value)))
            except ValueError:
                continue
            for key, value in row.items():
                if key == "step":
                    continue
                series.setdefault(key, []).append(_to_float(value or ""))
        expected_len = len(steps)
        for values in series.values():
            while len(values) < expected_len:
                values.append(None)
    return steps, series


def _values(series: Dict[str, List[Optional[float]]], key: str) -> List[float]:
    return [v for v in series.get(key, []) if v is not None]


def _window_mean(series: Dict[str, List[Optional[float]]], key: str, first: bool = False, n: int = 5) -> Optional[float]:
    values = _values(series, key)
    if not values:
        return None
    window = values[:n] if first else values[-n:]
    return sum(window) / len(window) if window else None


def _last(series: Dict[str, List[Optional[float]]], key: str) -> Optional[float]:
    for value in reversed(series.get(key, [])):
        if value is not None:
            return value
    return None


def _best(steps: List[int], series: Dict[str, List[Optional[float]]], key: str) -> Tuple[Optional[float], Optional[int]]:
    best_value: Optional[float] = None
    best_step: Optional[int] = None
    for step, value in zip(steps, series.get(key, [])):
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_step = step
    return best_value, best_step


def _mean(series: Dict[str, List[Optional[float]]], key: str) -> Optional[float]:
    values = _values(series, key)
    if not values:
        return None
    return sum(values) / len(values)


def _first_existing(series: Dict[str, List[Optional[float]]], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in series:
            return key
    return None


def _parse_best_checkpoint(run_dir: Path) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None, None, None
    best: Tuple[Optional[int], Optional[float], Optional[str]] = (None, None, None)
    pattern = re.compile(r"step=(?P<step>\d+).*?(?:reward_mean|val_reward|reward)=(?P<reward>-?\d+(?:\.\d+)?)")
    for path in ckpt_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if not match:
            continue
        step = int(match.group("step"))
        reward = float(match.group("reward"))
        if best[1] is None or reward > best[1]:
            best = (step, reward, path.name)
    return best


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    steps, series = _load_metrics(run_dir)
    blank_key = _first_existing(series, ["blank_acc", "answer_acc"])
    solve_key = _first_existing(series, ["solve", "exact_match"])
    first_reward = _window_mean(series, "reward_mean", first=True)
    last_reward = _window_mean(series, "reward_mean")
    best_reward, best_reward_step = _best(steps, series, "reward_mean")
    best_blank, best_blank_step = _best(steps, series, blank_key) if blank_key else (None, None)
    best_solve, best_solve_step = _best(steps, series, solve_key) if solve_key else (None, None)
    ckpt_step, ckpt_reward, ckpt_name = _parse_best_checkpoint(run_dir)
    return {
        "run_dir": run_dir,
        "metrics_found": bool(steps and series),
        "last_step": max(steps) if steps else None,
        "num_metric_steps": len(steps),
        "first_reward": first_reward,
        "last_reward": last_reward,
        "best_reward": best_reward,
        "best_reward_step": best_reward_step,
        "best_blank": best_blank,
        "best_blank_step": best_blank_step,
        "best_solve": best_solve,
        "best_solve_step": best_solve_step,
        "last_degen": _last(series, "degen"),
        "mean_skipped": _mean(series, "skipped"),
        "last_unique_ratio": _last(series, "unique_ratio"),
        "last_grad_norm": _last(series, "grad_norm"),
        "last_ref_energy_drift": _last(series, "ref_energy_drift"),
        "last_old_policy_energy_drift": _last(series, "old_policy_energy_drift"),
        "last_log_ratio_clamp_rate": _last(series, "log_ratio_clamp_rate"),
        "best_ckpt_step": ckpt_step,
        "best_ckpt_reward": ckpt_reward,
        "best_ckpt_name": ckpt_name,
        "hparams": _read_json(run_dir / "config" / "hparams.json"),
        "base_ckpt": _read_json(run_dir / "config" / "base_ckpt_ref.json"),
        "metadata": _read_key_value(run_dir / "config" / "fusion_rjob_metadata.txt"),
    }


def _delta(current: Optional[float], baseline: Optional[float]) -> str:
    if current is None or baseline is None:
        return "n/a"
    return _fmt(current - baseline)


def _metric_table(current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> List[str]:
    rows = [
        ("last_step", "Last parsed step"),
        ("first_reward", "First-window reward"),
        ("last_reward", "Last-window reward"),
        ("best_reward", "Best reward"),
        ("best_blank", "Best blank/answer accuracy"),
        ("best_solve", "Best solve/exact-match"),
        ("last_degen", "Last degeneration rate"),
        ("mean_skipped", "Mean skipped-step rate"),
        ("last_unique_ratio", "Last unique completion ratio"),
        ("last_grad_norm", "Last grad norm"),
        ("last_ref_energy_drift", "Last ref energy drift"),
        ("last_old_policy_energy_drift", "Last old-policy energy drift"),
        ("last_log_ratio_clamp_rate", "Last log-ratio clamp rate"),
        ("best_ckpt_reward", "Best checkpoint reward"),
    ]
    lines = ["| Metric | Current | Baseline | Delta |", "|---|---:|---:|---:|"]
    for key, label in rows:
        current_value = current.get(key)
        baseline_value = baseline.get(key) if baseline else None
        lines.append(f"| {label} | {_fmt(current_value)} | {_fmt(baseline_value)} | {_delta(current_value, baseline_value)} |")
    return lines


def _config_table(current: Dict[str, Any]) -> List[str]:
    hparams = current.get("hparams") or {}
    base_ckpt = current.get("base_ckpt") or {}
    metadata = current.get("metadata") or {}
    rows = [
        ("fusion_branch", metadata.get("fusion_branch")),
        ("fusion_commit", metadata.get("fusion_commit")),
        ("sft_checkpoint", metadata.get("sft_checkpoint") or base_ckpt.get("ckpt_path") or hparams.get("sft_checkpoint")),
        ("sft_valid_loss", base_ckpt.get("valid_loss")),
        ("sudoku_data_dir", metadata.get("sudoku_data_dir")),
        ("num_gpus", metadata.get("num_gpus") or hparams.get("num_gpus")),
        ("rl_loss_type", metadata.get("rl_loss_type") or hparams.get("rl_loss_type")),
        ("max_steps", metadata.get("max_steps") or hparams.get("max_steps")),
        ("num_generations", metadata.get("num_generations") or hparams.get("num_generations")),
        ("temperature", metadata.get("temperature") or hparams.get("temperature")),
        ("top_p", metadata.get("top_p") or hparams.get("top_p")),
        ("learning_rate", metadata.get("learning_rate") or hparams.get("learning_rate")),
        ("muon_lr", metadata.get("muon_lr") or hparams.get("muon_lr")),
        ("beta", metadata.get("beta") or hparams.get("beta")),
        ("energy_kl_mode", metadata.get("energy_kl_mode") or hparams.get("energy_kl_mode")),
        ("skip_consensus", metadata.get("skip_consensus") or hparams.get("skip_consensus")),
    ]
    lines = ["| Key | Value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| `{key}` | `{_fmt(value)}` |")
    return lines


def _conclusions(current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    current_best = current.get("best_reward")
    baseline_best = baseline.get("best_reward") if baseline else None
    current_solve = current.get("best_solve")
    baseline_solve = baseline.get("best_solve") if baseline else None

    if baseline and current_best is not None and baseline_best is not None:
        if current_best > baseline_best + 1e-6:
            notes.append(
                f"Current run has the better parsed reward peak ({_fmt(current_best)} vs {_fmt(baseline_best)})."
            )
        elif current_best < baseline_best - 1e-6:
            notes.append(
                f"Baseline still has the better parsed reward peak ({_fmt(baseline_best)} vs {_fmt(current_best)})."
            )
        else:
            notes.append("Current and baseline parsed reward peaks are effectively tied.")
    elif not baseline:
        notes.append("No baseline metrics were found, so this report only summarizes the current run.")

    if current_solve is not None:
        if current_solve <= 0:
            notes.append("No full-solve/exact-match signal was observed in parsed metrics; Sudoku RL remains unconverged.")
        elif baseline_solve is not None and current_solve > baseline_solve:
            notes.append(f"Full-solve/exact-match improved from {_fmt(baseline_solve)} to {_fmt(current_solve)}.")
        else:
            notes.append(f"Best full-solve/exact-match signal is {_fmt(current_solve)}; inspect validation samples before claiming convergence.")
    else:
        notes.append("No solve/exact-match metric was available in the parsed CSV; convergence cannot be claimed from this report.")

    last_degen = current.get("last_degen")
    mean_skipped = current.get("mean_skipped")
    grad_norm = current.get("last_grad_norm")
    ref_drift = current.get("last_ref_energy_drift")
    if isinstance(last_degen, float) and last_degen >= 0.9:
        notes.append("Late degeneration is high; prefer earlier checkpoints or safer exploration settings.")
    if isinstance(mean_skipped, float) and mean_skipped > 0.2:
        notes.append("Skipped-step rate is high; the new `any` consensus is protecting training but reducing update density.")
    if isinstance(grad_norm, float) and grad_norm < 1e-8:
        notes.append("Final gradient norm is near zero; check whether reward variance or skip-degen gates removed most updates.")
    if isinstance(ref_drift, float) and abs(ref_drift) > 0.5:
        notes.append("Reference energy drift is large; keep KL/energy anchoring enabled and inspect generated trajectories.")
    return notes


def write_report(current: Dict[str, Any], baseline: Optional[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Sudoku RL Optimization Analysis",
        "",
        "## Runs",
        "",
        f"- Current: `{current['run_dir']}`",
        f"- Baseline: `{baseline['run_dir']}`" if baseline else "- Baseline: `not found or metrics unavailable`",
        "",
        "## Metrics",
        "",
        *_metric_table(current, baseline),
        "",
        "## Configuration",
        "",
        *_config_table(current),
        "",
        "## Conclusion",
        "",
    ]
    lines.extend(f"- {note}" for note in _conclusions(current, baseline))
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Analyzer report: `{current['run_dir'] / 'analysis_report.md'}`",
        f"- Metrics CSV: `{current['run_dir'] / 'analysis_metrics.csv'}`",
        f"- Overview plot: `{current['run_dir'] / 'analysis_overview.png'}`",
        f"- Stability plot: `{current['run_dir'] / 'analysis_stability.png'}`",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Current Sudoku RL run stage directory, e.g. .../sft_train")
    parser.add_argument("--baseline-run-dir", type=Path, default=None, help="Optional pre-optimization run stage directory")
    parser.add_argument("--out", type=Path, default=None, help="Output Markdown path")
    args = parser.parse_args()

    current = summarize_run(args.run_dir)
    if not current["metrics_found"]:
        raise FileNotFoundError(f"No analyzer metrics found at {_metrics_path(args.run_dir)}")

    baseline = None
    if args.baseline_run_dir is not None and _metrics_path(args.baseline_run_dir).exists():
        baseline = summarize_run(args.baseline_run_dir)
        if not baseline["metrics_found"]:
            baseline = None

    out_path = args.out or (args.run_dir / "sudoku_rl_analysis_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(current, baseline, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
