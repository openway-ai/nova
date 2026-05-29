#!/usr/bin/env python3
"""Rank Sudoku SFT checkpoints by lightweight generation metrics.

Lightning top-k checkpoints are cheap to select by CE loss, but Sudoku RL starts
care about generation health: parse rate, solve rate, filled-cell accuracy, and
OOD test behavior. This script evaluates a bounded set of candidate checkpoints
with eval_sudoku_samples.py and writes a ranked table for RL-start selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


STEP_RE = re.compile(r"(?:s=)?step=(\d+)|periodic-s=(\d+)")


@dataclass
class Candidate:
    path: Path
    step: int
    kind: str


def _step_from_name(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if not match:
        return -1
    for group in match.groups():
        if group is not None:
            return int(group)
    return -1


def _discover_checkpoints(ckpt_dir: Path, max_ckpts: int) -> List[Candidate]:
    paths = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
    candidates: List[Candidate] = []
    for path in paths:
        name = path.name
        if name == "last.ckpt":
            kind = "last"
        elif name.startswith("periodic-"):
            kind = "periodic"
        else:
            kind = "topk"
        candidates.append(Candidate(path=path, step=_step_from_name(path), kind=kind))

    topk = [c for c in candidates if c.kind == "topk"]
    periodic = [c for c in candidates if c.kind == "periodic"]
    last = [c for c in candidates if c.kind == "last"]

    # Keep all top-k first, then the most recent periodic checkpoints. This preserves
    # CE-selected candidates while adding late-training candidates for RL-start checks.
    topk.sort(key=lambda c: (c.step, c.path.stat().st_mtime), reverse=True)
    periodic.sort(key=lambda c: (c.step, c.path.stat().st_mtime), reverse=True)
    last.sort(key=lambda c: c.path.stat().st_mtime, reverse=True)
    ordered = topk + periodic + last

    seen = set()
    deduped = []
    for cand in ordered:
        resolved = cand.path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(cand)
        if len(deduped) >= max_ckpts:
            break
    return deduped


def _split_summary(summary: Dict, split: str) -> Dict:
    by_split = summary.get("splits", {})
    if split in by_split:
        return by_split[split]
    # Older summary variants used a top-level per_split key in some runs.
    by_split = summary.get("per_split", {})
    return by_split.get(split, {})


def _score(summary: Dict) -> float:
    val = _split_summary(summary, "val")
    test = _split_summary(summary, "test")
    overall = summary.get("overall", summary)

    test_solve = float(test.get("fully_solved_pct", 0.0))
    val_solve = float(val.get("fully_solved_pct", 0.0))
    test_filled = float(test.get("filled_cell_acc", 0.0))
    val_filled = float(val.get("filled_cell_acc", 0.0))
    parsed = float(overall.get("parsed_pct", 0.0))
    valid = float(overall.get("valid_sudoku_pct", 0.0))

    # OOD test solve is weighted highest because this checkpoint is intended as
    # a robust RL starting policy, not merely a low-CE teacher-forced model.
    return (
        0.40 * test_solve
        + 0.25 * val_solve
        + 0.15 * test_filled
        + 0.10 * val_filled
        + 0.07 * parsed
        + 0.03 * valid
    )


def _load_summary(out_dir: Path) -> Optional[Dict]:
    path = out_dir / "results" / "summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _run_eval(args: argparse.Namespace, cand: Candidate, out_dir: Path) -> int:
    eval_script = Path(__file__).with_name("eval_sudoku_samples.py")
    cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(cand.path),
        "--run_dir",
        str(args.run_dir),
        "--out_dir",
        str(out_dir),
        "--num_samples",
        str(args.num_samples),
        "--prompt_style",
        args.prompt_style,
        "--max_tokens",
        str(args.max_tokens),
        "--no_per_sample_print",
        "--resume",
    ]
    if args.splits:
        cmd.extend(["--splits", *args.splits])
    if args.single_gpu:
        cmd.append("--single_gpu")
    if args.gpus is not None:
        cmd.extend(["--gpus", str(args.gpus)])
    if args.max_error_dumps > 0:
        cmd.extend(["--max_error_dumps", str(args.max_error_dumps)])
    proc = subprocess.run(cmd, text=True)
    return int(proc.returncode)


def _row(cand: Candidate, summary: Dict, out_dir: Path) -> Dict:
    val = _split_summary(summary, "val")
    test = _split_summary(summary, "test")
    overall = summary.get("overall", summary)
    return {
        "score": _score(summary),
        "step": cand.step,
        "kind": cand.kind,
        "checkpoint": str(cand.path),
        "eval_dir": str(out_dir),
        "val_solve": float(val.get("fully_solved_pct", 0.0)),
        "test_solve": float(test.get("fully_solved_pct", 0.0)),
        "val_filled": float(val.get("filled_cell_acc", 0.0)),
        "test_filled": float(test.get("filled_cell_acc", 0.0)),
        "parsed": float(overall.get("parsed_pct", 0.0)),
        "valid_sudoku": float(overall.get("valid_sudoku_pct", 0.0)),
    }


def _write_outputs(rows: List[Dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    with (out_root / "checkpoint_ranking.json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    with (out_root / "checkpoint_ranking.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Sudoku SFT Checkpoint Ranking", ""]
    if rows:
        best = rows[0]
        lines.append(f"Best checkpoint: `{best['checkpoint']}`")
        lines.append("")
        lines.append("| rank | score | step | kind | val solve | test solve | val filled | test filled | parsed |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for i, row in enumerate(rows, start=1):
            lines.append(
                f"| {i} | {row['score']:.6f} | {row['step']} | {row['kind']} | "
                f"{row['val_solve']:.4f} | {row['test_solve']:.4f} | "
                f"{row['val_filled']:.4f} | {row['test_filled']:.4f} | {row['parsed']:.4f} |"
            )
    else:
        lines.append("No completed checkpoint evaluations.")
    (out_root / "checkpoint_ranking.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="SFT run directory, e.g. .../sft_train")
    parser.add_argument("--ckpt_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--max_ckpts", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--prompt_style", default="fixed", choices=["fixed", "random", "legacy_zh"])
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--single_gpu", action="store_true")
    parser.add_argument("--max_error_dumps", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_dir = args.ckpt_dir or (run_dir / "checkpoints")
    out_root = args.out_dir or (run_dir / "eval_logs" / "sudoku_ckpt_sweep")
    candidates = _discover_checkpoints(ckpt_dir, args.max_ckpts)
    if not candidates:
        print(f"[select_sudoku_sft_ckpt] no checkpoints found in {ckpt_dir}")
        return 2

    rows: List[Dict] = []
    for i, cand in enumerate(candidates, start=1):
        safe_stem = cand.path.stem.replace("/", "_")
        cand_out = out_root / f"{i:02d}_{cand.kind}_step{cand.step}_{safe_stem}"
        summary = _load_summary(cand_out) if args.skip_existing else None
        if summary is None:
            print(f"[select_sudoku_sft_ckpt] evaluating {i}/{len(candidates)}: {cand.path}")
            rc = _run_eval(args, cand, cand_out)
            if rc != 0:
                print(f"[select_sudoku_sft_ckpt] WARN eval failed rc={rc}: {cand.path}")
                continue
            summary = _load_summary(cand_out)
        if summary is None:
            print(f"[select_sudoku_sft_ckpt] WARN missing summary: {cand_out}")
            continue
        rows.append(_row(cand, summary, cand_out))

    if not rows:
        print("[select_sudoku_sft_ckpt] no successful eval rows")
        return 3
    _write_outputs(rows, out_root)
    best = max(rows, key=lambda r: r["score"])
    print(f"[select_sudoku_sft_ckpt] best_score={best['score']:.6f}")
    print(f"[select_sudoku_sft_ckpt] best_checkpoint={best['checkpoint']}")
    print(f"[select_sudoku_sft_ckpt] report={out_root / 'checkpoint_ranking.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
