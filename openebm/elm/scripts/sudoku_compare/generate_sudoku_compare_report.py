#!/usr/bin/env python3
"""Write a compact Markdown report for a Sudoku comparison run."""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_env_file(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.is_file():
        return {}
    out: Dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key] = value
    return out


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"comparison CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    except ValueError:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(str(value)))
    except ValueError:
        return default


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def metric(row: Dict[str, str], key: str) -> float:
    aliases = {
        "board_acc": ("board_acc", "fully_solved_pct"),
        "cell_acc": ("cell_acc", "filled_cell_acc"),
        "valid_rate": ("valid_rate", "parsed_pct"),
        "valid_sudoku_pct": ("valid_sudoku_pct", "valid_sudoku_rate"),
        "avg_time": ("avg_time", "avg_s_per_sample"),
    }
    for candidate in aliases.get(key, (key,)):
        if row.get(candidate) not in (None, ""):
            return parse_float(row.get(candidate))
    return 0.0


def model_name(row: Dict[str, str]) -> str:
    return row.get("model") or "unknown"


def sort_key(row: Dict[str, str]) -> Tuple[float, float, float, float]:
    return (
        metric(row, "board_acc"),
        metric(row, "cell_acc"),
        metric(row, "valid_rate"),
        -metric(row, "avg_time"),
    )


def select_best(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    return max(rows, key=sort_key)


def has_any(rows: Iterable[Dict[str, str]], key: str) -> bool:
    return any(row.get(key) not in (None, "") for row in rows)


def markdown_table(rows: Sequence[Dict[str, str]]) -> str:
    include_missing = has_any(rows, "missing_n")
    include_valid_sudoku = has_any(rows, "valid_sudoku_pct")
    headers = ["model", "params", "n"]
    if include_missing:
        headers.append("missing")
    headers.extend(["board_acc", "cell_acc", "valid_rate"])
    if include_valid_sudoku:
        headers.append("valid_sudoku")
    headers.append("avg_time")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    ordered = sorted(rows, key=sort_key, reverse=True)
    for row in ordered:
        values = [
            md_escape(model_name(row)),
            md_escape(row.get("params", "")),
            str(parse_int(row.get("n"))),
        ]
        if include_missing:
            values.append(str(parse_int(row.get("missing_n"))))
        values.extend(
            [
                pct(metric(row, "board_acc")),
                pct(metric(row, "cell_acc")),
                pct(metric(row, "valid_rate")),
            ]
        )
        if include_valid_sudoku:
            values.append(pct(metric(row, "valid_sudoku_pct")))
        values.append(f"{metric(row, 'avg_time'):.3f}s")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_checkpoint_meta(path: Path) -> Dict[str, str]:
    name = path.name
    meta: Dict[str, str] = {"path": str(path)}
    step = re.search(r"step=(\d+)", name)
    if step:
        meta["step"] = step.group(1)
    losses = re.findall(r"valid_loss(?:_[A-Za-z0-9]+)?=([0-9]+(?:\.[0-9]+)?)", name)
    if losses:
        meta["valid_loss"] = losses[-1]
    lr = re.search(r"lr([0-9]+(?:\.[0-9]+)?(?:e[+-]?\d+)?)", name, flags=re.I)
    if lr:
        meta["lr"] = lr.group(1)
    batch = re.search(r"bs([A-Za-z0-9x]+)", name)
    if batch:
        meta["batch"] = batch.group(1)
    if "muon_adamw" in name:
        meta["optimizer"] = "muon_adamw"
    return meta


def config_item(
    env: Dict[str, str],
    key: str,
    label: Optional[str] = None,
    default: str = "",
) -> str:
    return f"- {label or key}: `{env.get(key, default)}`"


def conclusion(rows: Sequence[Dict[str, str]]) -> str:
    best = select_best(rows)
    if best is None:
        return "No model metrics were found."
    ordered = sorted(rows, key=sort_key, reverse=True)
    best_name = model_name(best)
    sentence = (
        f"{best_name} is best by board accuracy, with "
        f"`board_acc={pct(metric(best, 'board_acc'))}`, "
        f"`cell_acc={pct(metric(best, 'cell_acc'))}`, and "
        f"`valid_rate={pct(metric(best, 'valid_rate'))}`."
    )
    if len(ordered) > 1:
        runner = ordered[1]
        sentence += (
            f" The next model is {model_name(runner)} at "
            f"`board_acc={pct(metric(runner, 'board_acc'))}`."
        )
    sentence += (
        " The ranking is determined by `board_acc`, with `cell_acc`, "
        "`valid_rate`, and lower `avg_time` used as tie-breakers."
    )
    return sentence


def write_report(args: argparse.Namespace) -> None:
    rows = read_rows(args.comparison_csv)
    env = read_env_file(args.run_env)
    ckpt_meta = parse_checkpoint_meta(args.checkpoint)
    out = args.out_md
    out.parent.mkdir(parents=True, exist_ok=True)

    report_label = args.report_label or env.get("RUN_ID", "Sudoku comparison")
    data_dir = args.data_dir or Path(env.get("DATA_DIR", ""))
    results_root = args.results_root or Path(env.get("RUN_OUTPUT_ROOT", ""))
    report_dir = args.report_dir or Path(env.get("REPORT_OUT_DIR", out.parent))

    lines = [
        f"# Sudoku Model Comparison Report - {report_label}",
        "",
        f"Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        "",
        "## Metric Comparison",
        "",
        markdown_table(rows),
        "",
        "Metric notes: `board_acc` is full-grid solve accuracy; `cell_acc` is "
        "blank-cell accuracy where supported; `valid_rate` is the parsed-output "
        "rate used by the existing comparison scripts.",
        "",
        "## Conclusion",
        "",
        conclusion(rows),
        "",
        "## Run Configuration",
        "",
        "- EBM checkpoint: `" + ckpt_meta["path"] + "`",
    ]
    for key in ["step", "valid_loss", "lr", "batch", "optimizer"]:
        if key in ckpt_meta:
            lines.append(f"- checkpoint_{key}: `{ckpt_meta[key]}`")
    lines.extend(
        [
            f"- data_dir: `{data_dir}`",
            "- split: `test`",
            config_item(env, "START_INDEX", default="0"),
            config_item(env, "NUM_SAMPLES", default="-1"),
            config_item(env, "STAGE_SIZE", default="0"),
            config_item(env, "LLM_BACKEND", default="vllm"),
            config_item(env, "THINKING", default="disable"),
            config_item(env, "TEMPERATURE", default="0"),
            config_item(env, "TOP_P", default="1.0"),
            config_item(env, "SMALL_MODELS", default="qwen3_1p7b llama3p2_1b"),
            config_item(env, "QWEN27_MODEL", default="qwen3p6_27b"),
            config_item(env, "R1_MODEL", default="deepseek_r1_0528"),
            config_item(env, "SMALL_MAX_TOKENS", default="2048"),
            config_item(env, "QWEN27_MAX_TOKENS", default="4096"),
            config_item(env, "R1_MAX_TOKENS", default="8192"),
            config_item(env, "EBM_DTYPE", default="float32"),
            config_item(env, "EBM_GPUS", default="0"),
            "",
            "## Output Paths",
            "",
            f"- raw_outputs: `{results_root}`",
            f"- comparison_csv: `{args.comparison_csv}`",
            f"- report_dir: `{report_dir}`",
            f"- this_report: `{out}`",
        ]
    )
    out.write_text("\n".join(lines) + "\n")
    print(f"[generate_sudoku_compare_report] wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--run-env", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--report-label", default="")
    return parser


def main() -> None:
    write_report(build_parser().parse_args())


if __name__ == "__main__":
    main()
