import argparse
import csv
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


LN_2 = math.log(2.0)
PROGRESS_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)")
LOGGED_STEP_RE = re.compile(r"(?:^|[,\s])step=([0-9.e+-]+)(?:$|[,\s])")
TRAIN_RE = re.compile(r"train_loss=([0-9.e+-]+)")
VALID_RE = re.compile(r"valid_loss=([0-9.e+-]+)")
TRAIN_BPB_RE = re.compile(r"(?:^|[,\s])train_bpb=([0-9.e+-]+)(?:$|[,\s])")
VALID_BPB_RE = re.compile(r"(?:^|[,\s])valid_bpb=([0-9.e+-]+)(?:$|[,\s])")


def extract_points(log_path: Path):
    points = {}
    with log_path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.replace("\r", "\n")
            for segment in line.split("\n"):
                if "train_loss=" not in segment and "valid_loss=" not in segment and "valid_bpb=" not in segment:
                    continue
                logged_step_match = LOGGED_STEP_RE.search(segment)
                progress_step_match = PROGRESS_STEP_RE.search(segment)
                if logged_step_match:
                    step = int(float(logged_step_match.group(1)))
                elif progress_step_match:
                    step = int(progress_step_match.group(1))
                else:
                    continue
                point = points.setdefault(step, {"step": step})

                train_match = TRAIN_RE.search(segment)
                if train_match:
                    point["train_loss"] = float(train_match.group(1))

                valid_match = VALID_RE.search(segment)
                if valid_match:
                    point["valid_loss"] = float(valid_match.group(1))

                train_bpb_match = TRAIN_BPB_RE.search(segment)
                if train_bpb_match:
                    point["train_bpb_logged"] = float(train_bpb_match.group(1))

                valid_bpb_match = VALID_BPB_RE.search(segment)
                if valid_bpb_match:
                    point["valid_bpb"] = float(valid_bpb_match.group(1))

    estimate_train_bpb(points)

    return [points[step] for step in sorted(points)]


def estimate_train_bpb(points):
    byte_constants = []
    for point in points.values():
        if "valid_loss" in point and "valid_bpb" in point and point["valid_bpb"] > 0:
            byte_constants.append(point["valid_loss"] / (LN_2 * point["valid_bpb"]))

    bytes_per_token = statistics.median(byte_constants) if byte_constants else 1.0

    for point in points.values():
        logged_train_bpb = point.get("train_bpb_logged")
        if logged_train_bpb is not None and logged_train_bpb > 0:
            point["train_bpb"] = logged_train_bpb
        elif "train_loss" in point:
            point["train_bpb"] = point["train_loss"] / (LN_2 * bytes_per_token)


def filter_points(points, min_step: int):
    return [point for point in points if point["step"] >= min_step]


def filter_points_range(points, min_step: int, max_step: int | None):
    filtered = [point for point in points if point["step"] >= min_step]
    if max_step is not None:
        filtered = [point for point in filtered if point["step"] <= max_step]
    return filtered


def smooth_series(values, window: int):
    if window <= 1 or len(values) <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def write_csv(ve_points, nove_points, output_path: Path):
    ve_map = {p["step"]: p for p in ve_points}
    nove_map = {p["step"]: p for p in nove_points}
    all_steps = sorted(set(ve_map) | set(nove_map))

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "optimizer_update_step",
                "ve_train_bpb",
                "ve_valid_bpb",
                "ve_valid_loss",
                "nove_train_bpb",
                "nove_valid_bpb",
                "nove_valid_loss",
            ]
        )
        for step in all_steps:
            ve = ve_map.get(step, {})
            nove = nove_map.get(step, {})
            writer.writerow(
                [
                    step,
                    f"{ve.get('train_bpb', float('nan')):.6f}" if "train_bpb" in ve else "",
                    f"{ve.get('valid_bpb', float('nan')):.6f}" if "valid_bpb" in ve else "",
                    f"{ve.get('valid_loss', float('nan')):.6f}" if "valid_loss" in ve else "",
                    f"{nove.get('train_bpb', float('nan')):.6f}" if "train_bpb" in nove else "",
                    f"{nove.get('valid_bpb', float('nan')):.6f}" if "valid_bpb" in nove else "",
                    f"{nove.get('valid_loss', float('nan')):.6f}" if "valid_loss" in nove else "",
                ]
            )


def plot_bpb(ve_points, nove_points, output_path: Path, smooth_window: int):
    plt.figure(figsize=(12, 6))
    ve_train = [p for p in ve_points if "train_bpb" in p]
    ve_valid = [p for p in ve_points if "valid_bpb" in p]
    nove_train = [p for p in nove_points if "train_bpb" in p]
    nove_valid = [p for p in nove_points if "valid_bpb" in p]

    plt.plot(
        [p["step"] for p in ve_train],
        smooth_series([p["train_bpb"] for p in ve_train], smooth_window),
        label="VE train_bpb (approx)",
        linewidth=2,
    )
    plt.plot(
        [p["step"] for p in ve_valid],
        smooth_series([p["valid_bpb"] for p in ve_valid], smooth_window),
        label="VE valid_bpb",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        [p["step"] for p in nove_train],
        smooth_series([p["train_bpb"] for p in nove_train], smooth_window),
        label="No-VE train_bpb (approx)",
        linewidth=2,
    )
    plt.plot(
        [p["step"] for p in nove_valid],
        smooth_series([p["valid_bpb"] for p in nove_valid], smooth_window),
        label="No-VE valid_bpb",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Optimizer Update Step")
    plt.ylabel("BPB")
    plt.title("VE vs No-VE BPB Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loss(ve_points, nove_points, output_path: Path, smooth_window: int):
    plt.figure(figsize=(12, 6))
    ve_valid = [p for p in ve_points if "valid_loss" in p]
    nove_valid = [p for p in nove_points if "valid_loss" in p]
    plt.plot(
        [p["step"] for p in ve_valid],
        smooth_series([p["valid_loss"] for p in ve_valid], smooth_window),
        label="VE valid_loss",
        linewidth=2,
    )
    plt.plot(
        [p["step"] for p in nove_valid],
        smooth_series([p["valid_loss"] for p in nove_valid], smooth_window),
        label="No-VE valid_loss",
        linewidth=2,
    )
    plt.xlabel("Optimizer Update Step")
    plt.ylabel("Loss")
    plt.title("VE vs No-VE Valid Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ve-log", type=Path, required=True)
    parser.add_argument("--nove-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    ve_points = filter_points_range(extract_points(args.ve_log), args.min_step, args.max_step)
    nove_points = filter_points_range(extract_points(args.nove_log), args.min_step, args.max_step)
    if not ve_points:
        raise SystemExit(f"No parsable points found in VE log: {args.ve_log}")
    if not nove_points:
        raise SystemExit(f"No parsable points found in no-VE log: {args.nove_log}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_parts = []
    if args.min_step > 0:
        suffix_parts.append(f"from_step_{args.min_step}")
    if args.max_step is not None:
        suffix_parts.append(f"to_step_{args.max_step}")
    if args.smooth_window > 1:
        suffix_parts.append(f"smooth_{args.smooth_window}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    csv_path = args.output_dir / f"ve_vs_nove_curves_{stamp}{suffix}.csv"
    bpb_png = args.output_dir / f"ve_vs_nove_bpb_{stamp}{suffix}.png"
    loss_png = args.output_dir / f"ve_vs_nove_loss_{stamp}{suffix}.png"

    write_csv(ve_points, nove_points, csv_path)
    plot_bpb(ve_points, nove_points, bpb_png, args.smooth_window)
    plot_loss(ve_points, nove_points, loss_png, args.smooth_window)

    print(csv_path)
    print(bpb_png)
    print(loss_png)


if __name__ == "__main__":
    main()
