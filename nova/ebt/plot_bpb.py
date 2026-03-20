#!/usr/bin/env python3
"""Live-updating plot of valid_bpb_loss from lightning_logs/version_*/metrics.csv."""

import argparse
import signal
import sys
import time
from pathlib import Path


def find_latest_metrics_csv(logs_dir: Path) -> Path | None:
    versions = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("version_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    for version_dir in reversed(versions):
        csv = version_dir / "metrics.csv"
        if csv.exists():
            return csv
    return None


def plot_once(csv_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    val = df[["step", "valid_bpb_loss"]].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(val["step"], val["valid_bpb_loss"], marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("valid_bpb_loss")
    ax.set_title(f"valid_bpb_loss — {csv_path.parent.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-plot valid_bpb_loss from lightning_logs")
    parser.add_argument("--logs-dir", default="lightning_logs", help="Path to lightning_logs dir")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <logs_dir>/bpb_live.png)")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output) if args.output else logs_dir / "bpb_live.png"

    def handle_sigint(sig, frame):
        print(f"\nExiting. Plot saved to: {output_path.resolve()}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Watching {logs_dir.resolve()} every {args.interval}s — output: {output_path.resolve()}")
    print("Press Ctrl+C to stop.")

    while True:
        csv_path = find_latest_metrics_csv(logs_dir)
        if csv_path is None:
            print("No metrics.csv found yet, waiting...")
        else:
            try:
                plot_once(csv_path, output_path)
                print(f"[{time.strftime('%H:%M:%S')}] Updated {output_path} from {csv_path.parent.name}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
