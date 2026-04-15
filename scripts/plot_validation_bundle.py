#!/usr/bin/env python3
"""Generate trajectory-centric plots from a validation benchmark summary."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_summary import load_json_file
from eval.validation_visuals import (
    build_timepoint_delta_series,
    build_timepoint_metric_series,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_timepoint_series(series: dict[str, list[dict]], output_path: Path, metric_label: str) -> None:
    if not series:
        return
    plt.figure(figsize=(9, 5))
    for label, rows in series.items():
        if not rows:
            continue
        x_values = [row["timepoint"] for row in rows]
        y_values = [row["value"] for row in rows]
        plt.plot(x_values, y_values, marker="o", label=label)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} across timepoints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_timepoint_deltas(rows: list[dict], output_path: Path, metric_label: str) -> None:
    if not rows:
        return
    x_values = [row["timepoint"] for row in rows]
    y_values = [row["value"] for row in rows]
    colors = ["#4C72B0" if value >= 0 else "#C44E52" for value in y_values]

    plt.figure(figsize=(9, 5))
    plt.bar(x_values, y_values, color=colors)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs Euclidean by timepoint")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation bundle trajectories")
    parser.add_argument(
        "--summary-path",
        type=str,
        required=True,
        help="Path to validation_benchmark_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to summary directory)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    summary = load_json_file(summary_path)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_series = build_timepoint_metric_series(summary, "mean_progress_delta")
    safe_series = build_timepoint_metric_series(summary, "safe_fraction")
    delta_series = build_timepoint_delta_series(summary, "delta_safe_fraction")

    plot_timepoint_series(
        progress_series,
        output_dir / "validation_timepoint_progress_delta.png",
        "Mean progress delta",
    )
    plot_timepoint_series(
        safe_series,
        output_dir / "validation_timepoint_safe_fraction.png",
        "Safe fraction",
    )
    plot_timepoint_deltas(
        delta_series,
        output_dir / "validation_timepoint_safe_delta.png",
        "Safe fraction delta",
    )

    logger.info("Saved validation trajectory plots to %s", output_dir)


if __name__ == "__main__":
    main()
