#!/usr/bin/env python3
"""Render publication-style plots from a validation trajectory projection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_projection_visuals import (
    build_projection_arrow_rows,
    build_projection_phase_series,
)
from eval.validation_summary import load_json_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BRANCH_COLORS = {
    "productive": "#2563eb",
    "alternative": "#dc2626",
    "somatic_retention": "#d97706",
    "ambiguous": "#7c3aed",
    "unknown": "#64748b",
}

SAFE_ZONE_COLORS = {
    "safe_zone": "#059669",
    "not_safe": "#94a3b8",
}


def _plot_scatter_with_arrows(
    *,
    series: dict[str, list[dict]],
    arrows: list[dict],
    color_map: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    for arrow in arrows:
        plt.plot(
            [arrow["x0"], arrow["x1"]],
            [arrow["y0"], arrow["y1"]],
            color="#cbd5e1",
            linewidth=0.7,
            alpha=0.6,
            zorder=1,
        )
    for label, rows in series.items():
        if not rows:
            continue
        plt.scatter(
            [row["x"] for row in rows],
            [row["y"] for row in rows],
            s=24,
            label=label,
            color=color_map.get(label, "#64748b"),
            alpha=0.9,
            zorder=2,
        )
    plt.xlabel("Projection 1")
    plt.ylabel("Projection 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation trajectory projections")
    parser.add_argument(
        "--projection-path",
        type=str,
        required=True,
        help="Path to validation_trajectory_projection.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to the projection file directory)",
    )
    args = parser.parse_args()

    projection_path = Path(args.projection_path)
    projection = load_json_file(projection_path)
    output_dir = Path(args.output_dir) if args.output_dir else projection_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in projection.get("runs", []):
        label = str(run.get("label", "unknown"))
        arrows = build_projection_arrow_rows(run)
        branch_series = build_projection_phase_series(
            run,
            phase="perturbed",
            category_key="branch_label",
        )
        safe_series = build_projection_phase_series(
            {
                "rows": [
                    dict(
                        row,
                        safe_zone_label=(
                            "safe_zone" if bool(row.get("longevity_safe_zone")) else "not_safe"
                        ),
                    )
                    for row in run.get("rows", [])
                ]
            },
            phase="perturbed",
            category_key="safe_zone_label",
        )
        _plot_scatter_with_arrows(
            series=branch_series,
            arrows=arrows,
            color_map=BRANCH_COLORS,
            title=f"Trajectory projection by branch: {label}",
            output_path=output_dir / f"validation_projection_branch_{label}.png",
        )
        _plot_scatter_with_arrows(
            series=safe_series,
            arrows=arrows,
            color_map=SAFE_ZONE_COLORS,
            title=f"Trajectory projection by safe zone: {label}",
            output_path=output_dir / f"validation_projection_safe_zone_{label}.png",
        )

    logger.info("Saved validation trajectory projection plots to %s", output_dir)


if __name__ == "__main__":
    main()
