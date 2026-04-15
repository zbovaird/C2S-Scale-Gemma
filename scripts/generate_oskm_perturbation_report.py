#!/usr/bin/env python3
"""Generate static plots and a markdown report from perturbation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from eval.perturbation_report import get_top_shift_rows, summarize_shift_by_category


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_shift_histogram(rows: list[dict], output_path: Path) -> None:
    values = [float(row.get("l2_shift", 0.0)) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20, color="#4C72B0", edgecolor="white")
    plt.xlabel("Fused embedding L2 shift")
    plt.ylabel("Cell count")
    plt.title("Distribution of perturbation-induced embedding shifts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_oskm_scatter(rows: list[dict], output_path: Path) -> None:
    x_values = [float(row.get("baseline_oskm_score", 0.0)) for row in rows]
    y_values = [float(row.get("l2_shift", 0.0)) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, alpha=0.6, s=18, color="#55A868")
    plt.xlabel("Baseline OSKM score")
    plt.ylabel("Fused embedding L2 shift")
    plt.title("OSKM baseline score vs embedding shift")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_category_bars(summary_rows: list[dict], output_path: Path) -> None:
    categories = [row["category"] for row in summary_rows]
    mean_shifts = [row["mean_shift"] for row in summary_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(categories, mean_shifts, color="#C44E52")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean fused embedding shift")
    plt.title("Average perturbation response by category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_markdown_report(
    output_path: Path,
    embedding_summary: dict,
    top_shift_rows: list[dict],
    category_summary: list[dict],
    perturbation_summary: dict | None,
) -> None:
    lines = [
        "# OSKM Perturbation Report",
        "",
        "## Embedding Shift Summary",
        "",
    ]

    for representation, summary in embedding_summary.items():
        lines.extend(
            [
                f"### {representation}",
                f"- Mean L2 shift: {summary['mean_l2_shift']:.4f}",
                f"- Median L2 shift: {summary['median_l2_shift']:.4f}",
                f"- Max L2 shift: {summary['max_l2_shift']:.4f}",
                f"- Mean cosine similarity: {summary['mean_cosine_similarity']:.4f}",
                "",
            ]
        )

    if perturbation_summary is not None:
        lines.extend(["## Perturbation Metadata", "", "```json", json.dumps(perturbation_summary, indent=2), "```", ""])

    lines.extend(["## Top Shifted Cells", ""])
    for row in top_shift_rows:
        lines.append(
            f"- cell_id={row.get('cell_id', row['cell_index'])}, "
            f"cell_type={row.get('cell_type', 'unknown')}, "
            f"l2_shift={float(row.get('l2_shift', 0.0)):.4f}"
        )

    lines.extend(["", "## Shift by Category", ""])
    for row in category_summary:
        lines.append(
            f"- {row['category']}: mean={row['mean_shift']:.4f}, "
            f"median={row['median_shift']:.4f}, n={row['count']}"
        )

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `shift_histogram.png`",
            "- `oskm_score_vs_shift.png`",
            "- `shift_by_cell_type.png`",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an OSKM perturbation report")
    parser.add_argument(
        "--comparison-dir",
        type=str,
        required=True,
        help="Directory produced by compare_oskm_perturbation_embeddings.py",
    )
    parser.add_argument(
        "--perturbation-summary",
        type=str,
        default=None,
        help="Optional perturbation summary JSON from perturb_oskm_expression.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report files (defaults to comparison dir)",
    )
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    output_dir = Path(args.output_dir) if args.output_dir else comparison_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_summary = load_json(comparison_dir / "embedding_shift_summary.json")
    fused_shift_rows = load_json(comparison_dir / "fused_embedding_shift_frame.json")
    perturbation_summary = (
        load_json(Path(args.perturbation_summary))
        if args.perturbation_summary is not None
        else None
    )

    category_summary = summarize_shift_by_category(fused_shift_rows, "cell_type")
    top_shift_rows = get_top_shift_rows(fused_shift_rows, top_n=15)

    plot_shift_histogram(fused_shift_rows, output_dir / "shift_histogram.png")
    plot_oskm_scatter(fused_shift_rows, output_dir / "oskm_score_vs_shift.png")
    plot_category_bars(category_summary, output_dir / "shift_by_cell_type.png")
    write_markdown_report(
        output_dir / "OSKM_PERTURBATION_REPORT.md",
        embedding_summary=embedding_summary,
        top_shift_rows=top_shift_rows,
        category_summary=category_summary,
        perturbation_summary=perturbation_summary,
    )

    logger.info("Generated OSKM perturbation report in %s", output_dir)


if __name__ == "__main__":
    main()
