#!/usr/bin/env python3
"""Generate static plots and a markdown report from perturbation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from eval.alignment_ablation import load_ablation_dirs_from_manifest
from eval.perturbation_report import (
    get_top_shift_rows,
    summarize_alignment_ablation,
    summarize_boolean_flag,
    summarize_risk_by_branch,
    summarize_shift_by_category,
    summarize_value_by_category,
)


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


def plot_risk_by_branch(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    branches = [row["branch_label"] for row in summary_rows]
    mean_risks = [row["mean_risk_score"] for row in summary_rows]

    plt.figure(figsize=(9, 5))
    plt.bar(branches, mean_risks, color="#8172B3")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean risk score")
    plt.title("Average risk score by inferred branch")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_progress_vs_risk(rows: list[dict], output_path: Path) -> None:
    filtered_rows = [
        row
        for row in rows
        if row.get("progress_delta") is not None and row.get("risk_score") is not None
    ]
    if not filtered_rows:
        return

    x_values = [float(row["progress_delta"]) for row in filtered_rows]
    y_values = [float(row["risk_score"]) for row in filtered_rows]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, alpha=0.6, s=18, color="#937860")
    plt.xlabel("Progress delta")
    plt.ylabel("Risk score")
    plt.title("Reprogramming progress vs risk proxy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_zone_counts(zone_summaries: list[dict], output_path: Path) -> None:
    labels = [summary["flag_key"] for summary in zone_summaries]
    counts = [summary["count"] for summary in zone_summaries]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, counts, color=["#64B5CD", "#4C72B0", "#C44E52"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Cell count")
    plt.title("Cells in partial-window / safe-zone / risk cohorts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_marker_panel_scatter(rows: list[dict], output_path: Path) -> None:
    filtered_rows = [
        row
        for row in rows
        if row.get("rejuvenation_score") is not None
        and row.get("pluripotency_marker_score") is not None
    ]
    if not filtered_rows:
        return

    x_values = [float(row["rejuvenation_score"]) for row in filtered_rows]
    y_values = [float(row["pluripotency_marker_score"]) for row in filtered_rows]
    colors = [float(row.get("risk_score", 0.0) or 0.0) for row in filtered_rows]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, c=colors, cmap="viridis", alpha=0.7, s=20)
    plt.xlabel("Rejuvenation panel score")
    plt.ylabel("Pluripotency-risk marker score")
    plt.title("Marker panel balance across perturbed cells")
    plt.colorbar(label="Risk score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_alignment_ablation(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    labels = [row["label"] for row in summary_rows]
    metric_values = [row["metric_value"] for row in summary_rows]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, metric_values, color="#4C72B0")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean fused embedding L2 shift")
    plt.title("Alignment-mode ablation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_markdown_report(
    output_path: Path,
    embedding_summary: dict,
    top_shift_rows: list[dict],
    category_summary: list[dict],
    risk_summary: list[dict],
    zone_summaries: list[dict],
    marker_summary: dict[str, list[dict]],
    ablation_summary: list[dict],
    overlay_summary: dict | None,
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

    if overlay_summary is not None:
        lines.extend(
            [
                "## Reprogramming Heuristic Profile",
                "",
                "### Dataset profile",
                "",
                "```json",
                json.dumps(
                    {
                        "dataset_profile": overlay_summary.get("dataset_profile"),
                        "dataset_manifest": overlay_summary.get("dataset_manifest"),
                    },
                    indent=2,
                ),
                "```",
                "",
                "### Reference labels",
                "",
                "```json",
                json.dumps(overlay_summary.get("reference_labels", {}), indent=2),
                "```",
                "",
                "### Window profile",
                "",
                "```json",
                json.dumps(overlay_summary.get("window_profile", {}), indent=2),
                "```",
                "",
            ]
        )

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

    lines.extend(["", "## Risk by Branch", ""])
    for row in risk_summary:
        lines.append(
            f"- {row['branch_label']}: mean_risk={row['mean_risk_score']:.4f}, "
            f"max_risk={row['max_risk_score']:.4f}, n={row['count']}"
        )

    lines.extend(["", "## Partial Reprogramming and Safety Heuristics", ""])
    for row in zone_summaries:
        lines.append(
            f"- {row['flag_key']}: count={row['count']}, fraction={row['fraction']:.4f}"
        )

    lines.extend(["", "## Marker Panel Summary", ""])
    for summary_name, rows in marker_summary.items():
        lines.append(f"### {summary_name}")
        for row in rows:
            lines.append(
                f"- {row['category']}: mean={row['mean_value']:.4f}, "
                f"max={row['max_value']:.4f}, n={row['count']}"
            )
        lines.append("")

    if ablation_summary:
        lines.extend(["## Alignment Ablation", ""])
        for row in ablation_summary:
            lines.append(
                f"- {row['label']} ({row['alignment_mode']}): "
                f"{row['metric_key']}={row['metric_value']:.4f}, "
                f"mean_cosine_similarity={row['mean_cosine_similarity']:.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "- `shift_histogram.png`",
            "- `oskm_score_vs_shift.png`",
            "- `shift_by_cell_type.png`",
            "- `risk_by_branch.png`",
            "- `progress_vs_risk.png`",
            "- `zone_counts.png`",
            "- `marker_panel_balance.png`",
            "- `alignment_ablation.png`",
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
    parser.add_argument(
        "--ablation-comparison-dir",
        action="append",
        default=None,
        help="Optional additional comparison directory to include in alignment ablations",
    )
    parser.add_argument(
        "--ablation-manifest",
        type=str,
        default=None,
        help="Optional manifest from run_alignment_ablation.py",
    )
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    output_dir = Path(args.output_dir) if args.output_dir else comparison_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_summary = load_json(comparison_dir / "embedding_shift_summary.json")
    fused_shift_rows = load_json(comparison_dir / "fused_embedding_shift_frame.json")
    overlay_summary_path = comparison_dir / "reprogramming_overlay_summary.json"
    overlay_summary = load_json(overlay_summary_path) if overlay_summary_path.exists() else None
    perturbation_summary = (
        load_json(Path(args.perturbation_summary))
        if args.perturbation_summary is not None
        else None
    )

    category_summary = summarize_shift_by_category(fused_shift_rows, "cell_type")
    risk_summary = summarize_risk_by_branch(fused_shift_rows)
    zone_summaries = [
        summarize_boolean_flag(fused_shift_rows, "partial_reprogramming_window"),
        summarize_boolean_flag(fused_shift_rows, "longevity_safe_zone"),
        summarize_boolean_flag(fused_shift_rows, "pluripotency_risk_flag"),
    ]
    marker_summary = {
        "rejuvenation_by_branch": summarize_value_by_category(
            fused_shift_rows,
            "branch_label",
            "rejuvenation_score",
        ),
        "pluripotency_markers_by_branch": summarize_value_by_category(
            fused_shift_rows,
            "branch_label",
            "pluripotency_marker_score",
        ),
    }
    top_shift_rows = get_top_shift_rows(fused_shift_rows, top_n=15)
    ablation_runs = []
    if overlay_summary is not None:
        ablation_runs.append(
            {
                "label": comparison_dir.name,
                "alignment_mode": overlay_summary.get("alignment", {}).get(
                    "alignment_mode",
                    "unknown",
                ),
                "dataset_profile": overlay_summary.get("dataset_profile"),
                "embedding_summary": embedding_summary,
            }
        )
    ablation_dirs = [Path(path) for path in (args.ablation_comparison_dir or [])]
    if args.ablation_manifest is not None:
        ablation_dirs.extend(
            load_ablation_dirs_from_manifest(Path(args.ablation_manifest), comparison_dir)
        )
    if ablation_dirs:
        for candidate_path in ablation_dirs:
            candidate_embedding_summary = load_json(
                candidate_path / "embedding_shift_summary.json"
            )
            candidate_overlay_path = candidate_path / "reprogramming_overlay_summary.json"
            candidate_overlay = (
                load_json(candidate_overlay_path)
                if candidate_overlay_path.exists()
                else {}
            )
            ablation_runs.append(
                {
                    "label": candidate_path.name,
                    "alignment_mode": candidate_overlay.get("alignment", {}).get(
                        "alignment_mode",
                        "unknown",
                    ),
                    "dataset_profile": candidate_overlay.get("dataset_profile"),
                    "embedding_summary": candidate_embedding_summary,
                }
            )
    ablation_summary = summarize_alignment_ablation(ablation_runs)

    plot_shift_histogram(fused_shift_rows, output_dir / "shift_histogram.png")
    plot_oskm_scatter(fused_shift_rows, output_dir / "oskm_score_vs_shift.png")
    plot_category_bars(category_summary, output_dir / "shift_by_cell_type.png")
    plot_risk_by_branch(risk_summary, output_dir / "risk_by_branch.png")
    plot_progress_vs_risk(fused_shift_rows, output_dir / "progress_vs_risk.png")
    plot_zone_counts(zone_summaries, output_dir / "zone_counts.png")
    plot_marker_panel_scatter(fused_shift_rows, output_dir / "marker_panel_balance.png")
    plot_alignment_ablation(ablation_summary, output_dir / "alignment_ablation.png")
    write_markdown_report(
        output_dir / "OSKM_PERTURBATION_REPORT.md",
        embedding_summary=embedding_summary,
        top_shift_rows=top_shift_rows,
        category_summary=category_summary,
        risk_summary=risk_summary,
        zone_summaries=zone_summaries,
        marker_summary=marker_summary,
        ablation_summary=ablation_summary,
        overlay_summary=overlay_summary,
        perturbation_summary=perturbation_summary,
    )

    logger.info("Generated OSKM perturbation report in %s", output_dir)


if __name__ == "__main__":
    main()
