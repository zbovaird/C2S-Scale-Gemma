"""Markdown writers for validation-bundle summaries."""

from __future__ import annotations

from pathlib import Path


def write_markdown_summary(output_path: Path, summary: dict) -> None:
    """Write a human-readable validation benchmark summary."""
    lines = [
        f"# Validation Benchmark: {summary.get('track_name', 'unknown')}",
        "",
        f"- dataset_profile: {summary.get('dataset_profile')}",
        f"- primary_metrics: {', '.join(summary.get('primary_metrics', []))}",
        "",
        "## Recommendation",
        "",
        f"- status: {summary.get('recommendation', {}).get('status')}",
        f"- preferred_alignment: {summary.get('recommendation', {}).get('preferred_alignment')}",
        f"- reason: {summary.get('recommendation', {}).get('reason')}",
        "",
        "## Interpretation Limits",
        "",
    ]
    for limit in summary.get("interpretation_limits", []):
        lines.append(f"- {limit}")
    lines.extend(
        [
            "",
            "## Runs",
            "",
        ]
    )
    recommendation_evidence = summary.get("recommendation", {}).get("evidence", {})
    top_supporting = recommendation_evidence.get("top_supporting_timepoints", [])
    top_concerning = recommendation_evidence.get("top_concerning_timepoints", [])
    if top_supporting:
        lines.extend(["## Recommendation Evidence: Supporting", ""])
        for row in top_supporting:
            lines.append(
                f"- {row['timepoint']}: "
                f"support_score={row['support_score']:.4f}, "
                f"delta_safe_fraction={row['delta_safe_fraction']:.4f}, "
                f"delta_productive_fraction={row['delta_productive_fraction']:.4f}, "
                f"delta_risk_fraction={row['delta_risk_fraction']:.4f}"
            )
        lines.append("")
    if top_concerning:
        lines.extend(["## Recommendation Evidence: Concerning", ""])
        for row in top_concerning:
            lines.append(
                f"- {row['timepoint']}: "
                f"concern_score={row['concern_score']:.4f}, "
                f"delta_safe_fraction={row['delta_safe_fraction']:.4f}, "
                f"delta_productive_fraction={row['delta_productive_fraction']:.4f}, "
                f"delta_risk_fraction={row['delta_risk_fraction']:.4f}"
            )
        lines.append("")
    for row in summary.get("runs", []):
        lines.append(
            f"- {row['label']} ({row['alignment_mode']}): "
            f"mean_l2_shift={row['mean_l2_shift']:.4f}, "
            f"mean_cosine_similarity={row['mean_cosine_similarity']:.4f}, "
            f"productive_fraction={row['productive_fraction']:.4f}, "
            f"safe_fraction={row['safe_fraction']:.4f}, "
            f"risk_fraction={row['risk_fraction']:.4f}"
        )
    for label, rows in summary.get("timepoint_summaries", {}).items():
        if not rows:
            continue
        lines.extend(["", f"## Timepoint Progression: {label}", ""])
        for row in rows:
            lines.append(
                f"- {row['timepoint']}: "
                f"mean_l2_shift={row['mean_l2_shift']:.4f}, "
                f"mean_progress_delta={row['mean_progress_delta']:.4f}, "
                f"productive_fraction={row['productive_fraction']:.4f}, "
                f"safe_fraction={row['safe_fraction']:.4f}, "
                f"risk_fraction={row['risk_fraction']:.4f}"
            )
    comparison_rows = summary.get("timepoint_comparison", [])
    if comparison_rows:
        lines.extend(["", "## Timepoint Comparison vs Euclidean", ""])
        for row in comparison_rows:
            lines.append(
                f"- {row['label']} @ {row['timepoint']}: "
                f"delta_mean_l2_shift={row['delta_mean_l2_shift']:.4f}, "
                f"delta_productive_fraction={row['delta_productive_fraction']:.4f}, "
                f"delta_safe_fraction={row['delta_safe_fraction']:.4f}, "
                f"delta_risk_fraction={row['delta_risk_fraction']:.4f}, "
                f"delta_mean_progress_delta={row['delta_mean_progress_delta']:.4f}"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")
