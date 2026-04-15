"""Helpers for summarizing validation-bundle outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence


def load_json_file(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_validation_benchmark_rows(
    validation_manifest: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> list[dict]:
    """Build benchmark rows for each run in a validation bundle."""
    primary_metrics = validation_manifest.get("report_recommendation", {}).get(
        "primary_metrics",
        [],
    )
    rows = []
    for payload in run_payloads:
        embedding_summary = payload.get("embedding_summary", {}).get("fused_embeddings", {})
        overlay_summary = payload.get("overlay_summary", {})
        branch_summary = overlay_summary.get("branch_summary", {})
        zone_summary = overlay_summary.get("zone_summary", {})
        total_branch_count = sum(
            int(branch.get("count", 0)) for branch in branch_summary.values()
        )
        productive_count = int(branch_summary.get("productive", {}).get("count", 0))

        row = {
            "label": payload.get("label", "unknown"),
            "alignment_mode": payload.get("alignment_mode", "unknown"),
            "dataset_profile": payload.get("dataset_profile"),
            "mean_l2_shift": float(embedding_summary.get("mean_l2_shift", 0.0)),
            "mean_cosine_similarity": float(
                embedding_summary.get("mean_cosine_similarity", 0.0)
            ),
            "productive_fraction": (
                productive_count / total_branch_count if total_branch_count else 0.0
            ),
            "safe_fraction": float(zone_summary.get("longevity_safe_zone_fraction", 0.0)),
            "risk_fraction": float(zone_summary.get("pluripotency_risk_fraction", 0.0)),
            "primary_metrics": list(primary_metrics),
        }
        rows.append(row)
    return rows


def build_timepoint_progression_rows(
    *,
    track_config: Dict[str, Any],
    fused_shift_rows: Sequence[Dict[str, Any]],
) -> list[dict]:
    """Aggregate perturbation/trajectory summaries by configured timepoint."""
    timepoint_column = track_config.get("timepoint_column")
    if not timepoint_column:
        return []

    grouped_rows: dict[str, list[Dict[str, Any]]] = {}
    for row in fused_shift_rows:
        timepoint = str(row.get(timepoint_column, "unknown"))
        grouped_rows.setdefault(timepoint, []).append(row)

    expected_timepoints = [str(value) for value in track_config.get("expected_timepoints", [])]
    ordered_timepoints = expected_timepoints + sorted(
        timepoint for timepoint in grouped_rows.keys() if timepoint not in expected_timepoints
    )

    summary_rows = []
    for timepoint in ordered_timepoints:
        rows = grouped_rows.get(timepoint)
        if not rows:
            continue
        n_cells = len(rows)
        productive_count = sum(
            1 for row in rows if str(row.get("branch_label", "")) == "productive"
        )
        safe_count = sum(bool(row.get("longevity_safe_zone")) for row in rows)
        risk_count = sum(bool(row.get("pluripotency_risk_flag")) for row in rows)
        mean_l2_shift = sum(float(row.get("l2_shift", 0.0)) for row in rows) / n_cells
        mean_progress_delta = sum(float(row.get("progress_delta", 0.0)) for row in rows) / n_cells
        summary_rows.append(
            {
                "timepoint": timepoint,
                "n_cells": n_cells,
                "mean_l2_shift": mean_l2_shift,
                "mean_progress_delta": mean_progress_delta,
                "productive_fraction": productive_count / n_cells,
                "safe_fraction": safe_count / n_cells,
                "risk_fraction": risk_count / n_cells,
            }
        )
    return summary_rows


def build_validation_benchmark_summary(
    validation_manifest: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact benchmark summary for a validation bundle."""
    rows = build_validation_benchmark_rows(validation_manifest, run_payloads)
    rows_sorted = sorted(rows, key=lambda row: row["mean_l2_shift"])
    best_run = rows_sorted[0] if rows_sorted else None
    track_config = validation_manifest.get("track", {})
    timepoint_summaries = {}
    for payload in run_payloads:
        label = str(payload.get("label", "unknown"))
        timepoint_summaries[label] = build_timepoint_progression_rows(
            track_config=track_config,
            fused_shift_rows=payload.get("fused_shift_rows", []),
        )
    timepoint_comparison = build_timepoint_comparison_rows(timepoint_summaries)
    return {
        "track_name": validation_manifest.get("track_name"),
        "dataset_profile": validation_manifest.get("dataset_profile"),
        "primary_metrics": validation_manifest.get("report_recommendation", {}).get(
            "primary_metrics",
            [],
        ),
        "runs": rows_sorted,
        "best_by_mean_l2_shift": best_run,
        "timepoint_summaries": timepoint_summaries,
        "timepoint_comparison": timepoint_comparison,
    }


def build_timepoint_comparison_rows(
    timepoint_summaries: Dict[str, Sequence[Dict[str, Any]]],
    baseline_label: str = "euclidean",
) -> list[dict]:
    """Compare timepoint summaries against a baseline alignment run."""
    baseline_rows = {
        str(row["timepoint"]): row for row in timepoint_summaries.get(baseline_label, [])
    }
    comparison_rows = []
    for label, rows in timepoint_summaries.items():
        if label == baseline_label:
            continue
        for row in rows:
            timepoint = str(row["timepoint"])
            baseline_row = baseline_rows.get(timepoint)
            if baseline_row is None:
                continue
            comparison_rows.append(
                {
                    "label": label,
                    "baseline_label": baseline_label,
                    "timepoint": timepoint,
                    "delta_mean_l2_shift": float(row["mean_l2_shift"])
                    - float(baseline_row["mean_l2_shift"]),
                    "delta_productive_fraction": float(row["productive_fraction"])
                    - float(baseline_row["productive_fraction"]),
                    "delta_safe_fraction": float(row["safe_fraction"])
                    - float(baseline_row["safe_fraction"]),
                    "delta_risk_fraction": float(row["risk_fraction"])
                    - float(baseline_row["risk_fraction"]),
                    "delta_mean_progress_delta": float(row["mean_progress_delta"])
                    - float(baseline_row["mean_progress_delta"]),
                }
            )
    return comparison_rows
