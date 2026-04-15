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


def build_validation_benchmark_summary(
    validation_manifest: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact benchmark summary for a validation bundle."""
    rows = build_validation_benchmark_rows(validation_manifest, run_payloads)
    rows_sorted = sorted(rows, key=lambda row: row["mean_l2_shift"])
    best_run = rows_sorted[0] if rows_sorted else None
    return {
        "track_name": validation_manifest.get("track_name"),
        "dataset_profile": validation_manifest.get("dataset_profile"),
        "primary_metrics": validation_manifest.get("report_recommendation", {}).get(
            "primary_metrics",
            [],
        ),
        "runs": rows_sorted,
        "best_by_mean_l2_shift": best_run,
    }
