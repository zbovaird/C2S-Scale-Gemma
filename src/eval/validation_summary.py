"""Helpers for summarizing validation-bundle outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

DEFAULT_INTERPRETATION_LIMITS = [
    "Validation metrics are representation-level evidence, not proof of safe biological reprogramming.",
    "Partial-window and longevity-safe-zone labels are heuristics that require independent biological validation.",
    "Projection views are qualitative diagnostics and should not be interpreted as causal mechanisms.",
    "Alignment recommendations compare configured runs only; they do not establish clinical or in vivo safety.",
]


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
        alignment_summary = overlay_summary.get("alignment", {})
        branch_summary = overlay_summary.get("branch_summary", {})
        zone_summary = overlay_summary.get("zone_summary", {})
        total_branch_count = sum(
            int(branch.get("count", 0)) for branch in branch_summary.values()
        )
        productive_count = int(branch_summary.get("productive", {}).get("count", 0))

        row = {
            "label": payload.get("label", "unknown"),
            "alignment_mode": payload.get("alignment_mode", "unknown"),
            "geometry_distance_backend": payload.get(
                "geometry_distance_backend",
                alignment_summary.get("geometry_distance_backend"),
            ),
            "geometry_fallback_used": payload.get(
                "geometry_fallback_used",
                alignment_summary.get("geometry_fallback_used"),
            ),
            "fusion_graph_source": payload.get(
                "fusion_graph_source",
                alignment_summary.get("fusion_graph_source"),
            ),
            "alignment_graph_source": payload.get(
                "alignment_graph_source",
                alignment_summary.get("alignment_graph_source"),
            ),
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
    recommendation = build_alignment_recommendation(
        track_config=track_config,
        benchmark_rows=rows_sorted,
        timepoint_comparison=timepoint_comparison,
    )
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
        "recommendation": recommendation,
        "interpretation_limits": build_interpretation_limits(validation_manifest),
    }


def build_interpretation_limits(validation_manifest: Dict[str, Any]) -> list[str]:
    """Return interpretation limits that should travel with validation artifacts."""
    custom_limits = validation_manifest.get("track", {}).get("interpretation_limits", [])
    limits = list(DEFAULT_INTERPRETATION_LIMITS)
    limits.extend(str(limit) for limit in custom_limits if str(limit).strip())
    return limits


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


def build_alignment_recommendation(
    *,
    track_config: Dict[str, Any],
    benchmark_rows: Sequence[Dict[str, Any]],
    timepoint_comparison: Sequence[Dict[str, Any]],
    baseline_label: str = "euclidean",
    candidate_label: str = "projective",
) -> Dict[str, Any]:
    """Produce a track-level recommendation about alignment preference."""
    recommendation_config = track_config.get("recommendation", {})
    baseline_row = next((row for row in benchmark_rows if row.get("label") == baseline_label), None)
    candidate_row = next((row for row in benchmark_rows if row.get("label") == candidate_label), None)
    if baseline_row is None or candidate_row is None:
        return {
            "status": "unavailable",
            "preferred_alignment": None,
            "reason": "Required alignment runs were not both present.",
            "evidence": {"top_supporting_timepoints": [], "top_concerning_timepoints": []},
        }

    safe_gain = float(candidate_row["safe_fraction"]) - float(baseline_row["safe_fraction"])
    productive_gain = float(candidate_row["productive_fraction"]) - float(
        baseline_row["productive_fraction"]
    )
    risk_increase = float(candidate_row["risk_fraction"]) - float(baseline_row["risk_fraction"])
    l2_increase = float(candidate_row["mean_l2_shift"]) - float(baseline_row["mean_l2_shift"])
    timepoint_safe_gains = sum(
        1
        for row in timepoint_comparison
        if row.get("label") == candidate_label and float(row.get("delta_safe_fraction", 0.0)) > 0
    )

    meets_positive = (
        safe_gain >= float(recommendation_config.get("min_safe_fraction_gain", 0.0))
        and productive_gain
        >= float(recommendation_config.get("min_productive_fraction_gain", 0.0))
        and risk_increase
        <= float(recommendation_config.get("max_risk_fraction_increase", 0.0))
        and l2_increase
        <= float(recommendation_config.get("max_mean_l2_shift_increase", float("inf")))
        and timepoint_safe_gains
        >= int(recommendation_config.get("min_timepoint_safe_gains", 0))
    )

    if meets_positive:
        status = "prefer_projective"
        preferred_alignment = candidate_row.get("alignment_mode")
        reason = "Projective alignment improved safe/productive behavior within track thresholds."
    elif safe_gain > 0 and risk_increase <= float(
        recommendation_config.get("max_risk_fraction_increase", 0.0)
    ):
        status = "mixed"
        preferred_alignment = None
        reason = "Projective alignment shows some safety gains, but not enough consistent improvement yet."
    else:
        status = "prefer_euclidean"
        preferred_alignment = baseline_row.get("alignment_mode")
        reason = "Projective alignment does not currently clear the track-specific safety/productivity thresholds."

    candidate_timepoint_rows = [
        row for row in timepoint_comparison if row.get("label") == candidate_label
    ]
    evidence = build_recommendation_evidence(candidate_timepoint_rows)

    return {
        "status": status,
        "preferred_alignment": preferred_alignment,
        "reason": reason,
        "metrics": {
            "safe_fraction_gain": safe_gain,
            "productive_fraction_gain": productive_gain,
            "risk_fraction_increase": risk_increase,
            "mean_l2_shift_increase": l2_increase,
            "timepoint_safe_gains": timepoint_safe_gains,
        },
        "thresholds": recommendation_config,
        "evidence": evidence,
    }


def build_recommendation_evidence(
    timepoint_comparison_rows: Sequence[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, list[dict]]:
    """Return the strongest positive and negative timepoint evidence rows."""
    scored_rows = []
    for row in timepoint_comparison_rows:
        support_score = (
            float(row.get("delta_safe_fraction", 0.0))
            + float(row.get("delta_productive_fraction", 0.0))
            - float(row.get("delta_risk_fraction", 0.0))
        )
        concern_score = (
            float(row.get("delta_risk_fraction", 0.0))
            - float(row.get("delta_safe_fraction", 0.0))
            - float(row.get("delta_productive_fraction", 0.0))
        )
        scored_row = dict(row)
        scored_row["timepoint"] = str(row.get("timepoint", "unknown"))
        scored_row["support_score"] = support_score
        scored_row["concern_score"] = concern_score
        scored_rows.append(scored_row)

    top_supporting = sorted(
        scored_rows,
        key=lambda row: row["support_score"],
        reverse=True,
    )[:top_k]
    top_concerning = sorted(
        scored_rows,
        key=lambda row: row["concern_score"],
        reverse=True,
    )[:top_k]
    return {
        "top_supporting_timepoints": top_supporting,
        "top_concerning_timepoints": top_concerning,
    }
