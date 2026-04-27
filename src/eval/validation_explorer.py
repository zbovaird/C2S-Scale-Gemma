"""Helpers for exporting interactive validation-explorer payloads."""

from __future__ import annotations

from typing import Any, Dict

from eval.validation_visuals import (
    build_timepoint_delta_series,
    build_timepoint_metric_series,
)


def build_validation_explorer_charts(summary: Dict[str, Any]) -> list[dict]:
    """Build chart-ready specs for the validation explorer."""
    return [
        {
            "id": "progression_by_timepoint",
            "title": "Mean progress delta across timepoints",
            "kind": "line",
            "metric_key": "mean_progress_delta",
            "series": build_timepoint_metric_series(summary, "mean_progress_delta"),
        },
        {
            "id": "safe_fraction_by_timepoint",
            "title": "Safe fraction across timepoints",
            "kind": "line",
            "metric_key": "safe_fraction",
            "series": build_timepoint_metric_series(summary, "safe_fraction"),
        },
        {
            "id": "risk_fraction_by_timepoint",
            "title": "Risk fraction across timepoints",
            "kind": "line",
            "metric_key": "risk_fraction",
            "series": build_timepoint_metric_series(summary, "risk_fraction"),
        },
        {
            "id": "safe_fraction_delta",
            "title": "Safe fraction delta vs Euclidean",
            "kind": "bar",
            "metric_key": "delta_safe_fraction",
            "series": build_timepoint_delta_series(summary, "delta_safe_fraction"),
        },
        {
            "id": "productive_fraction_delta",
            "title": "Productive fraction delta vs Euclidean",
            "kind": "bar",
            "metric_key": "delta_productive_fraction",
            "series": build_timepoint_delta_series(summary, "delta_productive_fraction"),
        },
    ]


def build_validation_explorer_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured payload for interactive validation exploration."""
    runs = list(summary.get("runs", []))
    recommendation = dict(summary.get("recommendation", {}))
    charts = build_validation_explorer_charts(summary)
    return {
        "track_name": summary.get("track_name"),
        "dataset_profile": summary.get("dataset_profile"),
        "overview_cards": [
            {
                "id": "best_by_mean_l2_shift",
                "label": "Best by mean L2 shift",
                "value": (summary.get("best_by_mean_l2_shift") or {}).get("label"),
            },
            {
                "id": "recommendation_status",
                "label": "Recommendation status",
                "value": recommendation.get("status"),
            },
            {
                "id": "preferred_alignment",
                "label": "Preferred alignment",
                "value": recommendation.get("preferred_alignment"),
            },
        ],
        "run_table": runs,
        "timepoint_summaries": summary.get("timepoint_summaries", {}),
        "timepoint_comparison": summary.get("timepoint_comparison", []),
        "charts": charts,
        "recommendation": recommendation,
        "interpretation_limits": summary.get("interpretation_limits", []),
        "explorer_sections": [
            {
                "id": "run_table",
                "title": "Run overview",
                "kind": "table",
            },
            {
                "id": "timepoint_progression",
                "title": "Timepoint progression",
                "kind": "multi_series",
            },
            {
                "id": "charts",
                "title": "Trajectory charts",
                "kind": "chart_group",
            },
            {
                "id": "timepoint_comparison",
                "title": "Projective vs Euclidean deltas",
                "kind": "delta_series",
            },
            {
                "id": "recommendation_evidence",
                "title": "Recommendation evidence",
                "kind": "evidence_list",
            },
            {
                "id": "interpretation_limits",
                "title": "Interpretation limits",
                "kind": "note_list",
            },
        ],
    }
