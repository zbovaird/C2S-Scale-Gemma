"""Helpers for exporting interactive validation-explorer payloads."""

from __future__ import annotations

from typing import Any, Dict


def build_validation_explorer_payload(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured payload for interactive validation exploration."""
    runs = list(summary.get("runs", []))
    recommendation = dict(summary.get("recommendation", {}))
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
        "recommendation": recommendation,
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
                "id": "timepoint_comparison",
                "title": "Projective vs Euclidean deltas",
                "kind": "delta_series",
            },
            {
                "id": "recommendation_evidence",
                "title": "Recommendation evidence",
                "kind": "evidence_list",
            },
        ],
    }
