"""Helpers for preparing validation-bundle trajectory plots."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def build_timepoint_metric_series(
    summary: Dict[str, Any],
    metric_key: str,
) -> Dict[str, list[dict]]:
    """Build ordered per-run timepoint series for one metric."""
    series = {}
    for label, rows in summary.get("timepoint_summaries", {}).items():
        series[label] = [
            {
                "timepoint": row["timepoint"],
                "value": float(row.get(metric_key, 0.0)),
            }
            for row in rows
        ]
    return series


def build_timepoint_delta_series(
    summary: Dict[str, Any],
    metric_key: str,
) -> list[dict]:
    """Build ordered timepoint deltas for one comparison metric."""
    return [
        {
            "label": str(row.get("label", "unknown")),
            "timepoint": str(row.get("timepoint", "unknown")),
            "value": float(row.get(metric_key, 0.0)),
        }
        for row in summary.get("timepoint_comparison", [])
    ]
