"""Helpers for plotting validation trajectory projection artifacts."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def build_projection_phase_series(
    run_projection: Dict[str, Any],
    *,
    phase: str = "perturbed",
    category_key: str = "branch_label",
) -> Dict[str, list[dict]]:
    """Group projection rows for scatter plotting."""
    if phase not in {"baseline", "perturbed"}:
        raise ValueError("phase must be 'baseline' or 'perturbed'.")

    x_key = f"{phase}_x"
    y_key = f"{phase}_y"
    series: dict[str, list[dict]] = {}
    for row in run_projection.get("rows", []):
        category = str(row.get(category_key, "unknown"))
        series.setdefault(category, []).append(
            {
                "cell_id": row.get("cell_id"),
                "timepoint": row.get("timepoint", "unknown"),
                "x": float(row.get(x_key, 0.0)),
                "y": float(row.get(y_key, 0.0)),
            }
        )
    return series


def build_projection_arrow_rows(run_projection: Dict[str, Any]) -> list[dict]:
    """Build line-segment rows that connect baseline to perturbed positions."""
    return [
        {
            "cell_id": row.get("cell_id"),
            "timepoint": row.get("timepoint", "unknown"),
            "x0": float(row.get("baseline_x", 0.0)),
            "y0": float(row.get("baseline_y", 0.0)),
            "x1": float(row.get("perturbed_x", 0.0)),
            "y1": float(row.get("perturbed_y", 0.0)),
            "branch_label": row.get("branch_label"),
            "longevity_safe_zone": bool(row.get("longevity_safe_zone")),
            "pluripotency_risk_flag": bool(row.get("pluripotency_risk_flag")),
        }
        for row in run_projection.get("rows", [])
    ]


def build_projection_timepoint_series(
    run_projection: Dict[str, Any],
    *,
    phase: str = "perturbed",
) -> Dict[str, list[dict]]:
    """Group projection rows by timepoint for stage-wise scatter plots."""
    return build_projection_phase_series(
        run_projection,
        phase=phase,
        category_key="timepoint",
    )
