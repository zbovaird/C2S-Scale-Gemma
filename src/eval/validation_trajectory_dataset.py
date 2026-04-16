"""Helpers for exporting cell-level validation trajectory artifacts."""

from __future__ import annotations

from typing import Any, Dict, Sequence


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_validation_run_cell_rows(
    *,
    track_config: Dict[str, Any],
    payload: Dict[str, Any],
) -> list[dict]:
    """Build cell-level rows for one validation run."""
    timepoint_column = track_config.get("timepoint_column")
    rows = []
    for row in payload.get("fused_shift_rows", []):
        normalized = {key: _normalize_scalar(value) for key, value in row.items()}
        normalized["label"] = str(payload.get("label", "unknown"))
        normalized["alignment_mode"] = str(payload.get("alignment_mode", "unknown"))
        normalized["dataset_profile"] = payload.get("dataset_profile")
        normalized["timepoint"] = (
            str(row.get(timepoint_column, "unknown")) if timepoint_column else "unknown"
        )
        rows.append(normalized)
    return rows


def build_timepoint_branch_cohort_rows(
    *,
    track_config: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> list[dict]:
    """Aggregate cell-level rows into timepoint/branch cohorts for plotting."""
    cohort_rows = []
    for payload in run_payloads:
        run_rows = build_validation_run_cell_rows(track_config=track_config, payload=payload)
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in run_rows:
            key = (str(row.get("timepoint", "unknown")), str(row.get("branch_label", "unknown")))
            grouped.setdefault(key, []).append(row)
        for (timepoint, branch_label), rows in grouped.items():
            n_cells = len(rows)
            if not n_cells:
                continue
            cohort_rows.append(
                {
                    "label": payload.get("label", "unknown"),
                    "alignment_mode": payload.get("alignment_mode", "unknown"),
                    "timepoint": timepoint,
                    "branch_label": branch_label,
                    "n_cells": n_cells,
                    "mean_l2_shift": sum(float(row.get("l2_shift", 0.0)) for row in rows)
                    / n_cells,
                    "mean_progress_delta": sum(
                        float(row.get("progress_delta", 0.0)) for row in rows
                    )
                    / n_cells,
                    "mean_risk_score": sum(float(row.get("risk_score", 0.0)) for row in rows)
                    / n_cells,
                    "partial_window_fraction": sum(
                        bool(row.get("partial_reprogramming_window")) for row in rows
                    )
                    / n_cells,
                    "safe_fraction": sum(
                        bool(row.get("longevity_safe_zone")) for row in rows
                    )
                    / n_cells,
                    "risk_fraction": sum(
                        bool(row.get("pluripotency_risk_flag")) for row in rows
                    )
                    / n_cells,
                }
            )
    return sorted(
        cohort_rows,
        key=lambda row: (str(row["label"]), str(row["timepoint"]), str(row["branch_label"])),
    )


def build_cell_alignment_delta_rows(
    *,
    track_config: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
    baseline_label: str = "euclidean",
    candidate_label: str = "projective",
) -> list[dict]:
    """Join paired runs by cell so alignment deltas can be visualized."""
    rows_by_label = {
        str(payload.get("label", "unknown")): build_validation_run_cell_rows(
            track_config=track_config,
            payload=payload,
        )
        for payload in run_payloads
    }
    baseline_rows = rows_by_label.get(baseline_label, [])
    candidate_rows = rows_by_label.get(candidate_label, [])

    def _row_key(row: Dict[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("cell_id", row.get("cell_index", "unknown"))),
            str(row.get("timepoint", "unknown")),
        )

    baseline_index = {_row_key(row): row for row in baseline_rows}
    delta_rows = []
    for row in candidate_rows:
        baseline_row = baseline_index.get(_row_key(row))
        if baseline_row is None:
            continue
        delta_rows.append(
            {
                "cell_id": row.get("cell_id", row.get("cell_index")),
                "cell_type": row.get("cell_type"),
                "timepoint": row.get("timepoint", "unknown"),
                "candidate_label": candidate_label,
                "baseline_label": baseline_label,
                "candidate_branch_label": row.get("branch_label"),
                "baseline_branch_label": baseline_row.get("branch_label"),
                "candidate_safe_zone": bool(row.get("longevity_safe_zone")),
                "baseline_safe_zone": bool(baseline_row.get("longevity_safe_zone")),
                "candidate_partial_window": bool(row.get("partial_reprogramming_window")),
                "baseline_partial_window": bool(
                    baseline_row.get("partial_reprogramming_window")
                ),
                "candidate_risk_flag": bool(row.get("pluripotency_risk_flag")),
                "baseline_risk_flag": bool(baseline_row.get("pluripotency_risk_flag")),
                "delta_l2_shift": float(row.get("l2_shift", 0.0))
                - float(baseline_row.get("l2_shift", 0.0)),
                "delta_progress_delta": float(row.get("progress_delta", 0.0))
                - float(baseline_row.get("progress_delta", 0.0)),
                "delta_risk_score": float(row.get("risk_score", 0.0))
                - float(baseline_row.get("risk_score", 0.0)),
                "delta_pluripotency_proximity": float(
                    row.get("pluripotency_proximity", 0.0)
                )
                - float(baseline_row.get("pluripotency_proximity", 0.0)),
                "delta_safe_zone": int(bool(row.get("longevity_safe_zone")))
                - int(bool(baseline_row.get("longevity_safe_zone"))),
                "delta_partial_window": int(bool(row.get("partial_reprogramming_window")))
                - int(bool(baseline_row.get("partial_reprogramming_window"))),
                "delta_risk_flag": int(bool(row.get("pluripotency_risk_flag")))
                - int(bool(baseline_row.get("pluripotency_risk_flag"))),
            }
        )
    return delta_rows


def build_validation_trajectory_dataset(
    validation_manifest: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a reusable validation trajectory dataset artifact."""
    track_config = dict(validation_manifest.get("track", {}))
    run_rows = {
        str(payload.get("label", "unknown")): build_validation_run_cell_rows(
            track_config=track_config,
            payload=payload,
        )
        for payload in run_payloads
    }
    return {
        "track_name": validation_manifest.get("track_name"),
        "dataset_profile": validation_manifest.get("dataset_profile"),
        "timepoint_column": track_config.get("timepoint_column"),
        "run_cell_rows": run_rows,
        "timepoint_branch_cohorts": build_timepoint_branch_cohort_rows(
            track_config=track_config,
            run_payloads=run_payloads,
        ),
        "cell_alignment_deltas": build_cell_alignment_delta_rows(
            track_config=track_config,
            run_payloads=run_payloads,
        ),
    }
