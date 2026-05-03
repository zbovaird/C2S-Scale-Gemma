"""Compare dataset inspection reports against validation-track expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.validation_tracks import resolve_validation_track


def build_validation_profile_check(
    *,
    inspection: Mapping[str, Any],
    track_name: str,
    track_config: Mapping[str, Any],
) -> dict:
    """Build a go/no-go profile check from an inspection report and track config."""
    expected_timepoints = [str(value) for value in track_config.get("expected_timepoints", [])]
    observed_timepoints = {
        str(value)
        for value in (inspection.get("timepoint_summary") or {}).get("values", [])
    }
    missing_timepoints = [
        value for value in expected_timepoints if value not in observed_timepoints
    ]
    expected_conditions = [str(value) for value in track_config.get("expected_conditions", [])]
    observed_conditions = {
        str(value)
        for value in (inspection.get("condition_summary") or {}).get("values", [])
    }
    missing_conditions = [
        value for value in expected_conditions if value not in observed_conditions
    ]
    missing_columns = [
        column
        for column, present in (
            (track_config.get("cell_type_column"), inspection.get("cell_type_column_present")),
            (track_config.get("timepoint_column"), inspection.get("timepoint_column_present")),
            (track_config.get("condition_column"), inspection.get("condition_column_present")),
            (track_config.get("age_column"), inspection.get("age_column_present")),
            (track_config.get("batch_column"), inspection.get("batch_column_present")),
        )
        if column and not present
    ]
    missing_oskm = list(inspection.get("missing_oskm_genes", []))
    n_resolved_oskm = len(inspection.get("resolved_oskm_genes", {}))

    if missing_columns:
        status = "fail"
    elif missing_timepoints or missing_conditions or missing_oskm:
        status = "review"
    else:
        status = "pass"

    return {
        "artifact_type": "validation_profile_check",
        "track_name": track_name,
        "dataset_name": inspection.get("dataset_name"),
        "status": status,
        "missing_columns": missing_columns,
        "expected_timepoints": expected_timepoints,
        "observed_timepoints": sorted(observed_timepoints),
        "missing_timepoints": missing_timepoints,
        "expected_conditions": expected_conditions,
        "observed_conditions": sorted(observed_conditions),
        "missing_conditions": missing_conditions,
        "control_condition": track_config.get("control_condition"),
        "full_oskm_condition": track_config.get("full_oskm_condition"),
        "resolved_oskm_genes": dict(inspection.get("resolved_oskm_genes", {})),
        "missing_oskm_genes": missing_oskm,
        "n_resolved_oskm_genes": n_resolved_oskm,
        "n_cells": inspection.get("n_cells"),
        "n_genes": inspection.get("n_genes"),
    }


def load_validation_profile_check(
    *,
    inspection_path: str | Path,
    track_name: str,
    track_config_path: str | Path,
) -> dict:
    """Load inputs and compare a dataset inspection to a named validation track."""
    inspection = json.loads(Path(inspection_path).read_text(encoding="utf-8"))
    track_config = resolve_validation_track(track_name, track_config_path)
    return build_validation_profile_check(
        inspection=inspection,
        track_name=track_name,
        track_config=track_config,
    )


def write_validation_profile_check(
    profile_check: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Write a validation profile check report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile_check, indent=2), encoding="utf-8")
