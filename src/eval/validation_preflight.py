"""Preflight and QA helpers for validation-bundle workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


REQUIRED_TRACK_FIELDS = (
    "dataset_profile",
    "primary_metrics",
)

REQUIRED_ARTIFACT_KEYS = (
    "summary_json",
    "summary_markdown",
    "explorer_payload",
    "explorer_html",
    "trajectory_dataset",
    "trajectory_geometry",
    "trajectory_projection",
    "trajectory_projection_html",
)

JSON_ARTIFACT_KEYS = (
    "summary_json",
    "explorer_payload",
    "trajectory_dataset",
    "trajectory_geometry",
    "trajectory_projection",
)


def _check_path_exists(check_id: str, path: str | Path, description: str) -> dict:
    file_path = Path(path)
    return {
        "id": check_id,
        "description": description,
        "path": str(file_path),
        "passed": file_path.exists(),
    }


def summarize_preflight_checks(checks: list[dict]) -> dict:
    """Summarize pass/fail state for a list of preflight checks."""
    failed_checks = [check for check in checks if not check.get("passed", False)]
    return {
        "status": "pass" if not failed_checks else "fail",
        "n_checks": len(checks),
        "n_failed": len(failed_checks),
        "checks": checks,
    }


def _build_track_field_checks(track_name: str, track_config: Mapping[str, Any]) -> list[dict]:
    checks = []
    for field in REQUIRED_TRACK_FIELDS:
        checks.append(
            {
                "id": f"track_field:{field}",
                "description": f"Validation track '{track_name}' defines {field}.",
                "passed": bool(track_config.get(field)),
                "value": track_config.get(field),
            }
        )

    is_condition_track = bool(track_config.get("condition_column"))
    axis_fields = (
        ("condition_column", "expected_conditions")
        if is_condition_track
        else ("timepoint_column", "expected_timepoints")
    )
    if not is_condition_track:
        checks.append(
            {
                "id": "track_field:cell_type_column",
                "description": (
                    f"Validation track '{track_name}' defines cell_type_column for "
                    "time-course profile checks."
                ),
                "passed": bool(track_config.get("cell_type_column")),
                "value": track_config.get("cell_type_column"),
            }
        )
    for field in axis_fields:
        checks.append(
            {
                "id": f"track_field:{field}",
                "description": f"Validation track '{track_name}' defines {field}.",
                "passed": bool(track_config.get(field)),
                "value": track_config.get(field),
            }
        )
    return checks


def build_validation_input_preflight(
    *,
    track_name: str,
    track_config: Mapping[str, Any],
    baseline_data_path: str | Path,
    perturbed_data_path: str | Path,
    euclidean_config: str | Path,
    euclidean_checkpoint: str | Path,
    projective_config: str | Path,
    projective_checkpoint: str | Path,
    dataset_profile_config: str | Path,
) -> dict:
    """Build a preflight report for inputs needed to run a validation bundle."""
    checks = _build_track_field_checks(track_name, track_config)
    checks.extend(
        [
            _check_path_exists(
                "baseline_data_path",
                baseline_data_path,
                "Baseline AnnData input exists.",
            ),
            _check_path_exists(
                "perturbed_data_path",
                perturbed_data_path,
                "Perturbed AnnData input exists.",
            ),
            _check_path_exists(
                "euclidean_config",
                euclidean_config,
                "Euclidean alignment config exists.",
            ),
            _check_path_exists(
                "euclidean_checkpoint",
                euclidean_checkpoint,
                "Euclidean alignment checkpoint exists.",
            ),
            _check_path_exists(
                "projective_config",
                projective_config,
                "Projective alignment config exists.",
            ),
            _check_path_exists(
                "projective_checkpoint",
                projective_checkpoint,
                "Projective alignment checkpoint exists.",
            ),
            _check_path_exists(
                "dataset_profile_config",
                dataset_profile_config,
                "Dataset profile registry exists.",
            ),
        ]
    )
    report = summarize_preflight_checks(checks)
    report["track_name"] = track_name
    report["dataset_profile"] = track_config.get("dataset_profile")
    return report


def build_validation_artifact_qa(artifact_manifest_path: str | Path) -> dict:
    """Validate that a one-command validation artifact export is complete."""
    manifest_path = Path(artifact_manifest_path)
    checks = [_check_path_exists("artifact_manifest", manifest_path, "Artifact manifest exists.")]
    if not manifest_path.exists():
        return summarize_preflight_checks(checks)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for key in REQUIRED_ARTIFACT_KEYS:
        artifact_path = manifest.get(key)
        checks.append(
            {
                "id": f"artifact:{key}",
                "description": f"Required artifact '{key}' exists.",
                "path": str(artifact_path) if artifact_path is not None else None,
                "passed": bool(artifact_path) and Path(artifact_path).exists(),
            }
        )
    for key in JSON_ARTIFACT_KEYS:
        artifact_path = manifest.get(key)
        passed = False
        if artifact_path and Path(artifact_path).exists():
            try:
                with Path(artifact_path).open("r", encoding="utf-8") as handle:
                    json.load(handle)
                passed = True
            except json.JSONDecodeError:
                passed = False
        checks.append(
            {
                "id": f"artifact_json:{key}",
                "description": f"Artifact '{key}' is valid JSON.",
                "path": str(artifact_path) if artifact_path is not None else None,
                "passed": passed,
            }
        )
    return summarize_preflight_checks(checks)
