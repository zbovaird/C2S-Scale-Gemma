"""Dataset readiness reports for named validation tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.reprogramming_profiles import load_profile_registry
from eval.validation_tracks import load_validation_track_registry


REQUIRED_PROFILE_FIELDS = (
    "accession",
    "species",
    "source_url",
    "baseline_data_hint",
    "cell_type_column",
    "timepoint_column",
)

REQUIRED_TRACK_FIELDS = (
    "dataset_profile",
    "baseline_data_hint",
    "perturbed_data_hint",
    "cell_type_column",
    "timepoint_column",
    "expected_timepoints",
    "primary_metrics",
)


def _missing_fields(payload: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if not payload.get(field)]


def build_validation_dataset_readiness(
    *,
    track_name: str,
    track_config: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
) -> dict:
    """Build one readiness row for a named validation track."""
    profile_name = str(track_config.get("dataset_profile", ""))
    profile = profile_registry.get("reprogramming_profiles", {}).get(profile_name)
    track_missing = _missing_fields(track_config, REQUIRED_TRACK_FIELDS)
    profile_missing = (
        _missing_fields(profile, REQUIRED_PROFILE_FIELDS) if profile is not None else list(REQUIRED_PROFILE_FIELDS)
    )
    baseline_hint = track_config.get("baseline_data_hint") or (
        profile or {}
    ).get("baseline_data_hint")
    perturbed_hint = track_config.get("perturbed_data_hint")
    baseline_exists = bool(baseline_hint) and Path(str(baseline_hint)).exists()
    perturbed_exists = bool(perturbed_hint) and Path(str(perturbed_hint)).exists()
    metadata_ready = profile is not None and not track_missing and not profile_missing

    if not metadata_ready:
        status = "incomplete_metadata"
    elif baseline_exists and perturbed_exists:
        status = "ready"
    else:
        status = "needs_data"

    return {
        "track_name": track_name,
        "title": track_config.get("title"),
        "dataset_profile": profile_name or None,
        "profile_present": profile is not None,
        "status": status,
        "track_missing_fields": track_missing,
        "profile_missing_fields": profile_missing,
        "accession": track_config.get("accession") or (profile or {}).get("accession"),
        "species": track_config.get("species") or (profile or {}).get("species"),
        "source_url": track_config.get("source_url") or (profile or {}).get("source_url"),
        "baseline_data_hint": baseline_hint,
        "baseline_data_exists": baseline_exists,
        "perturbed_data_hint": perturbed_hint,
        "perturbed_data_exists": perturbed_exists,
        "cell_type_column": track_config.get("cell_type_column") or (profile or {}).get("cell_type_column"),
        "timepoint_column": track_config.get("timepoint_column") or (profile or {}).get("timepoint_column"),
        "expected_timepoints": list(track_config.get("expected_timepoints", [])),
        "primary_metrics": list(track_config.get("primary_metrics", [])),
        "report_focus": track_config.get("report_focus"),
    }


def build_validation_readiness_report(
    *,
    track_registry: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
) -> dict:
    """Build a readiness report for all configured validation tracks."""
    rows = [
        build_validation_dataset_readiness(
            track_name=track_name,
            track_config=track_config,
            profile_registry=profile_registry,
        )
        for track_name, track_config in track_registry.get("validation_tracks", {}).items()
    ]
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status_counts": status_counts,
        "tracks": rows,
    }


def load_validation_readiness_report(
    *,
    track_config_path: str | Path,
    profile_config_path: str | Path,
) -> dict:
    """Load validation configs and build their readiness report."""
    return build_validation_readiness_report(
        track_registry=load_validation_track_registry(track_config_path),
        profile_registry=load_profile_registry(profile_config_path),
    )


def write_validation_readiness_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a validation readiness report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
