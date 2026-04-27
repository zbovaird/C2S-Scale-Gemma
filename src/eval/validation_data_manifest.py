"""Build data acquisition manifests for validation tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.reprogramming_profiles import load_profile_registry
from eval.validation_readiness import build_validation_readiness_report
from eval.validation_tracks import load_validation_track_registry


def build_validation_data_manifest(
    *,
    track_registry: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
) -> dict:
    """Build an actionable data manifest from validation tracks and profiles."""
    readiness = build_validation_readiness_report(
        track_registry=track_registry,
        profile_registry=profile_registry,
    )
    readiness_by_track = {row["track_name"]: row for row in readiness.get("tracks", [])}
    rows = []
    for track_name, track_config in track_registry.get("validation_tracks", {}).items():
        profile_name = track_config.get("dataset_profile")
        profile = profile_registry.get("reprogramming_profiles", {}).get(profile_name, {})
        readiness_row = readiness_by_track.get(track_name, {})
        rows.append(
            {
                "track_name": track_name,
                "title": track_config.get("title") or profile.get("title"),
                "dataset_profile": profile_name,
                "readiness_status": readiness_row.get("status"),
                "accession": track_config.get("accession") or profile.get("accession"),
                "species": track_config.get("species") or profile.get("species"),
                "modality": profile.get("modality"),
                "source_url": track_config.get("source_url") or profile.get("source_url"),
                "baseline_data_hint": track_config.get("baseline_data_hint")
                or profile.get("baseline_data_hint"),
                "perturbed_data_hint": track_config.get("perturbed_data_hint"),
                "required_obs_columns": [
                    value
                    for value in (
                        track_config.get("cell_type_column") or profile.get("cell_type_column"),
                        track_config.get("timepoint_column") or profile.get("timepoint_column"),
                    )
                    if value
                ],
                "expected_timepoints": list(track_config.get("expected_timepoints", [])),
                "notes": profile.get("notes"),
            }
        )
    return {
        "artifact_type": "validation_data_manifest",
        "status_counts": readiness.get("status_counts", {}),
        "datasets": rows,
    }


def load_validation_data_manifest(
    *,
    track_config_path: str | Path,
    profile_config_path: str | Path,
) -> dict:
    """Load validation configs and build their data manifest."""
    return build_validation_data_manifest(
        track_registry=load_validation_track_registry(track_config_path),
        profile_registry=load_profile_registry(profile_config_path),
    )


def write_validation_data_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a validation data manifest as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
