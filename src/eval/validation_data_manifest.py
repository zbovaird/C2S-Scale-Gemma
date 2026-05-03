"""Build data acquisition manifests for validation tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.reprogramming_profiles import load_profile_registry
from eval.validation_readiness import build_validation_readiness_report
from eval.validation_tracks import load_validation_track_registry


def infer_geo_supplement_url(accession: str | None, file_name: str | None) -> str | None:
    """Infer a GEO supplementary HTTPS URL for a processed file."""
    if not accession or not file_name or not accession.startswith("GSE"):
        return None
    digits = accession[3:]
    if len(digits) < 4 or not digits.isdigit():
        return None
    series_bucket = f"GSE{digits[:-3]}nnn"
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{series_bucket}/{accession}/suppl/{file_name}"
    )


def build_download_plan(accession: str | None, local_path: str | None) -> dict | None:
    """Build a non-executing download plan for a configured local data target."""
    if not local_path:
        return None
    file_name = Path(local_path).name
    source_url = infer_geo_supplement_url(accession, file_name)
    if source_url is None:
        return {
            "local_path": local_path,
            "file_name": file_name,
            "source_url": None,
            "command": None,
            "notes": "No automatic GEO supplementary URL could be inferred.",
        }
    return {
        "local_path": local_path,
        "file_name": file_name,
        "source_url": source_url,
        "command": f"mkdir -p {Path(local_path).parent} && curl -L {source_url} -o {local_path}",
        "notes": "Review file size and available disk space before downloading.",
    }


def build_candidate_dataset_rows(track_registry: Mapping[str, Any]) -> list[dict]:
    """Build review-only rows for candidate validation datasets."""
    rows = []
    for candidate_name, candidate in track_registry.get(
        "validation_dataset_candidates", {}
    ).items():
        rows.append(
            {
                "candidate_name": candidate_name,
                "title": candidate.get("title"),
                "study": candidate.get("study"),
                "accessions": list(candidate.get("accessions", [])),
                "portal_accessions": list(candidate.get("portal_accessions", [])),
                "publication": candidate.get("publication"),
                "species": candidate.get("species"),
                "trajectory": candidate.get("trajectory"),
                "induction": candidate.get("induction"),
                "timepoints": list(candidate.get("timepoints", [])),
                "modalities": list(candidate.get("modalities", [])),
                "format_hints": list(candidate.get("format_hints", [])),
                "source_urls": list(candidate.get("source_urls", [])),
                "local_data_hint": candidate.get("local_data_hint"),
                "validation_role": candidate.get("validation_role"),
                "strengths": candidate.get("strengths"),
                "limitations": candidate.get("limitations"),
                "notes": candidate.get("notes"),
                "acquisition_status": "candidate_review_only",
            }
        )
    return rows


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
        accession = track_config.get("accession") or profile.get("accession")
        baseline_data_hint = track_config.get("baseline_data_hint") or profile.get(
            "baseline_data_hint"
        )
        perturbed_data_hint = track_config.get("perturbed_data_hint")
        rows.append(
            {
                "track_name": track_name,
                "title": track_config.get("title") or profile.get("title"),
                "dataset_profile": profile_name,
                "readiness_status": readiness_row.get("status"),
                "accession": accession,
                "species": track_config.get("species") or profile.get("species"),
                "modality": profile.get("modality"),
                "source_url": track_config.get("source_url") or profile.get("source_url"),
                "baseline_data_hint": baseline_data_hint,
                "perturbed_data_hint": perturbed_data_hint,
                "download_plan": build_download_plan(accession, baseline_data_hint),
                "required_obs_columns": [
                    value
                    for value in (
                        track_config.get("cell_type_column") or profile.get("cell_type_column"),
                        track_config.get("timepoint_column") or profile.get("timepoint_column"),
                        track_config.get("condition_column") or profile.get("condition_column"),
                        track_config.get("age_column") or profile.get("age_column"),
                        track_config.get("batch_column") or profile.get("batch_column"),
                    )
                    if value
                ],
                "expected_timepoints": list(track_config.get("expected_timepoints", [])),
                "expected_conditions": list(track_config.get("expected_conditions", [])),
                "notes": profile.get("notes"),
            }
        )
    return {
        "artifact_type": "validation_data_manifest",
        "status_counts": readiness.get("status_counts", {}),
        "datasets": rows,
        "candidate_datasets": build_candidate_dataset_rows(track_registry),
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
