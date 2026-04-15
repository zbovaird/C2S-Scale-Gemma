"""Validation-track helpers for named OKSM studies."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict


def load_validation_track_registry(config_path: str | Path) -> Dict[str, Any]:
    """Load the validation-track registry from TOML."""
    path = Path(config_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_validation_track(
    track_name: str,
    config_path: str | Path,
) -> Dict[str, Any]:
    """Resolve one named validation track."""
    registry = load_validation_track_registry(config_path)
    track = registry.get("validation_tracks", {}).get(track_name)
    if track is None:
        raise ValueError(f"Unknown validation track: {track_name}")
    return dict(track)


def build_validation_bundle_manifest(
    *,
    track_name: str,
    track_config: Dict[str, Any],
    output_root: str | Path,
    baseline_data_path: str,
    perturbed_data_path: str,
    ablation_manifest_path: str | Path,
) -> Dict[str, Any]:
    """Build a manifest that ties a validation track to generated outputs."""
    return {
        "track_name": track_name,
        "track": dict(track_config),
        "output_root": str(Path(output_root)),
        "baseline_data_path": baseline_data_path,
        "perturbed_data_path": perturbed_data_path,
        "dataset_profile": track_config.get("dataset_profile"),
        "ablation_manifest_path": str(Path(ablation_manifest_path)),
        "report_recommendation": {
            "primary_comparison_dir": str(Path(output_root) / "projective"),
            "ablation_manifest": str(Path(ablation_manifest_path)),
            "primary_metrics": list(track_config.get("primary_metrics", [])),
        },
    }
