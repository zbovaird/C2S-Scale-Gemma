"""Helpers for applying dataset-specific reprogramming profiles."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tomllib
from typing import Any, Dict, Tuple


PROFILE_SECTIONS = ("references", "window_profile", "marker_panels")


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_profile_registry(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_dataset_profile(
    base_config: Dict[str, Any],
    profile_name: str | None,
    profile_config_path: str | Path | None,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    """Apply a named dataset profile onto the active config."""
    active_profile = profile_name or base_config.get("reprogramming", {}).get("dataset_profile")
    if not active_profile:
        return base_config, None
    if profile_config_path is None:
        raise ValueError("A profile config path is required when dataset_profile is set.")

    registry = load_profile_registry(profile_config_path)
    profile = registry.get("reprogramming_profiles", {}).get(active_profile)
    if profile is None:
        raise ValueError(f"Unknown dataset profile: {active_profile}")

    merged_config = deepcopy(base_config)
    reprogramming_config = deepcopy(merged_config.get("reprogramming", {}))
    for section in PROFILE_SECTIONS:
        existing = reprogramming_config.get(section, {})
        overrides = profile.get(section, {})
        reprogramming_config[section] = _deep_merge_dicts(existing, overrides)
    reprogramming_config["dataset_profile"] = active_profile
    reprogramming_config["dataset_manifest"] = {
        key: value for key, value in profile.items() if key not in PROFILE_SECTIONS
    }
    merged_config["reprogramming"] = reprogramming_config

    species = profile.get("species")
    if species:
        data_config = deepcopy(merged_config.get("data", {}))
        oskm_config = deepcopy(data_config.get("oskm", {}))
        oskm_config["species"] = species
        data_config["oskm"] = oskm_config
        merged_config["data"] = data_config

    return merged_config, reprogramming_config["dataset_manifest"]
