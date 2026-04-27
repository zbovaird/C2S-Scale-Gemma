"""Calibration-audit helpers for validation profile thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.reprogramming_profiles import load_profile_registry
from eval.validation_tracks import load_validation_track_registry


WINDOW_RANGE_FIELDS = (
    "partial_window_proximity_min",
    "partial_window_proximity_max",
    "partial_window_max_risk",
    "longevity_safe_proximity_max",
    "longevity_safe_max_risk",
    "pluripotency_risk_proximity_min",
    "pluripotency_risk_score_min",
    "min_rejuvenation_score",
    "safe_max_pluripotency_marker",
    "risk_min_pluripotency_marker",
)

RECOMMENDATION_FIELDS = (
    "min_safe_fraction_gain",
    "min_productive_fraction_gain",
    "max_risk_fraction_increase",
    "max_mean_l2_shift_increase",
    "min_timepoint_safe_gains",
)


def _issue(severity: str, check_id: str, message: str) -> dict:
    return {"severity": severity, "id": check_id, "message": message}


def _in_unit_interval(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= numeric <= 1.0


def audit_window_profile(window_profile: Mapping[str, Any]) -> list[dict]:
    """Audit dataset-specific heuristic window thresholds."""
    issues = []
    for field in WINDOW_RANGE_FIELDS:
        if field not in window_profile:
            issues.append(_issue("error", f"missing_window:{field}", f"Missing {field}."))
        elif not _in_unit_interval(window_profile[field]):
            issues.append(
                _issue(
                    "error",
                    f"range_window:{field}",
                    f"{field} should be between 0 and 1.",
                )
            )

    min_proximity = window_profile.get("partial_window_proximity_min")
    max_proximity = window_profile.get("partial_window_proximity_max")
    if min_proximity is not None and max_proximity is not None:
        if float(min_proximity) > float(max_proximity):
            issues.append(
                _issue(
                    "error",
                    "partial_window_order",
                    "partial_window_proximity_min exceeds partial_window_proximity_max.",
                )
            )
    safe_marker = window_profile.get("safe_max_pluripotency_marker")
    risk_marker = window_profile.get("risk_min_pluripotency_marker")
    if safe_marker is not None and risk_marker is not None:
        if float(safe_marker) > float(risk_marker):
            issues.append(
                _issue(
                    "warning",
                    "marker_threshold_order",
                    "safe_max_pluripotency_marker exceeds risk_min_pluripotency_marker.",
                )
            )
    return issues


def audit_recommendation_thresholds(recommendation: Mapping[str, Any]) -> list[dict]:
    """Audit track-specific alignment recommendation thresholds."""
    issues = []
    for field in RECOMMENDATION_FIELDS:
        if field not in recommendation:
            issues.append(
                _issue("error", f"missing_recommendation:{field}", f"Missing {field}.")
            )
            continue
        value = recommendation[field]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            issues.append(
                _issue(
                    "error",
                    f"numeric_recommendation:{field}",
                    f"{field} should be numeric.",
                )
            )
            continue
        if numeric < 0:
            issues.append(
                _issue(
                    "error",
                    f"negative_recommendation:{field}",
                    f"{field} should be non-negative.",
                )
            )
    return issues


def build_validation_calibration_row(
    *,
    track_name: str,
    track_config: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
) -> dict:
    """Build one calibration-audit row for a validation track."""
    profile_name = str(track_config.get("dataset_profile", ""))
    profile = profile_registry.get("reprogramming_profiles", {}).get(profile_name)
    issues = []
    if profile is None:
        issues.append(
            _issue("error", "missing_profile", f"Unknown dataset profile: {profile_name}")
        )
    else:
        issues.extend(audit_window_profile(profile.get("window_profile", {})))
    issues.extend(audit_recommendation_thresholds(track_config.get("recommendation", {})))

    n_errors = sum(1 for issue in issues if issue["severity"] == "error")
    n_warnings = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "pass" if not issues else "review"
    if n_errors:
        status = "fail"

    return {
        "track_name": track_name,
        "dataset_profile": profile_name or None,
        "status": status,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
        "window_profile": dict((profile or {}).get("window_profile", {})),
        "recommendation": dict(track_config.get("recommendation", {})),
    }


def build_validation_calibration_report(
    *,
    track_registry: Mapping[str, Any],
    profile_registry: Mapping[str, Any],
) -> dict:
    """Build a calibration-audit report for all validation tracks."""
    rows = [
        build_validation_calibration_row(
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
    return {"status_counts": status_counts, "tracks": rows}


def load_validation_calibration_report(
    *,
    track_config_path: str | Path,
    profile_config_path: str | Path,
) -> dict:
    """Load validation configs and build their calibration-audit report."""
    return build_validation_calibration_report(
        track_registry=load_validation_track_registry(track_config_path),
        profile_registry=load_profile_registry(profile_config_path),
    )


def write_validation_calibration_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a calibration-audit report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
