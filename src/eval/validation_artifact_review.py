"""Higher-level review helpers for exported validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from eval.validation_preflight import build_validation_artifact_qa
from eval.validation_summary import load_json_file


def _review_item(
    check_id: str,
    status: str,
    message: str,
    severity: str = "info",
    details: Mapping[str, Any] | None = None,
) -> dict:
    return {
        "id": check_id,
        "status": status,
        "severity": severity,
        "message": message,
        "details": dict(details or {}),
    }


def build_validation_artifact_review(artifact_manifest_path: str | Path) -> dict:
    """Build a content-level review report for exported validation artifacts."""
    manifest_path = Path(artifact_manifest_path)
    qa_report = build_validation_artifact_qa(manifest_path)
    review_items = [
        _review_item(
            "artifact_qa",
            "pass" if qa_report["status"] == "pass" else "fail",
            "Required artifacts exist and parse." if qa_report["status"] == "pass" else "Artifact QA failed.",
            "error" if qa_report["status"] != "pass" else "info",
            {"n_failed": qa_report["n_failed"]},
        )
    ]
    if qa_report["status"] != "pass":
        return _summarize_review(manifest_path, qa_report, review_items)

    manifest = load_json_file(manifest_path)
    summary = load_json_file(manifest["summary_json"])
    runs = list(summary.get("runs", []))
    geometry_summary = list(summary.get("trajectory_geometry_summary", []))
    interpretation_limits = list(summary.get("interpretation_limits", []))
    recommendation = dict(summary.get("recommendation", {}))

    review_items.append(
        _review_item(
            "interpretation_limits",
            "pass" if interpretation_limits else "fail",
            "Interpretation limits are present." if interpretation_limits else "Interpretation limits are missing.",
            "error" if not interpretation_limits else "info",
            {"n_limits": len(interpretation_limits)},
        )
    )
    review_items.append(
        _review_item(
            "recommendation_status",
            "review" if recommendation.get("status") in {"mixed", "unavailable"} else "pass",
            f"Recommendation status: {recommendation.get('status')}",
            "warning" if recommendation.get("status") in {"mixed", "unavailable"} else "info",
            {"preferred_alignment": recommendation.get("preferred_alignment")},
        )
    )

    fallback_runs = [
        row.get("label")
        for row in runs
        if bool(row.get("geometry_fallback_used", False))
    ]
    review_items.append(
        _review_item(
            "geometry_fallbacks",
            "review" if fallback_runs else "pass",
            "Some runs used geometry fallback distance." if fallback_runs else "No run-level geometry fallbacks reported.",
            "warning" if fallback_runs else "info",
            {"fallback_runs": fallback_runs},
        )
    )

    max_risk_fraction = max(
        [float(row.get("risk_fraction", 0.0)) for row in runs],
        default=0.0,
    )
    max_mean_geometry = max(
        [float(row.get("mean_geometry_distance", 0.0)) for row in geometry_summary],
        default=0.0,
    )
    review_items.append(
        _review_item(
            "risk_summary",
            "review" if max_risk_fraction > 0.0 else "pass",
            "Risk cohorts are present." if max_risk_fraction > 0.0 else "No risk cohorts reported.",
            "warning" if max_risk_fraction > 0.0 else "info",
            {"max_risk_fraction": max_risk_fraction},
        )
    )
    review_items.append(
        _review_item(
            "trajectory_geometry",
            "pass" if geometry_summary else "review",
            "Trajectory geometry summary is present." if geometry_summary else "Trajectory geometry summary is missing.",
            "warning" if not geometry_summary else "info",
            {"max_mean_geometry_distance": max_mean_geometry},
        )
    )
    return _summarize_review(manifest_path, qa_report, review_items)


def _summarize_review(
    manifest_path: Path,
    qa_report: Mapping[str, Any],
    review_items: list[dict],
) -> dict:
    n_fail = sum(1 for item in review_items if item["status"] == "fail")
    n_review = sum(1 for item in review_items if item["status"] == "review")
    status = "fail" if n_fail else "review" if n_review else "pass"
    return {
        "artifact_type": "validation_artifact_review",
        "artifact_manifest_path": str(manifest_path),
        "status": status,
        "n_fail": n_fail,
        "n_review": n_review,
        "qa_status": qa_report["status"],
        "review_items": review_items,
    }


def write_validation_artifact_review(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a validation artifact review report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
