import json

from eval.validation_calibration import (
    audit_recommendation_thresholds,
    audit_window_profile,
    build_validation_calibration_report,
    write_validation_calibration_report,
)


VALID_WINDOW_PROFILE = {
    "partial_window_proximity_min": 0.2,
    "partial_window_proximity_max": 0.7,
    "partial_window_max_risk": 0.5,
    "longevity_safe_proximity_max": 0.6,
    "longevity_safe_max_risk": 0.4,
    "pluripotency_risk_proximity_min": 0.8,
    "pluripotency_risk_score_min": 0.6,
    "min_rejuvenation_score": 0.3,
    "safe_max_pluripotency_marker": 0.3,
    "risk_min_pluripotency_marker": 0.6,
}

VALID_RECOMMENDATION = {
    "min_safe_fraction_gain": 0.05,
    "min_productive_fraction_gain": 0.05,
    "max_risk_fraction_increase": 0.03,
    "max_mean_l2_shift_increase": 0.10,
    "min_timepoint_safe_gains": 1,
}


def test_audit_window_profile_passes_valid_profile():
    assert audit_window_profile(VALID_WINDOW_PROFILE) == []


def test_audit_window_profile_flags_invalid_order_and_ranges():
    issues = audit_window_profile(
        {
            **VALID_WINDOW_PROFILE,
            "partial_window_proximity_min": 0.8,
            "partial_window_proximity_max": 0.2,
            "partial_window_max_risk": 1.2,
        }
    )
    issue_ids = {issue["id"] for issue in issues}

    assert "partial_window_order" in issue_ids
    assert "range_window:partial_window_max_risk" in issue_ids


def test_audit_recommendation_thresholds_flags_missing_and_negative_values():
    issues = audit_recommendation_thresholds(
        {
            "min_safe_fraction_gain": -0.1,
        }
    )
    issue_ids = {issue["id"] for issue in issues}

    assert "negative_recommendation:min_safe_fraction_gain" in issue_ids
    assert "missing_recommendation:max_risk_fraction_increase" in issue_ids


def test_build_validation_calibration_report_counts_pass_status():
    report = build_validation_calibration_report(
        track_registry={
            "validation_tracks": {
                "human_fibroblast_oskm": {
                    "dataset_profile": "gse242423_human_fibroblast_oskm",
                    "recommendation": VALID_RECOMMENDATION,
                }
            }
        },
        profile_registry={
            "reprogramming_profiles": {
                "gse242423_human_fibroblast_oskm": {
                    "window_profile": VALID_WINDOW_PROFILE,
                }
            }
        },
    )

    assert report["status_counts"]["pass"] == 1
    assert report["tracks"][0]["n_errors"] == 0


def test_write_validation_calibration_report_writes_json(tmp_path):
    output_path = tmp_path / "calibration.json"
    write_validation_calibration_report({"status_counts": {"pass": 1}}, output_path)

    assert json.loads(output_path.read_text())["status_counts"]["pass"] == 1
