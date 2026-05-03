import json

from eval.validation_profile_check import (
    build_validation_profile_check,
    load_validation_profile_check,
    write_validation_profile_check,
)


def _inspection():
    return {
        "dataset_name": "toy",
        "n_cells": 10,
        "n_genes": 4,
        "cell_type_column_present": True,
        "timepoint_column_present": True,
        "timepoint_summary": {"values": ["D0", "D2"]},
        "resolved_oskm_genes": {
            "POU5F1": "POU5F1",
            "SOX2": "SOX2",
            "KLF4": "KLF4",
            "MYC": "MYC",
        },
        "missing_oskm_genes": [],
    }


def _track_config():
    return {
        "cell_type_column": "cell_type",
        "timepoint_column": "timepoint",
        "expected_timepoints": ["D0", "D2"],
    }


def test_build_validation_profile_check_passes_matching_profile():
    report = build_validation_profile_check(
        inspection=_inspection(),
        track_name="toy_track",
        track_config=_track_config(),
    )

    assert report["artifact_type"] == "validation_profile_check"
    assert report["status"] == "pass"
    assert report["missing_columns"] == []
    assert report["missing_timepoints"] == []
    assert report["n_resolved_oskm_genes"] == 4


def test_build_validation_profile_check_flags_missing_columns_as_fail():
    inspection = _inspection()
    inspection["timepoint_column_present"] = False

    report = build_validation_profile_check(
        inspection=inspection,
        track_name="toy_track",
        track_config=_track_config(),
    )

    assert report["status"] == "fail"
    assert report["missing_columns"] == ["timepoint"]


def test_build_validation_profile_check_flags_missing_timepoints_for_review():
    report = build_validation_profile_check(
        inspection=_inspection(),
        track_name="toy_track",
        track_config={
            **_track_config(),
            "expected_timepoints": ["D0", "D2", "iPSC"],
        },
    )

    assert report["status"] == "review"
    assert report["missing_timepoints"] == ["iPSC"]


def test_build_validation_profile_check_passes_condition_screen_without_timepoints():
    inspection = {
        **_inspection(),
        "cell_type_column_present": False,
        "timepoint_column_present": False,
        "condition_column_present": True,
        "age_column_present": True,
        "batch_column_present": True,
        "condition_summary": {"values": ["NT", "SOKM"]},
    }

    report = build_validation_profile_check(
        inspection=inspection,
        track_name="mouse_adipo_oskm_condition_screen",
        track_config={
            "condition_column": "combination_short",
            "expected_conditions": ["NT", "SOKM"],
            "control_condition": "NT",
            "full_oskm_condition": "SOKM",
            "age_column": "age",
            "batch_column": "experiment",
        },
    )

    assert report["status"] == "pass"
    assert report["missing_columns"] == []
    assert report["missing_timepoints"] == []
    assert report["missing_conditions"] == []


def test_build_validation_profile_check_reviews_missing_condition_values():
    inspection = {
        **_inspection(),
        "condition_column_present": True,
        "condition_summary": {"values": ["NT"]},
    }

    report = build_validation_profile_check(
        inspection=inspection,
        track_name="toy_condition_track",
        track_config={
            "condition_column": "condition",
            "expected_conditions": ["NT", "SOKM"],
        },
    )

    assert report["status"] == "review"
    assert report["missing_conditions"] == ["SOKM"]


def test_load_and_write_validation_profile_check_round_trip(tmp_path):
    track_config = tmp_path / "tracks.toml"
    track_config.write_text(
        """
[validation_tracks.toy_track]
dataset_profile = "toy_profile"
cell_type_column = "cell_type"
timepoint_column = "timepoint"
expected_timepoints = ["D0", "D2"]
primary_metrics = ["safe_fraction"]
""".strip(),
        encoding="utf-8",
    )
    inspection_path = tmp_path / "inspection.json"
    inspection_path.write_text(json.dumps(_inspection()), encoding="utf-8")
    output_path = tmp_path / "profile_check.json"

    report = load_validation_profile_check(
        inspection_path=inspection_path,
        track_name="toy_track",
        track_config_path=track_config,
    )
    write_validation_profile_check(report, output_path)

    assert json.loads(output_path.read_text())["status"] == "pass"
