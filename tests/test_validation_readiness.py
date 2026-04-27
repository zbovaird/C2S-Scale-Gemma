import json

from eval.validation_readiness import (
    build_validation_dataset_readiness,
    build_validation_readiness_report,
    write_validation_readiness_report,
)


def test_build_validation_dataset_readiness_reports_needs_data_when_paths_missing():
    row = build_validation_dataset_readiness(
        track_name="human_fibroblast_oskm",
        track_config={
            "title": "Human fibroblast OSKM",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "baseline_data_hint": "missing_baseline.h5ad",
            "perturbed_data_hint": "missing_perturbed.h5ad",
            "cell_type_column": "cell_type",
            "timepoint_column": "timepoint",
            "expected_timepoints": ["D0", "D2"],
            "primary_metrics": ["safe_fraction"],
        },
        profile_registry={
            "reprogramming_profiles": {
                "gse242423_human_fibroblast_oskm": {
                    "accession": "GSE242423",
                    "species": "human",
                    "source_url": "https://example.test",
                    "baseline_data_hint": "missing_baseline.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                }
            }
        },
    )

    assert row["status"] == "needs_data"
    assert row["profile_present"] is True
    assert row["baseline_data_exists"] is False


def test_build_validation_dataset_readiness_reports_ready_when_data_exists(tmp_path):
    baseline_path = tmp_path / "baseline.h5ad"
    perturbed_path = tmp_path / "perturbed.h5ad"
    baseline_path.write_text("placeholder", encoding="utf-8")
    perturbed_path.write_text("placeholder", encoding="utf-8")

    row = build_validation_dataset_readiness(
        track_name="human_fibroblast_oskm",
        track_config={
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "baseline_data_hint": str(baseline_path),
            "perturbed_data_hint": str(perturbed_path),
            "cell_type_column": "cell_type",
            "timepoint_column": "timepoint",
            "expected_timepoints": ["D0", "D2"],
            "primary_metrics": ["safe_fraction"],
        },
        profile_registry={
            "reprogramming_profiles": {
                "gse242423_human_fibroblast_oskm": {
                    "accession": "GSE242423",
                    "species": "human",
                    "source_url": "https://example.test",
                    "baseline_data_hint": str(baseline_path),
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                }
            }
        },
    )

    assert row["status"] == "ready"
    assert row["perturbed_data_exists"] is True


def test_build_validation_readiness_report_counts_statuses():
    report = build_validation_readiness_report(
        track_registry={
            "validation_tracks": {
                "missing_profile": {
                    "dataset_profile": "missing",
                    "baseline_data_hint": "missing.h5ad",
                    "perturbed_data_hint": "missing_perturbed.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                    "expected_timepoints": ["D0"],
                    "primary_metrics": ["safe_fraction"],
                }
            }
        },
        profile_registry={"reprogramming_profiles": {}},
    )

    assert report["status_counts"]["incomplete_metadata"] == 1
    assert report["tracks"][0]["profile_present"] is False


def test_write_validation_readiness_report_writes_json(tmp_path):
    output_path = tmp_path / "readiness.json"
    write_validation_readiness_report({"status_counts": {"needs_data": 1}}, output_path)

    assert json.loads(output_path.read_text())["status_counts"]["needs_data"] == 1
