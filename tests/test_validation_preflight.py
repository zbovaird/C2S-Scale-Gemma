import json
from pathlib import Path

from eval.validation_preflight import (
    build_validation_artifact_qa,
    build_validation_input_preflight,
)


def test_build_validation_input_preflight_passes_with_required_files(tmp_path):
    baseline_path = tmp_path / "baseline.h5ad"
    perturbed_path = tmp_path / "perturbed.h5ad"
    euclidean_config = tmp_path / "euclidean.toml"
    euclidean_checkpoint = tmp_path / "euclidean.pt"
    projective_config = tmp_path / "projective.toml"
    projective_checkpoint = tmp_path / "projective.pt"
    profile_config = tmp_path / "profiles.toml"
    for path in (
        baseline_path,
        perturbed_path,
        euclidean_config,
        euclidean_checkpoint,
        projective_config,
        projective_checkpoint,
        profile_config,
    ):
        path.write_text("placeholder", encoding="utf-8")

    report = build_validation_input_preflight(
        track_name="human_fibroblast_oskm",
        track_config={
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "cell_type_column": "cell_type",
            "timepoint_column": "timepoint",
            "expected_timepoints": ["D0", "D2"],
            "primary_metrics": ["safe_fraction"],
        },
        baseline_data_path=baseline_path,
        perturbed_data_path=perturbed_path,
        euclidean_config=euclidean_config,
        euclidean_checkpoint=euclidean_checkpoint,
        projective_config=projective_config,
        projective_checkpoint=projective_checkpoint,
        dataset_profile_config=profile_config,
    )

    assert report["status"] == "pass"
    assert report["n_failed"] == 0


def test_build_validation_input_preflight_fails_missing_required_inputs(tmp_path):
    report = build_validation_input_preflight(
        track_name="human_fibroblast_oskm",
        track_config={"dataset_profile": "gse242423_human_fibroblast_oskm"},
        baseline_data_path=tmp_path / "missing_baseline.h5ad",
        perturbed_data_path=tmp_path / "missing_perturbed.h5ad",
        euclidean_config=tmp_path / "missing_euclidean.toml",
        euclidean_checkpoint=tmp_path / "missing_euclidean.pt",
        projective_config=tmp_path / "missing_projective.toml",
        projective_checkpoint=tmp_path / "missing_projective.pt",
        dataset_profile_config=tmp_path / "missing_profiles.toml",
    )

    failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
    assert report["status"] == "fail"
    assert "track_field:timepoint_column" in failed_ids
    assert "baseline_data_path" in failed_ids


def test_build_validation_input_preflight_accepts_condition_screen_track(tmp_path):
    paths = {
        name: tmp_path / name
        for name in (
            "baseline.h5ad",
            "perturbed.h5ad",
            "euclidean.toml",
            "euclidean.pt",
            "projective.toml",
            "projective.pt",
            "profiles.toml",
        )
    }
    for path in paths.values():
        path.write_text("placeholder", encoding="utf-8")

    report = build_validation_input_preflight(
        track_name="mouse_adipo_oskm_condition_screen",
        track_config={
            "dataset_profile": "gse176206_mouse_adipo_condition_screen",
            "condition_column": "combination_short",
            "expected_conditions": ["NT", "SOKM"],
            "primary_metrics": ["safe_fraction"],
        },
        baseline_data_path=paths["baseline.h5ad"],
        perturbed_data_path=paths["perturbed.h5ad"],
        euclidean_config=paths["euclidean.toml"],
        euclidean_checkpoint=paths["euclidean.pt"],
        projective_config=paths["projective.toml"],
        projective_checkpoint=paths["projective.pt"],
        dataset_profile_config=paths["profiles.toml"],
    )

    failed_ids = {check["id"] for check in report["checks"] if not check["passed"]}
    assert report["status"] == "pass"
    assert "track_field:timepoint_column" not in failed_ids
    assert "track_field:cell_type_column" not in failed_ids


def test_build_validation_artifact_qa_validates_manifest_outputs(tmp_path):
    summary_json = tmp_path / "validation_benchmark_summary.json"
    explorer_payload = tmp_path / "validation_explorer_payload.json"
    trajectory_dataset = tmp_path / "validation_trajectory_dataset.json"
    trajectory_geometry = tmp_path / "validation_trajectory_geometry.json"
    trajectory_projection = tmp_path / "validation_trajectory_projection.json"
    markdown = tmp_path / "VALIDATION_BENCHMARK.md"
    explorer_html = tmp_path / "validation_explorer.html"
    projection_html = tmp_path / "validation_trajectory_projection.html"
    for path in (
        summary_json,
        explorer_payload,
        trajectory_dataset,
        trajectory_geometry,
        trajectory_projection,
    ):
        path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    for path in (markdown, explorer_html, projection_html):
        path.write_text("ok", encoding="utf-8")

    manifest_path = tmp_path / "validation_artifacts_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_markdown": str(markdown),
                "explorer_payload": str(explorer_payload),
                "explorer_html": str(explorer_html),
                "trajectory_dataset": str(trajectory_dataset),
                "trajectory_geometry": str(trajectory_geometry),
                "trajectory_projection": str(trajectory_projection),
                "trajectory_projection_html": str(projection_html),
            }
        ),
        encoding="utf-8",
    )

    report = build_validation_artifact_qa(manifest_path)

    assert report["status"] == "pass"
    assert report["n_failed"] == 0


def test_build_validation_artifact_qa_flags_missing_manifest(tmp_path):
    report = build_validation_artifact_qa(tmp_path / "missing_manifest.json")

    assert report["status"] == "fail"
    assert report["checks"][0]["id"] == "artifact_manifest"
