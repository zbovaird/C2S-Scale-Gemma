from eval.validation_bundle_exports import build_validation_artifact_paths


def test_build_validation_artifact_paths_returns_expected_bundle_outputs():
    paths = build_validation_artifact_paths("artifacts/validation_bundle/human_fibroblast_oskm")

    assert paths["summary_json"].endswith("validation_benchmark_summary.json")
    assert paths["explorer_html"].endswith("validation_explorer.html")
    assert paths["trajectory_dataset"].endswith("validation_trajectory_dataset.json")
    assert paths["trajectory_projection_html"].endswith(
        "validation_trajectory_projection.html"
    )
