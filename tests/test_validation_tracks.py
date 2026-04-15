from eval.validation_tracks import (
    build_validation_bundle_manifest,
    resolve_validation_track,
)


def test_resolve_validation_track_returns_named_track():
    track = resolve_validation_track(
        "human_fibroblast_oskm",
        "configs/validation_tracks.toml",
    )

    assert track["dataset_profile"] == "gse242423_human_fibroblast_oskm"
    assert "expected_timepoints" in track
    assert "recommendation" in track


def test_build_validation_bundle_manifest_links_track_and_ablation_manifest():
    manifest = build_validation_bundle_manifest(
        track_name="human_fibroblast_oskm",
        track_config={
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "primary_metrics": ["mean_l2_shift", "safe_fraction"],
        },
        output_root="artifacts/validation_bundle/human_fibroblast_oskm",
        baseline_data_path="data/raw/GSE242423.h5ad",
        perturbed_data_path="artifacts/GSE242423_perturbed.h5ad",
        ablation_manifest_path="artifacts/validation_bundle/human_fibroblast_oskm/ablation_manifest.json",
    )

    assert manifest["track_name"] == "human_fibroblast_oskm"
    assert manifest["dataset_profile"] == "gse242423_human_fibroblast_oskm"
    assert manifest["report_recommendation"]["primary_metrics"] == [
        "mean_l2_shift",
        "safe_fraction",
    ]
