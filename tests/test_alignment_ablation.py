from eval.alignment_ablation import build_ablation_manifest, build_paired_alignment_runs


def test_build_paired_alignment_runs_creates_expected_labels_and_dirs():
    runs = build_paired_alignment_runs(
        output_root="artifacts/alignment_ablation",
        euclidean_config="configs/euclidean.toml",
        euclidean_checkpoint="artifacts/euclidean.pt",
        projective_config="configs/projective.toml",
        projective_checkpoint="artifacts/projective.pt",
    )

    assert runs[0]["label"] == "euclidean"
    assert runs[1]["label"] == "projective"
    assert runs[0]["output_dir"].endswith("alignment_ablation/euclidean")
    assert runs[1]["output_dir"].endswith("alignment_ablation/projective")


def test_build_ablation_manifest_preserves_run_results():
    manifest = build_ablation_manifest(
        output_root="artifacts/alignment_ablation",
        baseline_data_path="data/raw/base.h5ad",
        perturbed_data_path="data/raw/perturbed.h5ad",
        dataset_profile="gse242423_human_fibroblast_oskm",
        run_results=[
            {"label": "euclidean", "output_dir": "artifacts/alignment_ablation/euclidean"},
            {"label": "projective", "output_dir": "artifacts/alignment_ablation/projective"},
        ],
    )

    assert manifest["dataset_profile"] == "gse242423_human_fibroblast_oskm"
    assert len(manifest["runs"]) == 2
