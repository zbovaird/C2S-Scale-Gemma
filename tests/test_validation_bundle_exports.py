import json
from pathlib import Path

import numpy as np

from eval.validation_bundle_exports import (
    build_validation_artifact_paths,
    export_validation_bundle_artifacts,
)


def test_build_validation_artifact_paths_returns_expected_bundle_outputs():
    paths = build_validation_artifact_paths("artifacts/validation_bundle/human_fibroblast_oskm")

    assert paths["summary_json"].endswith("validation_benchmark_summary.json")
    assert paths["explorer_html"].endswith("validation_explorer.html")
    assert paths["trajectory_dataset"].endswith("validation_trajectory_dataset.json")
    assert paths["trajectory_geometry"].endswith("validation_trajectory_geometry.json")
    assert paths["trajectory_projection_html"].endswith(
        "validation_trajectory_projection.html"
    )


def _write_run_artifacts(run_dir: Path, cell_prefix: str, baseline_offset: float) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "embedding_shift_summary.json").write_text(
        json.dumps(
            {
                "fused_embeddings": {
                    "mean_l2_shift": 0.3 + baseline_offset,
                    "mean_cosine_similarity": 0.8,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "reprogramming_overlay_summary.json").write_text(
        json.dumps(
            {
                "branch_summary": {"productive": {"count": 1}},
                "zone_summary": {
                    "longevity_safe_zone_fraction": 1.0,
                    "pluripotency_risk_fraction": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "fused_embedding_shift_frame.json").write_text(
        json.dumps(
            [
                {
                    "cell_id": f"{cell_prefix}1",
                    "cell_type": "fibroblast",
                    "timepoint": "D2",
                    "l2_shift": 0.3,
                    "progress_delta": 0.2,
                    "branch_label": "productive",
                    "risk_score": 0.1,
                    "partial_reprogramming_window": True,
                    "longevity_safe_zone": True,
                    "pluripotency_risk_flag": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    np.save(run_dir / "baseline_fused_embeddings.npy", np.array([[baseline_offset, 0.0]]))
    np.save(run_dir / "perturbed_fused_embeddings.npy", np.array([[baseline_offset, 1.0]]))


def test_export_validation_bundle_artifacts_writes_main_outputs(tmp_path):
    euclidean_dir = tmp_path / "euclidean"
    projective_dir = tmp_path / "projective"
    _write_run_artifacts(euclidean_dir, "e", 0.0)
    _write_run_artifacts(projective_dir, "p", 10.0)

    ablation_manifest_path = tmp_path / "ablation_manifest.json"
    ablation_manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "euclidean",
                        "alignment_mode": "euclidean_cosine",
                        "dataset_profile": "gse242423_human_fibroblast_oskm",
                        "output_dir": str(euclidean_dir),
                    },
                    {
                        "label": "projective",
                        "alignment_mode": "projective_distance",
                        "dataset_profile": "gse242423_human_fibroblast_oskm",
                        "output_dir": str(projective_dir),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    validation_manifest_path = tmp_path / "validation_bundle.json"
    validation_manifest_path.write_text(
        json.dumps(
            {
                "track_name": "human_fibroblast_oskm",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "track": {"timepoint_column": "timepoint"},
                "report_recommendation": {"primary_metrics": ["safe_fraction"]},
                "ablation_manifest_path": str(ablation_manifest_path),
            }
        ),
        encoding="utf-8",
    )

    artifact_paths = export_validation_bundle_artifacts(validation_manifest_path)
    projection = json.loads(Path(artifact_paths["trajectory_projection"]).read_text())
    geometry = json.loads(Path(artifact_paths["trajectory_geometry"]).read_text())

    assert Path(artifact_paths["summary_json"]).exists()
    assert Path(artifact_paths["explorer_html"]).exists()
    assert Path(artifact_paths["trajectory_dataset"]).exists()
    assert Path(artifact_paths["trajectory_geometry"]).exists()
    assert Path(artifact_paths["trajectory_projection_html"]).exists()
    assert projection["projection_method"] == "shared_pca"
    assert geometry["artifact_type"] == "validation_trajectory_geometry"
