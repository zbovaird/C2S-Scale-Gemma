import numpy as np

from eval.validation_trajectory_projection import (
    build_projection_rows,
    build_validation_trajectory_projection,
    project_embedding_pair_to_2d,
)


def test_project_embedding_pair_to_2d_returns_paired_coordinates():
    baseline, perturbed = project_embedding_pair_to_2d(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
    )

    assert baseline.shape == (2, 2)
    assert perturbed.shape == (2, 2)


def test_build_projection_rows_preserves_annotations_and_deltas():
    rows = build_projection_rows(
        label="projective",
        alignment_mode="projective_distance",
        baseline_projection=np.array([[0.0, 0.0]]),
        perturbed_projection=np.array([[1.0, 2.0]]),
        fused_shift_rows=[
            {
                "cell_id": "c1",
                "cell_type": "fibroblast",
                "timepoint": "D2",
                "branch_label": "productive",
                "partial_reprogramming_window": True,
                "longevity_safe_zone": True,
                "pluripotency_risk_flag": False,
                "l2_shift": 0.4,
                "progress_delta": 0.2,
                "risk_score": 0.1,
            }
        ],
        timepoint_column="timepoint",
    )

    assert rows[0]["timepoint"] == "D2"
    assert rows[0]["delta_x"] == 1.0
    assert rows[0]["delta_y"] == 2.0
    assert rows[0]["longevity_safe_zone"] is True


def test_build_validation_trajectory_projection_packages_run_rows():
    projection = build_validation_trajectory_projection(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "track": {"timepoint_column": "timepoint"},
        },
        [
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "baseline_fused_embeddings": np.array([[0.0, 0.0, 1.0]]),
                "perturbed_fused_embeddings": np.array([[0.5, 0.5, 1.0]]),
                "fused_shift_rows": [
                    {
                        "cell_id": "c1",
                        "cell_type": "fibroblast",
                        "timepoint": "D2",
                        "branch_label": "productive",
                    }
                ],
            }
        ],
    )

    assert projection["projection_method"] == "pca"
    assert projection["runs"][0]["rows"][0]["cell_id"] == "c1"
