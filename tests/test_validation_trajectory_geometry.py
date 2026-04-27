import numpy as np

from eval.validation_trajectory_geometry import (
    build_validation_trajectory_geometry,
    compute_pairwise_geometry_distances,
    summarize_validation_trajectory_geometry,
)


def test_compute_pairwise_geometry_distances_falls_back_to_l2_when_projective_not_preferred():
    distances, backend, fallback_used = compute_pairwise_geometry_distances(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[3.0, 4.0], [1.0, 2.0]]),
        prefer_projective=False,
    )

    assert distances == [5.0, 1.0]
    assert backend == "euclidean_l2_fallback"
    assert fallback_used is False


def test_compute_pairwise_geometry_distances_falls_back_when_projective_rejects_shape():
    distances, backend, fallback_used = compute_pairwise_geometry_distances(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 1.0], [2.0, 1.0]]),
        prefer_projective=True,
    )

    assert distances == [1.0, 1.0]
    assert backend in {"projective_uhg_distance", "euclidean_l2_fallback"}
    if backend == "euclidean_l2_fallback":
        assert fallback_used is True


def test_build_validation_trajectory_geometry_exports_run_and_cell_rows():
    payload = build_validation_trajectory_geometry(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "track": {"timepoint_column": "timepoint"},
        },
        [
            {
                "label": "euclidean",
                "alignment_mode": "euclidean_cosine",
                "baseline_fused_embeddings": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "perturbed_fused_embeddings": np.array([[0.0, 1.0], [2.0, 1.0]]),
                "fused_shift_rows": [
                    {
                        "cell_id": "cell_a",
                        "cell_type": "fibroblast",
                        "timepoint": "D0",
                        "branch_label": "productive",
                        "longevity_safe_zone": True,
                    },
                    {
                        "cell_id": "cell_b",
                        "cell_type": "fibroblast",
                        "timepoint": "D2",
                    },
                ],
                "overlay_summary": {
                    "alignment": {"alignment_mode": "euclidean_cosine"}
                },
            }
        ],
    )

    assert payload["artifact_type"] == "validation_trajectory_geometry"
    assert payload["runs"][0]["geometry_distance_backend"] == "euclidean_l2_fallback"
    assert payload["runs"][0]["n_cells"] == 2
    assert payload["runs"][0]["cell_rows"][0]["cell_id"] == "cell_a"
    assert payload["runs"][0]["cell_rows"][0]["longevity_safe_zone"] is True


def test_summarize_validation_trajectory_geometry_returns_compact_rows():
    summary = summarize_validation_trajectory_geometry(
        {
            "runs": [
                {
                    "label": "euclidean",
                    "alignment_mode": "euclidean_cosine",
                    "geometry_distance_backend": "euclidean_l2_fallback",
                    "geometry_fallback_used": False,
                    "n_cells": 2,
                    "mean_geometry_distance": 1.0,
                    "max_geometry_distance": 1.5,
                    "cell_rows": [{"cell_id": "a"}],
                }
            ]
        }
    )

    assert summary == [
        {
            "label": "euclidean",
            "alignment_mode": "euclidean_cosine",
            "geometry_distance_backend": "euclidean_l2_fallback",
            "geometry_fallback_used": False,
            "n_cells": 2,
            "mean_geometry_distance": 1.0,
            "max_geometry_distance": 1.5,
        }
    ]
