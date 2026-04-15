from eval.validation_summary import (
    build_validation_benchmark_rows,
    build_validation_benchmark_summary,
)


def test_build_validation_benchmark_rows_uses_primary_metrics_and_overlay_data():
    rows = build_validation_benchmark_rows(
        {
            "report_recommendation": {
                "primary_metrics": ["mean_l2_shift", "safe_fraction"],
            }
        },
        [
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "embedding_summary": {
                    "fused_embeddings": {
                        "mean_l2_shift": 0.4,
                        "mean_cosine_similarity": 0.8,
                    }
                },
                "overlay_summary": {
                    "branch_summary": {
                        "productive": {"count": 7},
                        "alternative": {"count": 3},
                    },
                    "zone_summary": {
                        "longevity_safe_zone_fraction": 0.5,
                        "pluripotency_risk_fraction": 0.1,
                    },
                },
            }
        ],
    )

    assert rows[0]["primary_metrics"] == ["mean_l2_shift", "safe_fraction"]
    assert rows[0]["productive_fraction"] == 0.7
    assert rows[0]["safe_fraction"] == 0.5


def test_build_validation_benchmark_summary_picks_lowest_shift_run():
    summary = build_validation_benchmark_summary(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "report_recommendation": {
                "primary_metrics": ["mean_l2_shift", "safe_fraction"],
            },
        },
        [
            {
                "label": "euclidean",
                "alignment_mode": "euclidean_cosine",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "embedding_summary": {
                    "fused_embeddings": {
                        "mean_l2_shift": 0.7,
                        "mean_cosine_similarity": 0.6,
                    }
                },
                "overlay_summary": {"branch_summary": {}, "zone_summary": {}},
            },
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "embedding_summary": {
                    "fused_embeddings": {
                        "mean_l2_shift": 0.4,
                        "mean_cosine_similarity": 0.8,
                    }
                },
                "overlay_summary": {"branch_summary": {}, "zone_summary": {}},
            },
        ],
    )

    assert summary["best_by_mean_l2_shift"]["label"] == "projective"
