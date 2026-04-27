from eval.validation_summary import (
    build_alignment_recommendation,
    build_interpretation_limits,
    build_recommendation_evidence,
    build_validation_benchmark_rows,
    build_validation_benchmark_summary,
    build_timepoint_comparison_rows,
    build_timepoint_progression_rows,
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
                "geometry_distance_backend": "projective_uhg_distance",
                "geometry_fallback_used": False,
                "fusion_graph_source": "euclidean_embeddings",
                "alignment_graph_source": "hyperbolic_embeddings",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "embedding_summary": {
                    "fused_embeddings": {
                        "mean_l2_shift": 0.4,
                        "mean_cosine_similarity": 0.8,
                    }
                },
                "overlay_summary": {
                    "alignment": {
                        "geometry_distance_backend": "projective_uhg_distance",
                        "geometry_fallback_used": False,
                    },
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
    assert rows[0]["geometry_distance_backend"] == "projective_uhg_distance"
    assert rows[0]["geometry_fallback_used"] is False
    assert rows[0]["alignment_graph_source"] == "hyperbolic_embeddings"


def test_build_validation_benchmark_summary_picks_lowest_shift_run():
    summary = build_validation_benchmark_summary(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "track": {
                "recommendation": {
                    "min_safe_fraction_gain": 0.05,
                    "min_productive_fraction_gain": 0.05,
                    "max_risk_fraction_increase": 0.03,
                    "max_mean_l2_shift_increase": 0.10,
                    "min_timepoint_safe_gains": 1,
                }
            },
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
                "fused_shift_rows": [{"timepoint": "D2", "l2_shift": 0.7, "progress_delta": 0.0}],
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
                "overlay_summary": {
                    "branch_summary": {"productive": {"count": 7}, "alternative": {"count": 3}},
                    "zone_summary": {
                        "longevity_safe_zone_fraction": 0.2,
                        "pluripotency_risk_fraction": 0.0,
                    },
                },
                "fused_shift_rows": [{"timepoint": "D2", "l2_shift": 0.4, "progress_delta": 0.1, "branch_label": "productive", "longevity_safe_zone": True, "pluripotency_risk_flag": False}],
            },
        ],
    )

    assert summary["best_by_mean_l2_shift"]["label"] == "projective"
    assert "interpretation_limits" in summary


def test_build_timepoint_progression_rows_uses_expected_timepoint_order():
    rows = build_timepoint_progression_rows(
        track_config={
            "timepoint_column": "timepoint",
            "expected_timepoints": ["D0", "D2", "D4"],
        },
        fused_shift_rows=[
            {
                "timepoint": "D2",
                "l2_shift": 0.4,
                "progress_delta": 0.2,
                "branch_label": "productive",
                "longevity_safe_zone": True,
                "pluripotency_risk_flag": False,
            },
            {
                "timepoint": "D0",
                "l2_shift": 0.1,
                "progress_delta": 0.0,
                "branch_label": "alternative",
                "longevity_safe_zone": False,
                "pluripotency_risk_flag": False,
            },
        ],
    )

    assert [row["timepoint"] for row in rows] == ["D0", "D2"]
    assert rows[1]["productive_fraction"] == 1.0
    assert rows[1]["safe_fraction"] == 1.0


def test_build_timepoint_comparison_rows_compares_against_euclidean():
    rows = build_timepoint_comparison_rows(
        {
            "euclidean": [
                {
                    "timepoint": "D2",
                    "mean_l2_shift": 0.5,
                    "productive_fraction": 0.4,
                    "safe_fraction": 0.2,
                    "risk_fraction": 0.3,
                    "mean_progress_delta": 0.1,
                }
            ],
            "projective": [
                {
                    "timepoint": "D2",
                    "mean_l2_shift": 0.4,
                    "productive_fraction": 0.7,
                    "safe_fraction": 0.5,
                    "risk_fraction": 0.1,
                    "mean_progress_delta": 0.2,
                }
            ],
        }
    )

    assert rows[0]["label"] == "projective"
    assert rows[0]["delta_safe_fraction"] == 0.3
    assert rows[0]["delta_risk_fraction"] == -0.19999999999999998


def test_build_alignment_recommendation_prefers_projective_when_thresholds_clear():
    recommendation = build_alignment_recommendation(
        track_config={
            "recommendation": {
                "min_safe_fraction_gain": 0.05,
                "min_productive_fraction_gain": 0.05,
                "max_risk_fraction_increase": 0.03,
                "max_mean_l2_shift_increase": 0.10,
                "min_timepoint_safe_gains": 1,
            }
        },
        benchmark_rows=[
            {
                "label": "euclidean",
                "alignment_mode": "euclidean_cosine",
                "mean_l2_shift": 0.5,
                "productive_fraction": 0.4,
                "safe_fraction": 0.2,
                "risk_fraction": 0.2,
            },
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "mean_l2_shift": 0.45,
                "productive_fraction": 0.6,
                "safe_fraction": 0.35,
                "risk_fraction": 0.2,
            },
        ],
        timepoint_comparison=[
            {
                "label": "projective",
                "timepoint": "D2",
                "delta_safe_fraction": 0.1,
            }
        ],
    )

    assert recommendation["status"] == "prefer_projective"
    assert recommendation["preferred_alignment"] == "projective_distance"
    assert recommendation["evidence"]["top_supporting_timepoints"][0]["timepoint"] == "D2"


def test_build_recommendation_evidence_surfaces_best_and_worst_timepoints():
    evidence = build_recommendation_evidence(
        [
            {
                "timepoint": "D2",
                "delta_safe_fraction": 0.2,
                "delta_productive_fraction": 0.1,
                "delta_risk_fraction": -0.1,
            },
            {
                "timepoint": "D6",
                "delta_safe_fraction": -0.1,
                "delta_productive_fraction": -0.1,
                "delta_risk_fraction": 0.2,
            },
        ],
        top_k=1,
    )

    assert evidence["top_supporting_timepoints"][0]["timepoint"] == "D2"
    assert evidence["top_concerning_timepoints"][0]["timepoint"] == "D6"


def test_build_interpretation_limits_includes_defaults_and_track_specific_notes():
    limits = build_interpretation_limits(
        {
            "track": {
                "interpretation_limits": [
                    "Track-specific calibration is required before biological use."
                ]
            }
        }
    )

    assert any("representation-level evidence" in limit for limit in limits)
    assert "Track-specific calibration is required before biological use." in limits
