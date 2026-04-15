import numpy as np

from eval.reprogramming_metrics import (
    build_reprogramming_overlay_rows,
    summarize_branch_counts,
    summarize_zone_counts,
)


def test_build_reprogramming_overlay_rows_returns_branch_and_risk():
    baseline = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    perturbed = np.array(
        [
            [0.2, 0.2],
            [0.3, 0.2],
            [0.8, 0.8],
            [0.9, 0.8],
        ]
    )
    cell_types = ["fibroblast", "fibroblast", "ipsc", "ipsc"]

    rows = build_reprogramming_overlay_rows(
        baseline,
        perturbed,
        cell_types=cell_types,
        baseline_oskm_scores=[0.1, 0.2, 0.8, 0.9],
        perturbed_oskm_scores=[0.2, 0.3, 0.9, 1.0],
        baseline_rejuvenation_scores=[0.1, 0.1, 0.2, 0.2],
        perturbed_rejuvenation_scores=[0.6, 0.7, 0.3, 0.4],
        baseline_pluripotency_marker_scores=[0.0, 0.0, 0.3, 0.4],
        perturbed_pluripotency_marker_scores=[0.1, 0.2, 0.8, 0.9],
    )

    assert len(rows) == 4
    assert rows[0]["branch_label"] in {
        "productive",
        "alternative",
        "ambiguous",
        "somatic_retention",
    }
    assert rows[0]["risk_score"] is not None
    assert "rejuvenation_score" in rows[0]
    assert "pluripotency_marker_score" in rows[0]


def test_build_reprogramming_overlay_rows_respects_custom_window_profile():
    baseline = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    perturbed = np.array(
        [
            [0.4, 0.4],
            [0.5, 0.4],
            [0.8, 0.8],
            [0.9, 0.8],
        ]
    )
    rows = build_reprogramming_overlay_rows(
        baseline,
        perturbed,
        cell_types=["fibroblast", "fibroblast", "ipsc", "ipsc"],
        perturbed_oskm_scores=[0.2, 0.3, 0.9, 1.0],
        perturbed_rejuvenation_scores=[0.8, 0.9, 0.2, 0.3],
        perturbed_pluripotency_marker_scores=[0.1, 0.1, 0.9, 1.0],
        window_profile={
            "partial_window_proximity_min": 0.0,
            "partial_window_proximity_max": 1.0,
            "partial_window_max_risk": 1.0,
            "longevity_safe_max_risk": 1.0,
            "min_rejuvenation_score": 0.0,
            "safe_max_pluripotency_marker": 1.0,
        },
    )

    assert any(row["partial_reprogramming_window"] for row in rows)


def test_summarize_branch_counts_aggregates_risk():
    summary = summarize_branch_counts(
        [
            {"branch_label": "productive", "risk_score": 0.2},
            {"branch_label": "productive", "risk_score": 0.4},
            {"branch_label": "alternative", "risk_score": 0.9},
        ]
    )

    assert summary["productive"]["count"] == 2
    assert summary["productive"]["mean_risk_score"] == 0.30000000000000004
    assert summary["alternative"]["count"] == 1


def test_summarize_zone_counts_tracks_partial_and_safe_flags():
    summary = summarize_zone_counts(
        [
            {
                "partial_reprogramming_window": True,
                "longevity_safe_zone": True,
                "pluripotency_risk_flag": False,
            },
            {
                "partial_reprogramming_window": True,
                "longevity_safe_zone": False,
                "pluripotency_risk_flag": True,
            },
        ]
    )

    assert summary["partial_reprogramming_window_count"] == 2
    assert summary["longevity_safe_zone_count"] == 1
    assert summary["pluripotency_risk_count"] == 1
