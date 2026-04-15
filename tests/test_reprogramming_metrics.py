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
    )

    assert len(rows) == 4
    assert rows[0]["branch_label"] in {
        "productive",
        "alternative",
        "ambiguous",
        "somatic_retention",
    }
    assert rows[0]["risk_score"] is not None


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
