import numpy as np

from biology.regulatory_screening import (
    build_regulatory_pathway_scores,
    build_regulatory_screening_rows,
    summarize_regulatory_screening_rows,
)


def test_build_regulatory_pathway_scores_scores_configured_panels():
    scores = build_regulatory_pathway_scores(
        np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]),
        ["POU5F1", "SOX2", "TFAM"],
        panels={
            "oskm_core": ["POU5F1", "SOX2"],
            "metabolic": ["TFAM"],
        },
    )

    assert scores["oskm_core"].tolist() == [3.0, 1.0]
    assert scores["metabolic"].tolist() == [3.0, 0.0]


def test_build_regulatory_screening_rows_flags_high_signal_cells():
    rows = build_regulatory_screening_rows(
        expression_matrix=np.array([[10.0, 10.0], [1.0, 0.0], [0.0, 1.0]]),
        var_names=["POU5F1", "SOX2"],
        cell_ids=["a", "b", "c"],
        cell_types=["fibroblast", "pbmc", "pbmc"],
        panels={"oskm_core": ["POU5F1", "SOX2"]},
        high_signal_quantile=0.67,
    )

    assert rows[0]["cell_id"] == "a"
    assert rows[0]["oskm_core_score"] == 20.0
    assert rows[0]["high_regulatory_signal"] is True
    assert rows[1]["high_regulatory_signal"] is False


def test_summarize_regulatory_screening_rows_reports_fraction_and_mean():
    summary = summarize_regulatory_screening_rows(
        [
            {"regulatory_screen_score": 2.0, "high_regulatory_signal": True},
            {"regulatory_screen_score": 0.0, "high_regulatory_signal": False},
        ]
    )

    assert summary["n_cells"] == 2
    assert summary["high_signal_fraction"] == 0.5
    assert summary["mean_regulatory_screen_score"] == 1.0
