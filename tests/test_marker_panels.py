import numpy as np

from biology.panels import merge_marker_panels, resolve_marker_panel_genes, score_expression_panels


def test_resolve_marker_panel_genes_matches_case_insensitive_symbols():
    resolved = resolve_marker_panel_genes(
        ["Sirt1", "Foxo3", "Nanog"],
        ["SIRT1", "FOXO3", "PRDM14"],
    )

    assert resolved == ["Sirt1", "Foxo3"]


def test_score_expression_panels_returns_per_panel_scores():
    scores = score_expression_panels(
        np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 1.5]]),
        ["SIRT1", "FOXO3", "NANOG"],
        {
            "rejuvenation": ["SIRT1", "FOXO3"],
            "pluripotency_risk": ["NANOG"],
        },
    )

    assert np.allclose(scores["rejuvenation"], np.array([3.0, 0.5]))
    assert np.allclose(scores["pluripotency_risk"], np.array([3.0, 1.5]))


def test_merge_marker_panels_overrides_defaults():
    merged = merge_marker_panels({"rejuvenation": ["SIRT6"]})

    assert merged["rejuvenation"] == ("SIRT6",)
    assert "pluripotency_risk" in merged
