from eval.validation_explorer_html import render_validation_explorer_html


def test_render_validation_explorer_html_includes_core_sections():
    html = render_validation_explorer_html(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "overview_cards": [{"label": "Recommendation", "value": "prefer_projective"}],
            "run_table": [{"label": "projective", "safe_fraction": 0.4}],
            "charts": [
                {
                    "id": "safe_fraction_by_timepoint",
                    "title": "Safe fraction across timepoints",
                    "kind": "line",
                    "series": {"projective": [{"timepoint": "D2", "value": 0.4}]},
                }
            ],
            "recommendation": {
                "evidence": {
                    "top_supporting_timepoints": [{"timepoint": "D2", "delta_safe_fraction": 0.2}],
                    "top_concerning_timepoints": [],
                }
            },
        }
    )

    assert "<title>human_fibroblast_oskm</title>" in html
    assert "Validation Explorer" in html
    assert "Safe fraction across timepoints" in html
    assert "prefer_projective" in html


def test_render_validation_explorer_html_escapes_data_values():
    html = render_validation_explorer_html(
        {
            "track_name": "<unsafe>",
            "overview_cards": [{"label": "<label>", "value": "<value>"}],
            "run_table": [{"label": "<run>"}],
            "charts": [],
            "recommendation": {"evidence": {}},
        }
    )

    assert "<title>&lt;unsafe&gt;</title>" in html
    assert "${escapeHtml(card.label)}" in html
    assert "${escapeHtml(row[col])}" in html
