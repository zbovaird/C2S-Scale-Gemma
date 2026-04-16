from eval.validation_explorer import (
    build_validation_explorer_charts,
    build_validation_explorer_payload,
)


def test_build_validation_explorer_payload_exposes_core_sections():
    payload = build_validation_explorer_payload(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "best_by_mean_l2_shift": {"label": "projective"},
            "runs": [{"label": "projective"}],
            "timepoint_summaries": {"projective": [{"timepoint": "D2"}]},
            "timepoint_comparison": [{"timepoint": "D2"}],
            "recommendation": {"status": "prefer_projective"},
        }
    )

    assert payload["overview_cards"][0]["value"] == "projective"
    assert payload["recommendation"]["status"] == "prefer_projective"
    assert payload["explorer_sections"][0]["id"] == "run_table"
    assert payload["explorer_sections"][2]["id"] == "charts"


def test_build_validation_explorer_charts_packages_plot_ready_series():
    charts = build_validation_explorer_charts(
        {
            "timepoint_summaries": {
                "euclidean": [{"timepoint": "D0", "safe_fraction": 0.1, "risk_fraction": 0.3}],
                "projective": [{"timepoint": "D0", "safe_fraction": 0.2, "risk_fraction": 0.1}],
            },
            "timepoint_comparison": [
                {
                    "label": "projective",
                    "timepoint": "D0",
                    "delta_safe_fraction": 0.1,
                    "delta_productive_fraction": 0.05,
                }
            ],
        }
    )

    assert charts[1]["id"] == "safe_fraction_by_timepoint"
    assert charts[1]["series"]["projective"][0]["value"] == 0.2
    assert charts[3]["series"][0]["value"] == 0.1
