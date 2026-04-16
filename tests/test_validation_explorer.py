from eval.validation_explorer import build_validation_explorer_payload


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
