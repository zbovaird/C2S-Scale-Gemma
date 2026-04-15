from eval.reprogramming_profiles import resolve_dataset_profile


def test_resolve_dataset_profile_merges_profile_sections():
    base_config = {
        "data": {"oskm": {"species": "human"}},
        "reprogramming": {
            "references": {"somatic_labels": ["fibroblast"]},
            "window_profile": {"partial_window_max_risk": 0.6},
            "marker_panels": {"rejuvenation": ["SIRT1"]},
        },
    }

    merged_config, manifest = resolve_dataset_profile(
        base_config,
        profile_name="gse176206_mouse_transient_partial",
        profile_config_path="configs/reprogramming_profiles.toml",
    )

    assert merged_config["data"]["oskm"]["species"] == "mouse"
    assert merged_config["reprogramming"]["dataset_profile"] == "gse176206_mouse_transient_partial"
    assert "pluripotent_labels" in merged_config["reprogramming"]["references"]
    assert "pluripotency_risk" in merged_config["reprogramming"]["marker_panels"]
    assert manifest["accession"] == "GSE176206"


def test_resolve_dataset_profile_returns_original_when_not_requested():
    base_config = {"reprogramming": {"references": {"somatic_labels": ["fibroblast"]}}}

    merged_config, manifest = resolve_dataset_profile(
        base_config,
        profile_name=None,
        profile_config_path="configs/reprogramming_profiles.toml",
    )

    assert merged_config == base_config
    assert manifest is None
