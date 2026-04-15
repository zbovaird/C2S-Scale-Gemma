from eval.validation_visuals import (
    build_timepoint_delta_series,
    build_timepoint_metric_series,
)


def test_build_timepoint_metric_series_returns_per_run_values():
    series = build_timepoint_metric_series(
        {
            "timepoint_summaries": {
                "euclidean": [
                    {"timepoint": "D0", "safe_fraction": 0.1},
                    {"timepoint": "D2", "safe_fraction": 0.2},
                ]
            }
        },
        "safe_fraction",
    )

    assert series["euclidean"][0]["timepoint"] == "D0"
    assert series["euclidean"][1]["value"] == 0.2


def test_build_timepoint_delta_series_returns_comparison_values():
    rows = build_timepoint_delta_series(
        {
            "timepoint_comparison": [
                {"label": "projective", "timepoint": "D2", "delta_safe_fraction": 0.3}
            ]
        },
        "delta_safe_fraction",
    )

    assert rows[0]["label"] == "projective"
    assert rows[0]["value"] == 0.3
