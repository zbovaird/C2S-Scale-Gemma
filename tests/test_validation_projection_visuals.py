from eval.validation_projection_visuals import (
    build_projection_arrow_rows,
    build_projection_phase_series,
    build_projection_timepoint_series,
)


def test_build_projection_phase_series_groups_rows_by_branch():
    series = build_projection_phase_series(
        {
            "rows": [
                {
                    "cell_id": "c1",
                    "timepoint": "D2",
                    "branch_label": "productive",
                    "perturbed_x": 1.0,
                    "perturbed_y": 2.0,
                },
                {
                    "cell_id": "c2",
                    "timepoint": "D4",
                    "branch_label": "alternative",
                    "perturbed_x": 3.0,
                    "perturbed_y": 4.0,
                },
            ]
        }
    )

    assert series["productive"][0]["x"] == 1.0
    assert series["alternative"][0]["timepoint"] == "D4"


def test_build_projection_arrow_rows_returns_segments():
    rows = build_projection_arrow_rows(
        {
            "rows": [
                {
                    "cell_id": "c1",
                    "timepoint": "D2",
                    "baseline_x": 0.0,
                    "baseline_y": 0.0,
                    "perturbed_x": 1.0,
                    "perturbed_y": 2.0,
                    "branch_label": "productive",
                    "longevity_safe_zone": True,
                    "pluripotency_risk_flag": False,
                }
            ]
        }
    )

    assert rows[0]["x1"] == 1.0
    assert rows[0]["longevity_safe_zone"] is True


def test_build_projection_timepoint_series_groups_stage_rows():
    series = build_projection_timepoint_series(
        {
            "rows": [
                {
                    "cell_id": "c1",
                    "timepoint": "D0",
                    "baseline_x": 0.0,
                    "baseline_y": 1.0,
                },
                {
                    "cell_id": "c2",
                    "timepoint": "D2",
                    "baseline_x": 2.0,
                    "baseline_y": 3.0,
                },
            ]
        },
        phase="baseline",
    )

    assert series["D0"][0]["y"] == 1.0
    assert series["D2"][0]["x"] == 2.0
