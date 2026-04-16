from eval.validation_trajectory_dataset import (
    build_cell_alignment_delta_rows,
    build_timepoint_branch_cohort_rows,
    build_validation_trajectory_dataset,
)


def test_build_timepoint_branch_cohort_rows_aggregates_branch_metrics():
    rows = build_timepoint_branch_cohort_rows(
        track_config={"timepoint_column": "timepoint"},
        run_payloads=[
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "fused_shift_rows": [
                    {
                        "cell_id": "c1",
                        "timepoint": "D2",
                        "branch_label": "productive",
                        "l2_shift": 0.4,
                        "progress_delta": 0.3,
                        "risk_score": 0.1,
                        "partial_reprogramming_window": True,
                        "longevity_safe_zone": True,
                        "pluripotency_risk_flag": False,
                    },
                    {
                        "cell_id": "c2",
                        "timepoint": "D2",
                        "branch_label": "productive",
                        "l2_shift": 0.2,
                        "progress_delta": 0.1,
                        "risk_score": 0.2,
                        "partial_reprogramming_window": False,
                        "longevity_safe_zone": False,
                        "pluripotency_risk_flag": False,
                    },
                ],
            }
        ],
    )

    assert rows[0]["timepoint"] == "D2"
    assert rows[0]["branch_label"] == "productive"
    assert rows[0]["n_cells"] == 2
    assert rows[0]["safe_fraction"] == 0.5


def test_build_cell_alignment_delta_rows_joins_candidate_against_baseline():
    rows = build_cell_alignment_delta_rows(
        track_config={"timepoint_column": "timepoint"},
        run_payloads=[
            {
                "label": "euclidean",
                "alignment_mode": "euclidean_cosine",
                "fused_shift_rows": [
                    {
                        "cell_id": "c1",
                        "cell_type": "fibroblast",
                        "timepoint": "D2",
                        "branch_label": "alternative",
                        "l2_shift": 0.5,
                        "progress_delta": 0.0,
                        "risk_score": 0.3,
                        "pluripotency_proximity": 0.4,
                        "partial_reprogramming_window": False,
                        "longevity_safe_zone": False,
                        "pluripotency_risk_flag": True,
                    }
                ],
            },
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "fused_shift_rows": [
                    {
                        "cell_id": "c1",
                        "cell_type": "fibroblast",
                        "timepoint": "D2",
                        "branch_label": "productive",
                        "l2_shift": 0.3,
                        "progress_delta": 0.2,
                        "risk_score": 0.1,
                        "pluripotency_proximity": 0.2,
                        "partial_reprogramming_window": True,
                        "longevity_safe_zone": True,
                        "pluripotency_risk_flag": False,
                    }
                ],
            },
        ],
    )

    assert rows[0]["cell_id"] == "c1"
    assert rows[0]["delta_l2_shift"] == -0.2
    assert rows[0]["delta_safe_zone"] == 1
    assert rows[0]["baseline_branch_label"] == "alternative"
    assert rows[0]["candidate_branch_label"] == "productive"


def test_build_validation_trajectory_dataset_packages_run_rows_and_deltas():
    dataset = build_validation_trajectory_dataset(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "track": {"timepoint_column": "timepoint"},
        },
        [
            {
                "label": "projective",
                "alignment_mode": "projective_distance",
                "dataset_profile": "gse242423_human_fibroblast_oskm",
                "fused_shift_rows": [
                    {
                        "cell_id": "c1",
                        "timepoint": "D2",
                        "cell_type": "fibroblast",
                        "branch_label": "productive",
                    }
                ],
            }
        ],
    )

    assert dataset["track_name"] == "human_fibroblast_oskm"
    assert dataset["run_cell_rows"]["projective"][0]["timepoint"] == "D2"
    assert "timepoint_branch_cohorts" in dataset
