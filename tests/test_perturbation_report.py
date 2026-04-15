from eval.perturbation_report import (
    compute_shift_histogram,
    get_top_shift_rows,
    summarize_boolean_flag,
    summarize_risk_by_branch,
    summarize_shift_by_category,
    summarize_value_by_category,
)


def test_summarize_shift_by_category_orders_by_mean_shift():
    rows = [
        {"cell_type": "fibroblast", "l2_shift": 1.0},
        {"cell_type": "fibroblast", "l2_shift": 3.0},
        {"cell_type": "intermediate", "l2_shift": 5.0},
    ]

    summary = summarize_shift_by_category(rows, "cell_type")

    assert summary[0]["category"] == "intermediate"
    assert summary[1]["category"] == "fibroblast"
    assert summary[1]["mean_shift"] == 2.0


def test_get_top_shift_rows_returns_largest_rows_first():
    rows = [
        {"cell_id": "c1", "l2_shift": 0.1},
        {"cell_id": "c2", "l2_shift": 1.2},
        {"cell_id": "c3", "l2_shift": 0.7},
    ]

    top_rows = get_top_shift_rows(rows, top_n=2)

    assert [row["cell_id"] for row in top_rows] == ["c2", "c3"]


def test_compute_shift_histogram_counts_all_rows():
    rows = [{"l2_shift": value} for value in (0.0, 0.5, 1.0, 1.5)]
    histogram = compute_shift_histogram(rows, bins=2)

    assert sum(histogram["counts"]) == 4
    assert len(histogram["bin_edges"]) == 3


def test_summarize_risk_by_branch_orders_highest_risk_first():
    rows = [
        {"branch_label": "productive", "risk_score": 0.3},
        {"branch_label": "productive", "risk_score": 0.5},
        {"branch_label": "alternative", "risk_score": 0.9},
    ]

    summary = summarize_risk_by_branch(rows)

    assert summary[0]["branch_label"] == "alternative"
    assert summary[1]["branch_label"] == "productive"


def test_summarize_boolean_flag_counts_true_rows():
    rows = [
        {"longevity_safe_zone": True},
        {"longevity_safe_zone": False},
        {"longevity_safe_zone": True},
    ]

    summary = summarize_boolean_flag(rows, "longevity_safe_zone")

    assert summary["count"] == 2
    assert summary["n_cells"] == 3


def test_summarize_value_by_category_aggregates_generic_metric():
    rows = [
        {"branch_label": "productive", "rejuvenation_score": 0.2},
        {"branch_label": "productive", "rejuvenation_score": 0.8},
        {"branch_label": "alternative", "rejuvenation_score": 0.4},
    ]

    summary = summarize_value_by_category(rows, "branch_label", "rejuvenation_score")

    assert summary[0]["category"] == "productive"
    assert summary[0]["mean_value"] == 0.5
