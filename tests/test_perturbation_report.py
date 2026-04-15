from eval.perturbation_report import (
    compute_shift_histogram,
    get_top_shift_rows,
    summarize_shift_by_category,
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
