import json

import numpy as np

from eval.regulatory_screening_report import (
    build_regulatory_screening_report,
    build_regulatory_candidate_selection,
    load_regulatory_screening_report,
    select_regulatory_candidate_rows,
    write_regulatory_candidate_selection,
    write_regulatory_screening_report,
)


def test_build_regulatory_screening_report_includes_summary_and_rows():
    report = build_regulatory_screening_report(
        expression_matrix=np.array([[5.0, 5.0], [0.0, 1.0]]),
        var_names=["POU5F1", "SOX2"],
        cell_ids=["cell_a", "cell_b"],
        cell_types=["fibroblast", "pbmc"],
        dataset_name="candidate.h5ad",
        panels={"oskm_core": ["POU5F1", "SOX2"]},
        high_signal_quantile=0.5,
    )

    assert report["artifact_type"] == "regulatory_screening_report"
    assert report["dataset_name"] == "candidate.h5ad"
    assert report["summary"]["n_cells"] == 2
    assert report["rows"][0]["cell_id"] == "cell_a"
    assert report["rows"][0]["high_regulatory_signal"] is True


def test_write_regulatory_screening_report_writes_json(tmp_path):
    output_path = tmp_path / "regulatory_screening_report.json"
    write_regulatory_screening_report(
        {"artifact_type": "regulatory_screening_report"},
        output_path,
    )

    assert json.loads(output_path.read_text())["artifact_type"] == "regulatory_screening_report"


def test_select_regulatory_candidate_rows_filters_and_sorts_candidates():
    rows = [
        {
            "cell_index": 0,
            "cell_id": "low",
            "cell_type": "pbmc",
            "regulatory_screen_score": 1.0,
            "high_regulatory_signal": False,
        },
        {
            "cell_index": 1,
            "cell_id": "mid",
            "cell_type": "fibroblast",
            "regulatory_screen_score": 5.0,
            "high_regulatory_signal": True,
        },
        {
            "cell_index": 2,
            "cell_id": "high",
            "cell_type": "pbmc",
            "regulatory_screen_score": 7.0,
            "high_regulatory_signal": True,
        },
    ]

    selected = select_regulatory_candidate_rows(
        rows,
        top_n=1,
        cell_types=["pbmc"],
    )

    assert [row["cell_id"] for row in selected] == ["high"]


def test_build_regulatory_candidate_selection_reports_selected_cells():
    report = {
        "dataset_name": "candidate.h5ad",
        "high_signal_quantile": 0.75,
        "rows": [
            {
                "cell_index": 0,
                "cell_id": "cell_a",
                "cell_type": "pbmc",
                "regulatory_screen_score": 2.0,
                "high_regulatory_signal": True,
            },
            {
                "cell_index": 1,
                "cell_id": "cell_b",
                "cell_type": "pbmc",
                "regulatory_screen_score": 0.5,
                "high_regulatory_signal": False,
            },
        ],
    }

    selection = build_regulatory_candidate_selection(report, top_n=10)

    assert selection["artifact_type"] == "regulatory_candidate_selection"
    assert selection["summary"]["n_selected_cells"] == 1
    assert selection["selected_cell_ids"] == ["cell_a"]
    assert selection["selected_cell_indices"] == [0]


def test_regulatory_selection_io_round_trip(tmp_path):
    report_path = tmp_path / "screening.json"
    output_path = tmp_path / "selection.json"
    report = {"artifact_type": "regulatory_screening_report", "rows": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded_report = load_regulatory_screening_report(report_path)
    selection = build_regulatory_candidate_selection(loaded_report)
    write_regulatory_candidate_selection(selection, output_path)

    assert json.loads(output_path.read_text())["artifact_type"] == (
        "regulatory_candidate_selection"
    )
