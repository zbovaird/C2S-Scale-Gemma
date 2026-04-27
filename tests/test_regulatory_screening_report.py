import json

import numpy as np

from eval.regulatory_screening_report import (
    build_regulatory_screening_report,
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
