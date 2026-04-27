import json

from eval.validation_dataset_inspector import (
    build_validation_dataset_inspection,
    summarize_column_values,
    write_validation_dataset_inspection,
)


class FakeObs:
    def __init__(self, columns):
        self._columns = columns
        self.columns = list(columns)

    def __getitem__(self, key):
        return self._columns[key]


class FakeAnnData:
    def __init__(self):
        self.obs = FakeObs(
            {
                "cell_type": ["fibroblast", "fibroblast", "iPSC"],
                "timepoint": ["D0", "D2", "iPSC"],
            }
        )
        self.var_names = ["POU5F1", "SOX2", "KLF4", "ACTB"]
        self.n_obs = 3
        self.n_vars = 4


def test_summarize_column_values_returns_counts_and_values():
    summary = summarize_column_values(["D0", "D2", "D0"])

    assert summary["n_unique"] == 2
    assert summary["values"] == ["D0", "D2"]
    assert summary["top_counts"][0] == {"value": "D0", "count": 2}


def test_build_validation_dataset_inspection_reports_metadata_and_oskm():
    report = build_validation_dataset_inspection(
        FakeAnnData(),
        dataset_name="toy_oskm",
        species="human",
        cell_type_column="cell_type",
        timepoint_column="timepoint",
    )

    assert report["artifact_type"] == "validation_dataset_inspection"
    assert report["n_cells"] == 3
    assert report["cell_type_column_present"] is True
    assert report["timepoint_summary"]["values"] == ["D0", "D2", "iPSC"]
    assert report["resolved_oskm_genes"] == {
        "POU5F1": "POU5F1",
        "SOX2": "SOX2",
        "KLF4": "KLF4",
    }
    assert report["missing_oskm_genes"] == ["MYC"]
    assert report["ready_for_profile_review"] is True


def test_write_validation_dataset_inspection_writes_json(tmp_path):
    output_path = tmp_path / "inspection.json"

    write_validation_dataset_inspection(
        {"artifact_type": "validation_dataset_inspection"},
        output_path,
    )

    assert json.loads(output_path.read_text())["artifact_type"] == (
        "validation_dataset_inspection"
    )
