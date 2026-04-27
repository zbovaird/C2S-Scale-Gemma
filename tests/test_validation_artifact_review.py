import json

from eval.validation_artifact_review import (
    build_validation_artifact_review,
    write_validation_artifact_review,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_artifacts(tmp_path, *, summary):
    files = {
        "summary_json": tmp_path / "validation_benchmark_summary.json",
        "summary_markdown": tmp_path / "validation_benchmark_summary.md",
        "explorer_payload": tmp_path / "validation_explorer_payload.json",
        "explorer_html": tmp_path / "validation_explorer.html",
        "trajectory_dataset": tmp_path / "validation_trajectory_dataset.json",
        "trajectory_geometry": tmp_path / "validation_trajectory_geometry.json",
        "trajectory_projection": tmp_path / "validation_trajectory_projection.json",
        "trajectory_projection_html": tmp_path / "validation_trajectory_projection.html",
    }
    _write_json(files["summary_json"], summary)
    files["summary_markdown"].write_text("# Summary\n", encoding="utf-8")
    files["explorer_html"].write_text("<html></html>\n", encoding="utf-8")
    files["trajectory_projection_html"].write_text("<html></html>\n", encoding="utf-8")
    for key in (
        "explorer_payload",
        "trajectory_dataset",
        "trajectory_geometry",
        "trajectory_projection",
    ):
        _write_json(files[key], {"artifact": key})

    manifest_path = tmp_path / "validation_artifact_manifest.json"
    _write_json(manifest_path, {key: str(path) for key, path in files.items()})
    return manifest_path


def test_build_validation_artifact_review_passes_clean_export(tmp_path):
    manifest_path = _write_artifacts(
        tmp_path,
        summary={
            "runs": [
                {
                    "label": "projective",
                    "geometry_fallback_used": False,
                    "risk_fraction": 0.0,
                }
            ],
            "interpretation_limits": ["Representation-level evidence only."],
            "recommendation": {
                "status": "pass",
                "preferred_alignment": "projective",
            },
            "trajectory_geometry_summary": [
                {"label": "projective", "mean_geometry_distance": 0.4}
            ],
        },
    )

    report = build_validation_artifact_review(manifest_path)

    assert report["status"] == "pass"
    assert report["qa_status"] == "pass"
    assert report["n_review"] == 0


def test_build_validation_artifact_review_flags_items_for_scientific_review(tmp_path):
    manifest_path = _write_artifacts(
        tmp_path,
        summary={
            "runs": [
                {
                    "label": "euclidean",
                    "geometry_fallback_used": True,
                    "risk_fraction": 0.25,
                }
            ],
            "interpretation_limits": [],
            "recommendation": {"status": "mixed"},
            "trajectory_geometry_summary": [],
        },
    )

    report = build_validation_artifact_review(manifest_path)

    assert report["status"] == "fail"
    items = {item["id"]: item for item in report["review_items"]}
    assert items["interpretation_limits"]["status"] == "fail"
    assert items["geometry_fallbacks"]["status"] == "review"
    assert items["risk_summary"]["details"]["max_risk_fraction"] == 0.25


def test_write_validation_artifact_review(tmp_path):
    report = {"status": "pass", "review_items": []}
    output_path = tmp_path / "reports" / "review.json"

    write_validation_artifact_review(report, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
