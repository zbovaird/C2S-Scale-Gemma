import json

from eval.validation_review_protocol import (
    build_validation_review_protocol,
    write_validation_review_protocol,
)


def test_build_validation_review_protocol_orders_required_gates():
    protocol = build_validation_review_protocol(
        track_name="human_fibroblast_oskm",
        validation_manifest_path="artifacts/validation_bundle/human_fibroblast_oskm/validation_bundle.json",
        output_root="artifacts/validation_bundle/human_fibroblast_oskm",
    )

    assert protocol["track_name"] == "human_fibroblast_oskm"
    assert [step["id"] for step in protocol["steps"]][:3] == [
        "readiness",
        "calibration",
        "preflight",
    ]
    assert protocol["steps"][-1]["id"] == "interpretation_review"
    assert all(step["required"] for step in protocol["steps"])


def test_write_validation_review_protocol_writes_json(tmp_path):
    output_path = tmp_path / "validation_review_protocol.json"
    write_validation_review_protocol({"track_name": "human_fibroblast_oskm"}, output_path)

    assert json.loads(output_path.read_text())["track_name"] == "human_fibroblast_oskm"
