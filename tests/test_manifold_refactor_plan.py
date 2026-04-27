import json

from eval.manifold_refactor_plan import (
    build_manifold_refactor_plan,
    load_manifold_refactor_plan,
    write_manifold_refactor_plan,
)


def test_build_manifold_refactor_plan_groups_findings_into_ordered_stages():
    plan = build_manifold_refactor_plan(
        {
            "status": "needs_refactor",
            "n_findings": 2,
            "findings": [
                {
                    "path": "src/hgnn/encoder.py",
                    "severity": "blocker",
                    "line": 70,
                },
                {
                    "path": "src/fusion/align_losses.py",
                    "severity": "warning",
                    "line": 183,
                },
            ],
        }
    )

    assert plan["source_status"] == "needs_refactor"
    assert plan["stages"][0]["id"] == "encoder_projection_path"
    assert plan["stages"][0]["n_blockers"] == 1
    assert plan["stages"][2]["id"] == "alignment_geometry_path"
    assert plan["stages"][2]["n_warnings"] == 1


def test_load_and_write_manifold_refactor_plan_round_trip(tmp_path):
    readiness_path = tmp_path / "manifold_readiness.json"
    output_path = tmp_path / "manifold_refactor_plan.json"
    readiness_path.write_text(
        json.dumps(
            {
                "status": "review",
                "findings": [
                    {
                        "path": "src/fusion/trainer.py",
                        "severity": "warning",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = load_manifold_refactor_plan(readiness_path)
    write_manifold_refactor_plan(plan, output_path)

    saved = json.loads(output_path.read_text())
    assert saved["stages"][-1]["id"] == "trainer_embedding_selection"
    assert saved["stages"][-1]["n_warnings"] == 1
