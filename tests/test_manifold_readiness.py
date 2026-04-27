import json

from eval.manifold_readiness import (
    audit_file_for_patterns,
    build_manifold_readiness_report,
    write_manifold_readiness_report,
)


def test_audit_file_for_patterns_reports_matching_lines(tmp_path):
    target_file = tmp_path / "src" / "hgnn" / "encoder.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("layer = nn.Linear(1, 2)\n", encoding="utf-8")

    findings = audit_file_for_patterns(
        repo_root=tmp_path,
        target={
            "path": "src/hgnn/encoder.py",
            "patterns": (r"nn\.Linear",),
            "severity": "blocker",
            "recommendation": "Use manifold-native maps.",
        },
    )

    assert findings[0]["line"] == 1
    assert findings[0]["severity"] == "blocker"


def test_build_manifold_readiness_report_counts_blockers_and_warnings(tmp_path):
    encoder = tmp_path / "encoder.py"
    align = tmp_path / "align_losses.py"
    encoder.write_text("layer = nn.Linear(1, 2)\n", encoding="utf-8")
    align.write_text("distances = torch.cdist(x, y)\n", encoding="utf-8")

    report = build_manifold_readiness_report(
        repo_root=tmp_path,
        targets=[
            {
                "path": "encoder.py",
                "patterns": (r"nn\.Linear",),
                "severity": "blocker",
            },
            {
                "path": "align_losses.py",
                "patterns": (r"torch\.cdist",),
                "severity": "warning",
            },
        ],
    )

    assert report["status"] == "needs_refactor"
    assert report["n_blockers"] == 1
    assert report["n_warnings"] == 1


def test_write_manifold_readiness_report_writes_json(tmp_path):
    output_path = tmp_path / "manifold_readiness.json"
    write_manifold_readiness_report({"status": "review"}, output_path)

    assert json.loads(output_path.read_text())["status"] == "review"
