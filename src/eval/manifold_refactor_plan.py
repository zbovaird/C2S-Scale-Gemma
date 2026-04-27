"""Build staged refactor plans from manifold-readiness findings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STAGE_DEFINITIONS = (
    {
        "id": "encoder_projection_path",
        "title": "Encoder projection path",
        "path_contains": ("src/hgnn/encoder.py",),
        "success_criteria": "Input/output projections on the UHG encoder path are either manifold-native or explicitly documented tangent-space adapters.",
    },
    {
        "id": "hgnn_layer_transforms",
        "title": "HGNN layer transforms",
        "path_contains": ("src/hgnn/layers.py",),
        "success_criteria": "GraphSAGE, GIN, and attention transforms avoid untracked Euclidean operations for tensors intended to remain in manifold space.",
    },
    {
        "id": "alignment_geometry_path",
        "title": "Alignment geometry path",
        "path_contains": ("src/fusion/align_losses.py",),
        "success_criteria": "Projective/hyperbolic alignment modes use intended UHG distances, with Euclidean fallbacks explicit and tested.",
    },
    {
        "id": "trainer_embedding_selection",
        "title": "Trainer embedding selection",
        "path_contains": ("src/fusion/trainer.py",),
        "success_criteria": "Trainer selects graph embeddings that match the geometry expected by each alignment mode.",
    },
)


def _findings_for_stage(findings: list[dict], stage: Mapping[str, Any]) -> list[dict]:
    path_tokens = tuple(stage.get("path_contains", ()))
    return [
        finding
        for finding in findings
        if any(token in str(finding.get("path", "")) for token in path_tokens)
    ]


def build_manifold_refactor_plan(readiness_report: Mapping[str, Any]) -> dict:
    """Build an ordered implementation plan from manifold-readiness findings."""
    findings = list(readiness_report.get("findings", []))
    stages = []
    for index, stage in enumerate(STAGE_DEFINITIONS, start=1):
        stage_findings = _findings_for_stage(findings, stage)
        stages.append(
            {
                "order": index,
                "id": stage["id"],
                "title": stage["title"],
                "status": "pending" if stage_findings else "no_findings",
                "n_findings": len(stage_findings),
                "n_blockers": sum(
                    1 for finding in stage_findings if finding.get("severity") == "blocker"
                ),
                "n_warnings": sum(
                    1 for finding in stage_findings if finding.get("severity") == "warning"
                ),
                "success_criteria": stage["success_criteria"],
                "findings": stage_findings,
            }
        )
    return {
        "source_status": readiness_report.get("status"),
        "n_source_findings": readiness_report.get("n_findings", len(findings)),
        "stages": stages,
    }


def load_manifold_refactor_plan(readiness_report_path: str | Path) -> dict:
    """Load a readiness report and build a staged refactor plan."""
    with Path(readiness_report_path).open("r", encoding="utf-8") as handle:
        readiness_report = json.load(handle)
    return build_manifold_refactor_plan(readiness_report)


def write_manifold_refactor_plan(plan: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a staged manifold refactor plan as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
