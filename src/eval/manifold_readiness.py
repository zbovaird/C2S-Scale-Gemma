"""Static manifold-readiness audits for the UHG validation path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MANIFOLD_AUDIT_TARGETS = (
    {
        "path": "src/hgnn/encoder.py",
        "patterns": (r"nn\.Linear",),
        "severity": "blocker",
        "recommendation": "Replace Euclidean projections on the UHG encoder path with manifold-native maps or explicit tangent-space adapters.",
    },
    {
        "path": "src/hgnn/layers.py",
        "patterns": (r"nn\.Linear",),
        "severity": "blocker",
        "recommendation": "Audit UHG layer transforms and replace Euclidean linear maps where tensors are intended to stay in manifold space.",
    },
    {
        "path": "src/fusion/align_losses.py",
        "patterns": (r"torch\.cdist", r"graph_to_geometry = nn\.Linear"),
        "severity": "warning",
        "recommendation": "Keep Euclidean fallbacks explicit and verify projective alignment runs use the intended UHG distance path.",
    },
    {
        "path": "src/fusion/trainer.py",
        "patterns": (r"euclidean_embeddings",),
        "severity": "warning",
        "recommendation": "Confirm graph embeddings used for alignment remain in the geometry expected by the selected alignment loss.",
    },
)


def audit_file_for_patterns(
    *,
    repo_root: str | Path,
    target: Mapping[str, Any],
) -> list[dict]:
    """Audit one source file for configured manifold-readiness patterns."""
    path = Path(repo_root) / str(target["path"])
    if not path.exists():
        return [
            {
                "path": str(target["path"]),
                "line": None,
                "pattern": None,
                "severity": "blocker",
                "message": "Audit target file is missing.",
                "recommendation": target.get("recommendation"),
            }
        ]
    findings = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        for pattern in target.get("patterns", []):
            if re.search(str(pattern), line):
                findings.append(
                    {
                        "path": str(target["path"]),
                        "line": line_number,
                        "pattern": pattern,
                        "severity": target.get("severity", "warning"),
                        "snippet": line.strip(),
                        "recommendation": target.get("recommendation"),
                    }
                )
    return findings


def build_manifold_readiness_report(
    *,
    repo_root: str | Path,
    targets: Sequence[Mapping[str, Any]] = DEFAULT_MANIFOLD_AUDIT_TARGETS,
) -> dict:
    """Build a static manifold-readiness report for geometry-path refactors."""
    findings = []
    for target in targets:
        findings.extend(audit_file_for_patterns(repo_root=repo_root, target=target))
    n_blockers = sum(1 for finding in findings if finding["severity"] == "blocker")
    n_warnings = sum(1 for finding in findings if finding["severity"] == "warning")
    return {
        "status": "needs_refactor" if n_blockers else "review",
        "n_findings": len(findings),
        "n_blockers": n_blockers,
        "n_warnings": n_warnings,
        "findings": findings,
    }


def write_manifold_readiness_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a manifold-readiness report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
