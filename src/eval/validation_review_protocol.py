"""Build ordered review protocols for real validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def build_validation_review_protocol(
    *,
    track_name: str,
    validation_manifest_path: str | Path,
    output_root: str | Path,
) -> dict:
    """Build an ordered protocol for reviewing a real validation bundle."""
    root = Path(output_root)
    validation_manifest = Path(validation_manifest_path)
    artifact_manifest = root / "validation_artifacts_manifest.json"
    return {
        "track_name": track_name,
        "validation_manifest_path": str(validation_manifest),
        "output_root": str(root),
        "review_status": "not_started",
        "steps": [
            {
                "id": "readiness",
                "title": "Dataset readiness",
                "required": True,
                "success_criteria": "Track metadata is complete and required local baseline/perturbed data are present.",
                "command": "uv run scripts/report_validation_readiness.py --output-path artifacts/validation_readiness.json",
            },
            {
                "id": "calibration",
                "title": "Threshold calibration audit",
                "required": True,
                "success_criteria": "Heuristic window and recommendation thresholds are present, bounded, and internally consistent.",
                "command": "uv run scripts/report_validation_calibration.py --output-path artifacts/validation_calibration.json",
            },
            {
                "id": "preflight",
                "title": "Validation input preflight",
                "required": True,
                "success_criteria": "Selected track, datasets, configs, checkpoints, and profile registry pass preflight checks.",
                "command": "uv run scripts/run_validation_bundle.py ... --preflight-only",
            },
            {
                "id": "run_bundle",
                "title": "Run validation bundle",
                "required": True,
                "success_criteria": "Euclidean and projective runs complete and write validation_bundle.json.",
                "command": "uv run scripts/run_validation_bundle.py ...",
            },
            {
                "id": "export_artifacts",
                "title": "Export validation artifacts",
                "required": True,
                "success_criteria": "Benchmark, explorer, trajectory dataset, projection, and HTML artifacts are exported.",
                "command": f"uv run scripts/export_validation_bundle_artifacts.py --validation-manifest {validation_manifest}",
            },
            {
                "id": "artifact_qa",
                "title": "Artifact QA",
                "required": True,
                "success_criteria": "All required exported artifacts exist and JSON artifacts parse successfully.",
                "command": f"uv run scripts/qa_validation_artifacts.py --artifact-manifest {artifact_manifest}",
            },
            {
                "id": "interpretation_review",
                "title": "Interpretation review",
                "required": True,
                "success_criteria": "Reported recommendations are reviewed with interpretation-limit notes and not treated as biological safety claims.",
                "artifacts": [
                    str(root / "VALIDATION_BENCHMARK.md"),
                    str(root / "validation_explorer.html"),
                    str(root / "validation_trajectory_projection.html"),
                ],
            },
        ],
    }


def write_validation_review_protocol(protocol: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a validation review protocol manifest as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
