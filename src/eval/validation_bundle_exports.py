"""Helpers for exporting a full validation artifact bundle."""

from __future__ import annotations

from pathlib import Path


def build_validation_artifact_paths(output_dir: str | Path) -> dict:
    """Return the default output paths for bundle-level validation artifacts."""
    root = Path(output_dir)
    return {
        "summary_json": str(root / "validation_benchmark_summary.json"),
        "summary_markdown": str(root / "VALIDATION_BENCHMARK.md"),
        "explorer_payload": str(root / "validation_explorer_payload.json"),
        "explorer_html": str(root / "validation_explorer.html"),
        "trajectory_dataset": str(root / "validation_trajectory_dataset.json"),
        "trajectory_projection": str(root / "validation_trajectory_projection.json"),
        "trajectory_projection_html": str(root / "validation_trajectory_projection.html"),
        "artifact_manifest": str(root / "validation_artifacts_manifest.json"),
    }
