"""Helpers for exporting a full validation artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.validation_explorer import build_validation_explorer_payload
from eval.validation_explorer_html import render_validation_explorer_html
from eval.validation_markdown import write_markdown_summary
from eval.validation_summary import build_validation_benchmark_summary, load_json_file
from eval.validation_trajectory_dataset import build_validation_trajectory_dataset
from eval.validation_trajectory_geometry import build_validation_trajectory_geometry
from eval.validation_trajectory_projection import build_validation_trajectory_projection
from eval.validation_trajectory_projection_html import (
    render_validation_trajectory_projection_html,
)


def build_validation_artifact_paths(output_dir: str | Path) -> dict:
    """Return the default output paths for bundle-level validation artifacts."""
    root = Path(output_dir)
    return {
        "summary_json": str(root / "validation_benchmark_summary.json"),
        "summary_markdown": str(root / "VALIDATION_BENCHMARK.md"),
        "explorer_payload": str(root / "validation_explorer_payload.json"),
        "explorer_html": str(root / "validation_explorer.html"),
        "trajectory_dataset": str(root / "validation_trajectory_dataset.json"),
        "trajectory_geometry": str(root / "validation_trajectory_geometry.json"),
        "trajectory_projection": str(root / "validation_trajectory_projection.json"),
        "trajectory_projection_html": str(root / "validation_trajectory_projection.html"),
        "artifact_manifest": str(root / "validation_artifacts_manifest.json"),
    }


def load_validation_run_payloads(
    validation_manifest: dict,
    include_embeddings: bool = False,
) -> list[dict]:
    """Load run-level artifacts referenced by a validation manifest."""
    ablation_manifest = load_json_file(validation_manifest["ablation_manifest_path"])
    run_payloads = []
    for run in ablation_manifest.get("runs", []):
        output_dir_path = Path(run["output_dir"])
        payload = {
            "label": run.get("label"),
            "alignment_mode": run.get("alignment_mode"),
            "dataset_profile": run.get("dataset_profile"),
            "embedding_summary": load_json_file(
                output_dir_path / "embedding_shift_summary.json"
            ),
            "overlay_summary": load_json_file(
                output_dir_path / "reprogramming_overlay_summary.json"
            ),
            "fused_shift_rows": load_json_file(
                output_dir_path / "fused_embedding_shift_frame.json"
            ),
        }
        if include_embeddings:
            payload["baseline_fused_embeddings"] = np.load(
                output_dir_path / "baseline_fused_embeddings.npy"
            )
            payload["perturbed_fused_embeddings"] = np.load(
                output_dir_path / "perturbed_fused_embeddings.npy"
            )
        run_payloads.append(payload)
    return run_payloads


def export_validation_bundle_artifacts(
    validation_manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    """Export summary, explorer, and trajectory artifacts for a validation bundle."""
    manifest_path = Path(validation_manifest_path)
    validation_manifest = load_json_file(manifest_path)
    output_path = Path(output_dir) if output_dir is not None else manifest_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_validation_artifact_paths(output_path)

    summary_run_payloads = load_validation_run_payloads(
        validation_manifest,
        include_embeddings=False,
    )
    summary = build_validation_benchmark_summary(validation_manifest, summary_run_payloads)
    Path(artifact_paths["summary_json"]).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(Path(artifact_paths["summary_markdown"]), summary)

    explorer_payload = build_validation_explorer_payload(summary)
    Path(artifact_paths["explorer_payload"]).write_text(
        json.dumps(explorer_payload, indent=2),
        encoding="utf-8",
    )
    Path(artifact_paths["explorer_html"]).write_text(
        render_validation_explorer_html(explorer_payload),
        encoding="utf-8",
    )

    trajectory_run_payloads = load_validation_run_payloads(
        validation_manifest,
        include_embeddings=True,
    )
    trajectory_dataset = build_validation_trajectory_dataset(
        validation_manifest,
        trajectory_run_payloads,
    )
    Path(artifact_paths["trajectory_dataset"]).write_text(
        json.dumps(trajectory_dataset, indent=2),
        encoding="utf-8",
    )

    trajectory_geometry = build_validation_trajectory_geometry(
        validation_manifest,
        trajectory_run_payloads,
    )
    Path(artifact_paths["trajectory_geometry"]).write_text(
        json.dumps(trajectory_geometry, indent=2),
        encoding="utf-8",
    )

    trajectory_projection = build_validation_trajectory_projection(
        validation_manifest,
        trajectory_run_payloads,
    )
    Path(artifact_paths["trajectory_projection"]).write_text(
        json.dumps(trajectory_projection, indent=2),
        encoding="utf-8",
    )
    Path(artifact_paths["trajectory_projection_html"]).write_text(
        render_validation_trajectory_projection_html(trajectory_projection),
        encoding="utf-8",
    )

    Path(artifact_paths["artifact_manifest"]).write_text(
        json.dumps(artifact_paths, indent=2),
        encoding="utf-8",
    )
    return artifact_paths
