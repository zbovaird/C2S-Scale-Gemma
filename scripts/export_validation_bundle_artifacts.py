#!/usr/bin/env python3
"""Export the main summary, explorer, and trajectory artifacts for a validation bundle."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_bundle_exports import build_validation_artifact_paths
from eval.validation_explorer import build_validation_explorer_payload
from eval.validation_explorer_html import render_validation_explorer_html
from eval.validation_summary import (
    build_validation_benchmark_summary,
    load_json_file,
)
from eval.validation_trajectory_dataset import build_validation_trajectory_dataset
from eval.validation_trajectory_projection import build_validation_trajectory_projection
from eval.validation_trajectory_projection_html import (
    render_validation_trajectory_projection_html,
)
from summarize_validation_bundle import write_markdown_summary


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_run_payloads(validation_manifest: dict, include_embeddings: bool = False) -> list[dict]:
    ablation_manifest = load_json_file(validation_manifest["ablation_manifest_path"])
    run_payloads = []
    for run in ablation_manifest.get("runs", []):
        output_dir_path = Path(run["output_dir"])
        payload = {
            "label": run.get("label"),
            "alignment_mode": run.get("alignment_mode"),
            "dataset_profile": run.get("dataset_profile"),
            "embedding_summary": load_json_file(output_dir_path / "embedding_shift_summary.json"),
            "overlay_summary": load_json_file(output_dir_path / "reprogramming_overlay_summary.json"),
            "fused_shift_rows": load_json_file(output_dir_path / "fused_embedding_shift_frame.json"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation bundle artifacts")
    parser.add_argument(
        "--validation-manifest",
        type=str,
        required=True,
        help="Path to validation_bundle.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to the validation bundle directory)",
    )
    args = parser.parse_args()

    validation_manifest_path = Path(args.validation_manifest)
    validation_manifest = load_json_file(validation_manifest_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else validation_manifest_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_validation_artifact_paths(output_dir)

    summary_run_payloads = _load_run_payloads(validation_manifest, include_embeddings=False)
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

    trajectory_run_payloads = _load_run_payloads(validation_manifest, include_embeddings=True)
    trajectory_dataset = build_validation_trajectory_dataset(
        validation_manifest,
        trajectory_run_payloads,
    )
    Path(artifact_paths["trajectory_dataset"]).write_text(
        json.dumps(trajectory_dataset, indent=2),
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
    logger.info("Saved validation artifact bundle to %s", output_dir)


if __name__ == "__main__":
    main()
