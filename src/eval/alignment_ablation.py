"""Helpers for paired Euclidean vs projective alignment ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def build_paired_alignment_runs(
    *,
    output_root: str | Path,
    euclidean_config: str,
    euclidean_checkpoint: str,
    projective_config: str,
    projective_checkpoint: str,
) -> list[dict]:
    """Create paired run specs for Euclidean and projective alignment checkpoints."""
    root = Path(output_root)
    return [
        {
            "label": "euclidean",
            "config_path": euclidean_config,
            "checkpoint_path": euclidean_checkpoint,
            "output_dir": str(root / "euclidean"),
        },
        {
            "label": "projective",
            "config_path": projective_config,
            "checkpoint_path": projective_checkpoint,
            "output_dir": str(root / "projective"),
        },
    ]


def build_ablation_manifest(
    *,
    output_root: str | Path,
    baseline_data_path: str,
    perturbed_data_path: str,
    dataset_profile: str | None,
    run_results: Sequence[dict],
) -> dict:
    """Build a lightweight manifest describing a paired ablation run."""
    return {
        "output_root": str(Path(output_root)),
        "baseline_data_path": baseline_data_path,
        "perturbed_data_path": perturbed_data_path,
        "dataset_profile": dataset_profile,
        "runs": list(run_results),
    }


def load_ablation_dirs_from_manifest(
    manifest_path: str | Path,
    comparison_dir: str | Path,
) -> list[Path]:
    """Load additional ablation output directories from a manifest."""
    manifest_file = Path(manifest_path)
    comparison_dir_resolved = Path(comparison_dir).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    ablation_dirs = []
    for run in manifest.get("runs", []):
        output_dir = run.get("output_dir")
        if not output_dir:
            continue
        candidate = Path(output_dir)
        if candidate.resolve() == comparison_dir_resolved:
            continue
        ablation_dirs.append(candidate)
    return ablation_dirs
