#!/usr/bin/env python3
"""Run paired Euclidean vs projective embedding comparisons."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.alignment_ablation import build_ablation_manifest, build_paired_alignment_runs
from compare_oskm_perturbation_embeddings import run_embedding_comparison


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired alignment ablations")
    parser.add_argument("--baseline-data-path", type=str, required=True, help="Baseline .h5ad path")
    parser.add_argument("--perturbed-data-path", type=str, required=True, help="Perturbed .h5ad path")
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/alignment_ablation",
        help="Root output directory for paired comparisons",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding extraction batch size")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default=None,
        help="Optional named dataset profile from configs/reprogramming_profiles.toml",
    )
    parser.add_argument(
        "--dataset-profile-config",
        type=str,
        default="configs/reprogramming_profiles.toml",
        help="Path to the dataset profile registry",
    )
    parser.add_argument("--euclidean-config", type=str, required=True, help="Euclidean run config")
    parser.add_argument("--euclidean-checkpoint", type=str, required=True, help="Euclidean run checkpoint")
    parser.add_argument("--projective-config", type=str, required=True, help="Projective run config")
    parser.add_argument("--projective-checkpoint", type=str, required=True, help="Projective run checkpoint")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_specs = build_paired_alignment_runs(
        output_root=output_root,
        euclidean_config=args.euclidean_config,
        euclidean_checkpoint=args.euclidean_checkpoint,
        projective_config=args.projective_config,
        projective_checkpoint=args.projective_checkpoint,
    )

    run_results = []
    for run_spec in run_specs:
        logger.info("Running %s alignment comparison", run_spec["label"])
        result = run_embedding_comparison(
            config_path=run_spec["config_path"],
            checkpoint_path=run_spec["checkpoint_path"],
            baseline_data_path=args.baseline_data_path,
            perturbed_data_path=args.perturbed_data_path,
            output_dir=run_spec["output_dir"],
            batch_size=args.batch_size,
            dataset_profile=args.dataset_profile,
            dataset_profile_config=args.dataset_profile_config,
        )
        run_results.append(
            {
                "label": run_spec["label"],
                "config_path": run_spec["config_path"],
                "checkpoint_path": run_spec["checkpoint_path"],
                "output_dir": result["output_dir"],
                "alignment_mode": result["overlay_summary"].get("alignment", {}).get(
                    "alignment_mode",
                    "unknown",
                ),
                "dataset_profile": result["overlay_summary"].get("dataset_profile"),
            }
        )

    manifest = build_ablation_manifest(
        output_root=output_root,
        baseline_data_path=args.baseline_data_path,
        perturbed_data_path=args.perturbed_data_path,
        dataset_profile=args.dataset_profile,
        run_results=run_results,
    )
    manifest_path = output_root / "ablation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info("Saved paired alignment ablation manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
