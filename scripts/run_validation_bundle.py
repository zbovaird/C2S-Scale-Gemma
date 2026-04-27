#!/usr/bin/env python3
"""Run a named validation track with paired alignment ablations."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.alignment_ablation import build_ablation_manifest, build_paired_alignment_runs
from eval.validation_tracks import (
    build_validation_bundle_manifest,
    resolve_validation_track,
)
from eval.validation_preflight import build_validation_input_preflight
from compare_oskm_perturbation_embeddings import run_embedding_comparison


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named OKSM validation bundle")
    parser.add_argument("--track", type=str, required=True, help="Validation track name")
    parser.add_argument("--baseline-data-path", type=str, required=True, help="Baseline .h5ad path")
    parser.add_argument("--perturbed-data-path", type=str, required=True, help="Perturbed .h5ad path")
    parser.add_argument(
        "--track-config",
        type=str,
        default="configs/validation_tracks.toml",
        help="Path to the validation track registry",
    )
    parser.add_argument(
        "--dataset-profile-config",
        type=str,
        default="configs/reprogramming_profiles.toml",
        help="Path to the dataset profile registry",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/validation_bundle",
        help="Root output directory for the validation bundle",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding extraction batch size")
    parser.add_argument("--euclidean-config", type=str, required=True, help="Euclidean run config")
    parser.add_argument("--euclidean-checkpoint", type=str, required=True, help="Euclidean run checkpoint")
    parser.add_argument("--projective-config", type=str, required=True, help="Projective run config")
    parser.add_argument("--projective-checkpoint", type=str, required=True, help="Projective run checkpoint")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run input checks and exit before model execution",
    )
    args = parser.parse_args()

    track = resolve_validation_track(args.track, args.track_config)
    preflight_report = build_validation_input_preflight(
        track_name=args.track,
        track_config=track,
        baseline_data_path=args.baseline_data_path,
        perturbed_data_path=args.perturbed_data_path,
        euclidean_config=args.euclidean_config,
        euclidean_checkpoint=args.euclidean_checkpoint,
        projective_config=args.projective_config,
        projective_checkpoint=args.projective_checkpoint,
        dataset_profile_config=args.dataset_profile_config,
    )
    if preflight_report["status"] != "pass":
        failed_checks = [
            check["id"] for check in preflight_report["checks"] if not check["passed"]
        ]
        raise SystemExit(f"Validation preflight failed: {', '.join(failed_checks)}")
    if args.preflight_only:
        logger.info("Validation preflight passed for track %s", args.track)
        return

    dataset_profile = track.get("dataset_profile")
    output_root = Path(args.output_root) / args.track
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
        logger.info("Running %s validation comparison for track %s", run_spec["label"], args.track)
        result = run_embedding_comparison(
            config_path=run_spec["config_path"],
            checkpoint_path=run_spec["checkpoint_path"],
            baseline_data_path=args.baseline_data_path,
            perturbed_data_path=args.perturbed_data_path,
            output_dir=run_spec["output_dir"],
            batch_size=args.batch_size,
            dataset_profile=dataset_profile,
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

    ablation_manifest = build_ablation_manifest(
        output_root=output_root,
        baseline_data_path=args.baseline_data_path,
        perturbed_data_path=args.perturbed_data_path,
        dataset_profile=dataset_profile,
        run_results=run_results,
    )
    ablation_manifest_path = output_root / "ablation_manifest.json"
    with ablation_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(ablation_manifest, handle, indent=2)

    validation_manifest = build_validation_bundle_manifest(
        track_name=args.track,
        track_config=track,
        output_root=output_root,
        baseline_data_path=args.baseline_data_path,
        perturbed_data_path=args.perturbed_data_path,
        ablation_manifest_path=ablation_manifest_path,
    )
    validation_manifest_path = output_root / "validation_bundle.json"
    with validation_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(validation_manifest, handle, indent=2)

    logger.info("Saved validation bundle manifest to %s", validation_manifest_path)


if __name__ == "__main__":
    main()
