#!/usr/bin/env python3
"""Export a cell-level trajectory dataset from a validation bundle."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_summary import load_json_file
from eval.validation_trajectory_dataset import build_validation_trajectory_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_run_payloads(validation_manifest: dict) -> list[dict]:
    ablation_manifest = load_json_file(validation_manifest["ablation_manifest_path"])
    run_payloads = []
    for run in ablation_manifest.get("runs", []):
        output_dir_path = Path(run["output_dir"])
        run_payloads.append(
            {
                "label": run.get("label"),
                "alignment_mode": run.get("alignment_mode"),
                "dataset_profile": run.get("dataset_profile"),
                "fused_shift_rows": load_json_file(
                    output_dir_path / "fused_embedding_shift_frame.json"
                ),
            }
        )
    return run_payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation trajectory dataset")
    parser.add_argument(
        "--validation-manifest",
        type=str,
        required=True,
        help="Path to validation_bundle.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSON path (defaults next to the validation manifest)",
    )
    args = parser.parse_args()

    validation_manifest_path = Path(args.validation_manifest)
    validation_manifest = load_json_file(validation_manifest_path)
    trajectory_dataset = build_validation_trajectory_dataset(
        validation_manifest,
        _load_run_payloads(validation_manifest),
    )
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else validation_manifest_path.parent / "validation_trajectory_dataset.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(trajectory_dataset, handle, indent=2)

    logger.info("Saved validation trajectory dataset to %s", output_path)


if __name__ == "__main__":
    main()
