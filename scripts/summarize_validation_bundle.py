#!/usr/bin/env python3
"""Summarize a validation bundle into a compact benchmark report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_markdown import write_markdown_summary
from eval.validation_summary import (
    build_validation_benchmark_summary,
    load_json_file,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a validation bundle")
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

    ablation_manifest = load_json_file(validation_manifest["ablation_manifest_path"])
    run_payloads = []
    for run in ablation_manifest.get("runs", []):
        output_dir_path = Path(run["output_dir"])
        run_payloads.append(
            {
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
        )

    summary = build_validation_benchmark_summary(validation_manifest, run_payloads)
    summary_path = output_dir / "validation_benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    markdown_path = output_dir / "VALIDATION_BENCHMARK.md"
    write_markdown_summary(markdown_path, summary)

    logger.info("Saved validation benchmark summary to %s", summary_path)


if __name__ == "__main__":
    main()
