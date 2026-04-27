#!/usr/bin/env python3
"""Export geometry-distance trajectory artifacts from validation embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_bundle_exports import load_validation_run_payloads
from eval.validation_summary import load_json_file
from eval.validation_trajectory_geometry import build_validation_trajectory_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation trajectory geometry")
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
        help="Output JSON path",
    )
    args = parser.parse_args()

    validation_manifest = load_json_file(args.validation_manifest)
    run_payloads = load_validation_run_payloads(validation_manifest, include_embeddings=True)
    payload = build_validation_trajectory_geometry(validation_manifest, run_payloads)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.validation_manifest).parent / "validation_trajectory_geometry.json"
    )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
