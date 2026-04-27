#!/usr/bin/env python3
"""Run QA checks against exported validation bundle artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_preflight import build_validation_artifact_qa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA exported validation artifacts")
    parser.add_argument(
        "--artifact-manifest",
        type=str,
        required=True,
        help="Path to validation_artifacts_manifest.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the QA report JSON",
    )
    args = parser.parse_args()

    report = build_validation_artifact_qa(args.artifact_manifest)
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Validation artifact QA %s (%s/%s failed)",
        report["status"],
        report["n_failed"],
        report["n_checks"],
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
