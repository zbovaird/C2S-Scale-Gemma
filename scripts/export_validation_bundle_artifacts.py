#!/usr/bin/env python3
"""Export the main summary, explorer, and trajectory artifacts for a validation bundle."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_bundle_exports import (
    export_validation_bundle_artifacts,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    artifact_paths = export_validation_bundle_artifacts(
        args.validation_manifest,
        output_dir=args.output_dir,
    )
    logger.info(
        "Saved validation artifact bundle to %s",
        Path(artifact_paths["artifact_manifest"]).parent,
    )


if __name__ == "__main__":
    main()
