#!/usr/bin/env python3
"""Build a review protocol manifest for a validation bundle."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_review_protocol import (
    build_validation_review_protocol,
    write_validation_review_protocol,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation review protocol")
    parser.add_argument("--track", type=str, required=True, help="Validation track name")
    parser.add_argument(
        "--validation-manifest",
        type=str,
        required=True,
        help="Path to validation_bundle.json",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Validation bundle output root",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSON path (defaults to output root)",
    )
    args = parser.parse_args()

    protocol = build_validation_review_protocol(
        track_name=args.track,
        validation_manifest_path=args.validation_manifest,
        output_root=args.output_root,
    )
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(args.output_root) / "validation_review_protocol.json"
    )
    write_validation_review_protocol(protocol, output_path)
    logger.info("Saved validation review protocol to %s", output_path)


if __name__ == "__main__":
    main()
