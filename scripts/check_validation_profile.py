#!/usr/bin/env python3
"""Check a dataset inspection report against a validation track."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_profile_check import (
    load_validation_profile_check,
    write_validation_profile_check,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-path", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument(
        "--track-config",
        default="configs/validation_tracks.toml",
        help="Path to validation track registry",
    )
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    report = load_validation_profile_check(
        inspection_path=args.inspection_path,
        track_name=args.track,
        track_config_path=args.track_config,
    )
    write_validation_profile_check(report, args.output_path)
    logger.info("Validation profile check status: %s", report["status"])


if __name__ == "__main__":
    main()
