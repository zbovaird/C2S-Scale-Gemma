#!/usr/bin/env python3
"""Report readiness for configured validation tracks and datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_readiness import (
    load_validation_readiness_report,
    write_validation_readiness_report,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report validation dataset readiness")
    parser.add_argument(
        "--track-config",
        type=str,
        default="configs/validation_tracks.toml",
        help="Path to validation track registry",
    )
    parser.add_argument(
        "--profile-config",
        type=str,
        default="configs/reprogramming_profiles.toml",
        help="Path to reprogramming profile registry",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the readiness JSON report",
    )
    args = parser.parse_args()

    report = load_validation_readiness_report(
        track_config_path=args.track_config,
        profile_config_path=args.profile_config,
    )
    if args.output_path:
        write_validation_readiness_report(report, args.output_path)
    logger.info("Validation readiness: %s", json.dumps(report["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
