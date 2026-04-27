#!/usr/bin/env python3
"""Build a validation data acquisition manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_data_manifest import (
    load_validation_data_manifest,
    write_validation_data_manifest,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation data manifest")
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
        default="artifacts/validation_data_manifest.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    manifest = load_validation_data_manifest(
        track_config_path=args.track_config,
        profile_config_path=args.profile_config,
    )
    write_validation_data_manifest(manifest, args.output_path)
    logger.info("Saved validation data manifest to %s", args.output_path)


if __name__ == "__main__":
    main()
