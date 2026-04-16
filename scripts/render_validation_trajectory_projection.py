#!/usr/bin/env python3
"""Render a self-contained HTML viewer for trajectory projections."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_summary import load_json_file
from eval.validation_trajectory_projection_html import (
    render_validation_trajectory_projection_html,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation trajectory projection HTML")
    parser.add_argument(
        "--projection-path",
        type=str,
        default=None,
        help="Path to validation_trajectory_projection.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output HTML path (defaults next to the projection file)",
    )
    args = parser.parse_args()

    if not args.projection_path:
        raise SystemExit("Provide --projection-path.")

    projection_path = Path(args.projection_path)
    projection = load_json_file(projection_path)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else projection_path.parent / "validation_trajectory_projection.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_validation_trajectory_projection_html(projection),
        encoding="utf-8",
    )
    logger.info("Saved validation trajectory projection HTML to %s", output_path)


if __name__ == "__main__":
    main()
