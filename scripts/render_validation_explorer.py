#!/usr/bin/env python3
"""Render a self-contained HTML explorer for validation bundles."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_explorer import build_validation_explorer_payload
from eval.validation_explorer_html import render_validation_explorer_html
from eval.validation_summary import load_json_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render validation explorer HTML")
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Path to validation_benchmark_summary.json",
    )
    parser.add_argument(
        "--payload-path",
        type=str,
        default=None,
        help="Path to validation_explorer_payload.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output HTML path (defaults next to the chosen input)",
    )
    args = parser.parse_args()

    if not args.summary_path and not args.payload_path:
        raise SystemExit("Provide either --summary-path or --payload-path.")

    if args.payload_path:
        input_path = Path(args.payload_path)
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        input_path = Path(args.summary_path)
        summary = load_json_file(input_path)
        payload = build_validation_explorer_payload(summary)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else input_path.parent / "validation_explorer.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_validation_explorer_html(payload), encoding="utf-8")
    logger.info("Saved validation explorer HTML to %s", output_path)


if __name__ == "__main__":
    main()
