#!/usr/bin/env python3
"""Export a structured payload for validation-bundle exploration."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_explorer import build_validation_explorer_payload
from eval.validation_summary import load_json_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validation explorer payload")
    parser.add_argument(
        "--summary-path",
        type=str,
        required=True,
        help="Path to validation_benchmark_summary.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output JSON path (defaults next to the summary)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    summary = load_json_file(summary_path)
    payload = build_validation_explorer_payload(summary)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else summary_path.parent / "validation_explorer_payload.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved validation explorer payload to %s", output_path)


if __name__ == "__main__":
    main()
