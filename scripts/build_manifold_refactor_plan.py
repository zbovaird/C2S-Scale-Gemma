#!/usr/bin/env python3
"""Build a staged manifold refactor plan from readiness findings."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.manifold_refactor_plan import (
    load_manifold_refactor_plan,
    write_manifold_refactor_plan,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifold refactor plan")
    parser.add_argument(
        "--readiness-report",
        type=str,
        required=True,
        help="Path to manifold_readiness.json",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/manifold_refactor_plan.json",
        help="Path to write the staged refactor plan",
    )
    args = parser.parse_args()

    plan = load_manifold_refactor_plan(args.readiness_report)
    write_manifold_refactor_plan(plan, args.output_path)
    logger.info("Saved manifold refactor plan to %s", args.output_path)


if __name__ == "__main__":
    main()
