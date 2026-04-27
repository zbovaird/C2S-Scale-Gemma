#!/usr/bin/env python3
"""Report static manifold-readiness findings for the geometry path."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.manifold_readiness import (
    build_manifold_readiness_report,
    write_manifold_readiness_report,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report manifold-readiness findings")
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root to audit",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the audit JSON report",
    )
    args = parser.parse_args()

    report = build_manifold_readiness_report(repo_root=args.repo_root)
    if args.output_path:
        write_manifold_readiness_report(report, args.output_path)
    logger.info(
        "Manifold readiness: %s",
        json.dumps(
            {
                "status": report["status"],
                "n_blockers": report["n_blockers"],
                "n_warnings": report["n_warnings"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
