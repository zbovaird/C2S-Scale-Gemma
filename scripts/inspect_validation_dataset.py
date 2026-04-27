#!/usr/bin/env python
"""Inspect a downloaded validation AnnData file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.validation_dataset_inspector import (
    build_validation_dataset_inspection,
    write_validation_dataset_inspection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an .h5ad file.")
    parser.add_argument("--output", required=True, help="Path for the JSON report.")
    parser.add_argument("--dataset-name", required=True, help="Short dataset label.")
    parser.add_argument("--species", default="human", help="OSKM alias species.")
    parser.add_argument("--cell-type-column", default=None)
    parser.add_argument("--timepoint-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "scanpy is required to inspect .h5ad files. Install the project with "
            "single-cell dependencies before running this command."
        ) from exc

    adata = sc.read_h5ad(args.input)
    report = build_validation_dataset_inspection(
        adata,
        dataset_name=args.dataset_name,
        species=args.species,
        cell_type_column=args.cell_type_column,
        timepoint_column=args.timepoint_column,
    )
    write_validation_dataset_inspection(report, args.output)


if __name__ == "__main__":
    main()
