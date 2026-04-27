#!/usr/bin/env python3
"""Select OSKM-adjacent candidate cells from a regulatory screening report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.regulatory_screening_report import (
    build_regulatory_candidate_selection,
    load_regulatory_screening_report,
    write_regulatory_candidate_selection,
)


def _write_h5ad_subset(data_path: str, output_h5ad: str, selected_indices: list[int]) -> None:
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - dependency is project-level optional at runtime
        raise SystemExit("scanpy is required to write an .h5ad candidate subset.") from exc

    adata = sc.read_h5ad(data_path)
    subset = adata[selected_indices].copy()
    Path(output_h5ad).parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(output_h5ad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select candidate cells with OSKM-adjacent regulatory activity."
    )
    parser.add_argument("--screening-report", required=True, help="Input regulatory screening JSON")
    parser.add_argument(
        "--output-path",
        default="artifacts/regulatory_candidate_selection.json",
        help="Output selection JSON path",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Maximum cells to select")
    parser.add_argument(
        "--include-low-signal",
        action="store_true",
        help="Allow cells below the high-signal quantile if they meet other filters",
    )
    parser.add_argument(
        "--cell-type",
        action="append",
        default=None,
        help="Restrict to a cell type. May be passed more than once.",
    )
    parser.add_argument("--min-score", type=float, default=None, help="Minimum regulatory score")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional source .h5ad path. Required only when --output-h5ad is set.",
    )
    parser.add_argument(
        "--output-h5ad",
        default=None,
        help="Optional .h5ad path for the selected candidate-cell subset.",
    )
    args = parser.parse_args()

    report = load_regulatory_screening_report(args.screening_report)
    selection = build_regulatory_candidate_selection(
        report,
        top_n=args.top_n,
        require_high_signal=not args.include_low_signal,
        cell_types=args.cell_type,
        min_score=args.min_score,
    )
    write_regulatory_candidate_selection(selection, args.output_path)

    if args.output_h5ad:
        if not args.data_path:
            raise SystemExit("--data-path is required when --output-h5ad is set.")
        _write_h5ad_subset(
            args.data_path,
            args.output_h5ad,
            selection["selected_cell_indices"],
        )


if __name__ == "__main__":
    main()
