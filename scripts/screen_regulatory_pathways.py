#!/usr/bin/env python3
"""Screen an AnnData file for OSKM-adjacent regulatory pathway activity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.regulatory_screening_report import (
    build_regulatory_screening_report,
    write_regulatory_screening_report,
)


def _to_dense_matrix(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen regulatory pathway activity")
    parser.add_argument("--data-path", type=str, required=True, help="Input .h5ad path")
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/regulatory_screening_report.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--cell-type-column",
        type=str,
        default="cell_type",
        help="Observation column containing cell type labels",
    )
    parser.add_argument(
        "--high-signal-quantile",
        type=float,
        default=0.75,
        help="Quantile cutoff for high regulatory signal",
    )
    args = parser.parse_args()

    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - dependency is project-level optional at runtime
        raise SystemExit("scanpy is required to screen .h5ad files.") from exc

    adata = sc.read_h5ad(args.data_path)
    cell_types = (
        adata.obs[args.cell_type_column].astype(str).tolist()
        if args.cell_type_column in adata.obs
        else None
    )
    report = build_regulatory_screening_report(
        expression_matrix=_to_dense_matrix(adata.X),
        var_names=[str(name) for name in adata.var_names],
        cell_ids=[str(name) for name in adata.obs_names],
        cell_types=cell_types,
        dataset_name=args.data_path,
        high_signal_quantile=args.high_signal_quantile,
    )
    write_regulatory_screening_report(report, args.output_path)


if __name__ == "__main__":
    main()
