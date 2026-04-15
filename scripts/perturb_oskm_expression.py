#!/usr/bin/env python3
"""Apply in silico OSKM perturbations to an AnnData matrix."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from biology.perturbation import (
    apply_oskm_perturbation,
    compute_cellwise_delta_summary,
    summarize_perturbation_shift,
)

try:
    import scanpy as sc
except ImportError:  # pragma: no cover - optional runtime dependency
    sc = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply in silico OSKM perturbations")
    parser.add_argument("--data-path", type=str, required=True, help="Input .h5ad path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/oskm_perturbation",
        help="Directory for perturbed outputs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="silence",
        choices=["silence", "overexpress", "scale"],
        help="Perturbation mode",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=2.0,
        help="Scaling factor for overexpression/scale modes",
    )
    parser.add_argument(
        "--species",
        type=str,
        default="human",
        help="Species used for OSKM alias resolution",
    )
    parser.add_argument(
        "--target-genes",
        nargs="*",
        default=None,
        help="Optional canonical OSKM genes to perturb (e.g. POU5F1 SOX2)",
    )
    args = parser.parse_args()

    if sc is None:
        raise ImportError("scanpy is required to run perturb_oskm_expression.py")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.data_path)
    baseline_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    perturbed_matrix, metadata = apply_oskm_perturbation(
        baseline_matrix,
        adata.var_names.tolist(),
        mode=args.mode,
        species=args.species,
        factor=args.factor,
        target_genes=args.target_genes,
    )

    gene_summary = summarize_perturbation_shift(
        baseline_matrix,
        perturbed_matrix,
        adata.var_names.tolist(),
        metadata["perturbed_symbols"],
    )
    baseline_total = np.sum(baseline_matrix, axis=1)
    perturbed_total = np.sum(perturbed_matrix, axis=1)
    cell_summary = compute_cellwise_delta_summary(baseline_total, perturbed_total)

    perturbed_adata = adata.copy()
    perturbed_adata.X = perturbed_matrix
    perturbed_adata.uns["oskm_perturbation"] = {
        **metadata,
        "cell_summary": cell_summary,
    }

    output_stem = f"oskm_{args.mode}"
    perturbed_path = output_dir / f"{output_stem}.h5ad"
    summary_path = output_dir / f"{output_stem}_summary.json"

    perturbed_adata.write_h5ad(perturbed_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **metadata,
                "cell_summary": cell_summary,
                "gene_summary": gene_summary,
            },
            handle,
            indent=2,
        )

    logger.info("Wrote perturbed AnnData to %s", perturbed_path)
    logger.info("Wrote perturbation summary to %s", summary_path)


if __name__ == "__main__":
    main()
