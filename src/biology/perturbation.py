"""Utilities for in silico OSKM perturbations."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from biology.oskm import resolve_oskm_genes


def apply_oskm_perturbation(
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    mode: str = "silence",
    species: str = "human",
    factor: float = 2.0,
    target_genes: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply an in silico OSKM perturbation to an expression matrix.

    Modes:
    - ``silence``: zero out target genes
    - ``overexpress``: multiply target genes by ``factor``
    - ``scale``: same as ``overexpress`` but semantically neutral naming
    """

    matrix = np.asarray(expression_matrix).copy()
    if matrix.ndim != 2:
        raise ValueError("expression_matrix must be 2D")

    resolved = resolve_oskm_genes(var_names, species=species)
    if target_genes is None:
        selected_symbols = list(resolved.values())
    else:
        selected_symbols = []
        for canonical in target_genes:
            matched = resolved.get(canonical)
            if matched is not None:
                selected_symbols.append(matched)

    if not selected_symbols:
        return matrix, {
            "mode": mode,
            "factor": factor,
            "species": species,
            "perturbed_symbols": [],
            "perturbed_indices": [],
        }

    gene_index_lookup = {str(name): idx for idx, name in enumerate(var_names)}
    perturbed_indices = [gene_index_lookup[symbol] for symbol in selected_symbols]

    if mode == "silence":
        matrix[:, perturbed_indices] = 0.0
    elif mode in {"overexpress", "scale"}:
        matrix[:, perturbed_indices] = matrix[:, perturbed_indices] * factor
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")

    return matrix, {
        "mode": mode,
        "factor": factor,
        "species": species,
        "perturbed_symbols": selected_symbols,
        "perturbed_indices": perturbed_indices,
    }


def summarize_perturbation_shift(
    baseline_matrix: np.ndarray,
    perturbed_matrix: np.ndarray,
    var_names: Sequence[str],
    perturbed_symbols: Sequence[str],
) -> list[dict]:
    """Summarize mean expression changes for perturbed genes."""
    base = np.asarray(baseline_matrix)
    perturbed = np.asarray(perturbed_matrix)
    gene_index_lookup = {str(name): idx for idx, name in enumerate(var_names)}

    summary = []
    for symbol in perturbed_symbols:
        idx = gene_index_lookup[symbol]
        baseline_mean = float(np.mean(base[:, idx]))
        perturbed_mean = float(np.mean(perturbed[:, idx]))
        summary.append(
            {
                "gene": symbol,
                "baseline_mean": baseline_mean,
                "perturbed_mean": perturbed_mean,
                "delta": perturbed_mean - baseline_mean,
            }
        )
    return summary


def compute_cellwise_delta_summary(
    baseline_scores: Iterable[float],
    perturbed_scores: Iterable[float],
) -> dict:
    """Aggregate per-cell score shifts into a compact summary."""
    baseline = np.asarray(list(baseline_scores), dtype=float)
    perturbed = np.asarray(list(perturbed_scores), dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline_scores and perturbed_scores must have same shape")

    deltas = perturbed - baseline
    return {
        "mean_delta": float(np.mean(deltas)) if deltas.size else 0.0,
        "max_delta": float(np.max(deltas)) if deltas.size else 0.0,
        "min_delta": float(np.min(deltas)) if deltas.size else 0.0,
        "n_cells": int(deltas.size),
    }
