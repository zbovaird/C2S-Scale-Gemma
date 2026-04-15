"""Utilities for comparing baseline and perturbed embeddings."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def summarize_embedding_shift(
    baseline_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
) -> dict:
    """Summarize how far embeddings move after perturbation."""
    baseline = np.asarray(baseline_embeddings, dtype=float)
    perturbed = np.asarray(perturbed_embeddings, dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline_embeddings and perturbed_embeddings must match shape.")
    if baseline.ndim != 2:
        raise ValueError("Embedding arrays must be 2D.")

    deltas = perturbed - baseline
    l2_shift = np.linalg.norm(deltas, axis=1)

    baseline_norm = np.linalg.norm(baseline, axis=1)
    perturbed_norm = np.linalg.norm(perturbed, axis=1)
    denom = np.clip(baseline_norm * perturbed_norm, a_min=1e-8, a_max=None)
    cosine_similarity = np.sum(baseline * perturbed, axis=1) / denom

    return {
        "n_cells": int(baseline.shape[0]),
        "embedding_dim": int(baseline.shape[1]),
        "mean_l2_shift": float(np.mean(l2_shift)) if l2_shift.size else 0.0,
        "median_l2_shift": float(np.median(l2_shift)) if l2_shift.size else 0.0,
        "max_l2_shift": float(np.max(l2_shift)) if l2_shift.size else 0.0,
        "mean_cosine_similarity": float(np.mean(cosine_similarity))
        if cosine_similarity.size
        else 0.0,
    }


def build_embedding_shift_frame(
    baseline_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    metadata: Optional[Mapping[str, Sequence[object]]] = None,
) -> list[dict]:
    """Build row-wise records for before/after embedding comparisons."""
    baseline = np.asarray(baseline_embeddings, dtype=float)
    perturbed = np.asarray(perturbed_embeddings, dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline_embeddings and perturbed_embeddings must match shape.")

    rows = []
    deltas = perturbed - baseline
    l2_shift = np.linalg.norm(deltas, axis=1)

    for idx in range(baseline.shape[0]):
        row = {
            "cell_index": idx,
            "l2_shift": float(l2_shift[idx]),
            "baseline_norm": float(np.linalg.norm(baseline[idx])),
            "perturbed_norm": float(np.linalg.norm(perturbed[idx])),
        }
        if metadata:
            for key, values in metadata.items():
                if len(values) != baseline.shape[0]:
                    raise ValueError(f"Metadata column '{key}' has incorrect length.")
                row[key] = values[idx]
        rows.append(row)
    return rows
