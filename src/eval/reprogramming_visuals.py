"""Utilities for preparing OKSM / reprogramming progress visualizations."""

from typing import Mapping, Optional, Sequence

import numpy as np

DEFAULT_OSKM_GENES = ("POU5F1", "SOX2", "KLF4", "MYC")


def build_dataset_readiness_summary(
    obs: Mapping[str, Sequence[object]],
    var_names: Sequence[str],
    oskm_genes: Sequence[str] = DEFAULT_OSKM_GENES,
    timepoint_column: Optional[str] = None,
    batch_column: Optional[str] = None,
) -> dict:
    """Summarize whether a dataset is ready for OKSM-aware analysis."""
    obs_lengths = [len(values) for values in obs.values()] if obs else [0]
    n_cells = obs_lengths[0]
    if any(length != n_cells for length in obs_lengths):
        raise ValueError("All observation columns must have the same length.")

    var_name_set = {str(name).upper() for name in var_names}
    summary = {
        "n_cells": int(n_cells),
        "n_genes": int(len(var_names)),
    }

    if timepoint_column and timepoint_column in obs:
        summary["n_timepoints"] = int(len(set(obs[timepoint_column])))
    if batch_column and batch_column in obs:
        summary["n_batches"] = int(len(set(obs[batch_column])))

    for gene in oskm_genes:
        summary[f"oskm_present_{gene}"] = gene.upper() in var_name_set

    return summary


def build_sentence_debug_frame(
    cell_sentences: Sequence[str],
    oskm_genes: Sequence[str] = DEFAULT_OSKM_GENES,
) -> list[dict]:
    """Create a debugging table showing where OSKM genes land in each sentence."""
    rows = []
    tracked_genes = [gene.upper() for gene in oskm_genes]

    for idx, sentence in enumerate(cell_sentences):
        tokens = [token.strip() for token in sentence.split() if token.strip()]
        token_lookup = {token.upper(): position for position, token in enumerate(tokens)}
        row = {
            "cell_index": idx,
            "sentence_length": len(tokens),
        }
        for gene in tracked_genes:
            row[f"{gene.lower()}_position"] = token_lookup.get(gene)
        rows.append(row)

    return rows


def build_embedding_frame(
    embeddings: np.ndarray,
    metadata: Optional[Mapping[str, Sequence[object]]] = None,
    columns: Sequence[str] = ("x", "y"),
) -> dict[str, list]:
    """Combine low-dimensional embeddings with metadata for plotting."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    if embeddings.shape[1] < len(columns):
        raise ValueError("Embeddings have fewer dimensions than requested columns.")

    frame = {
        column: embeddings[:, idx].tolist()
        for idx, column in enumerate(columns)
    }
    if metadata:
        for key, values in metadata.items():
            if len(values) != embeddings.shape[0]:
                raise ValueError(f"Metadata column '{key}' has incorrect length.")
            frame[key] = list(values)
    return frame


def compute_oskm_score_matrix(
    expression_frame: Mapping[str, Sequence[float]],
    oskm_genes: Sequence[str] = DEFAULT_OSKM_GENES,
) -> np.ndarray:
    """Compute a simple additive OKSM score from an expression table."""
    available_genes = [gene for gene in oskm_genes if gene in expression_frame]
    if not available_genes:
        first_key = next(iter(expression_frame), None)
        n_rows = len(expression_frame[first_key]) if first_key is not None else 0
        return np.zeros(n_rows)

    stacked = np.vstack([np.asarray(expression_frame[gene]) for gene in available_genes])
    return np.sum(stacked, axis=0)
