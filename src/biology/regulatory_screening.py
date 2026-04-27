"""OSKM-adjacent regulatory pathway screening helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from biology.panels import score_expression_panels


DEFAULT_OSKM_REGULATORY_PANELS: dict[str, Sequence[str]] = {
    "oskm_core": ("POU5F1", "SOX2", "KLF4", "MYC"),
    "chromatin_remodeling": ("KDM1A", "KDM6B", "TET1", "TET2", "DNMT1"),
    "senescence_stress": ("CDKN1A", "CDKN2A", "TP53", "GADD45A", "SERPINE1"),
    "metabolic_plasticity": ("SIRT1", "FOXO3", "PPARGC1A", "NFE2L2", "TFAM"),
}


def build_regulatory_pathway_scores(
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    panels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, np.ndarray]:
    """Score OSKM-adjacent regulatory panels for each cell."""
    return score_expression_panels(
        expression_matrix,
        var_names,
        panels or DEFAULT_OSKM_REGULATORY_PANELS,
    )


def build_regulatory_screening_rows(
    *,
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    cell_ids: Sequence[str] | None = None,
    cell_types: Sequence[str] | None = None,
    panels: Mapping[str, Sequence[str]] | None = None,
    high_signal_quantile: float = 0.75,
) -> list[dict]:
    """Build cell-level screening rows for OSKM-adjacent regulatory activity."""
    matrix = np.asarray(expression_matrix)
    n_cells = matrix.shape[0]
    scores = build_regulatory_pathway_scores(matrix, var_names, panels)
    total_scores = np.zeros(n_cells, dtype=float)
    for values in scores.values():
        total_scores += np.asarray(values, dtype=float)
    threshold = float(np.quantile(total_scores, high_signal_quantile)) if n_cells else 0.0

    rows = []
    for idx in range(n_cells):
        row = {
            "cell_index": idx,
            "cell_id": str(cell_ids[idx]) if cell_ids is not None else str(idx),
            "cell_type": str(cell_types[idx]) if cell_types is not None else "unknown",
            "regulatory_screen_score": float(total_scores[idx]),
            "high_regulatory_signal": bool(total_scores[idx] >= threshold),
        }
        for panel_name, values in scores.items():
            row[f"{panel_name}_score"] = float(values[idx])
        rows.append(row)
    return rows


def summarize_regulatory_screening_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    """Summarize regulatory screening rows."""
    if not rows:
        return {
            "n_cells": 0,
            "high_signal_fraction": 0.0,
            "mean_regulatory_screen_score": 0.0,
        }
    high_count = sum(bool(row.get("high_regulatory_signal", False)) for row in rows)
    mean_score = sum(float(row.get("regulatory_screen_score", 0.0)) for row in rows) / len(rows)
    return {
        "n_cells": len(rows),
        "high_signal_fraction": high_count / len(rows),
        "mean_regulatory_screen_score": mean_score,
    }
