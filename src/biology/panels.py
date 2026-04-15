"""Marker-gene panels for rejuvenation and pluripotency-risk heuristics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_MARKER_PANELS: Dict[str, Sequence[str]] = {
    "rejuvenation": (
        "SIRT1",
        "FOXO3",
        "PPARGC1A",
        "TFAM",
        "NFE2L2",
    ),
    "pluripotency_risk": (
        "NANOG",
        "LIN28A",
        "DPPA4",
        "UTF1",
        "PRDM14",
    ),
}


def merge_marker_panels(
    custom_panels: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Sequence[str]]:
    """Merge custom marker panels onto the defaults."""
    panels = {key: tuple(values) for key, values in DEFAULT_MARKER_PANELS.items()}
    if custom_panels:
        for key, values in custom_panels.items():
            panels[key] = tuple(values)
    return panels


def resolve_marker_panel_genes(
    var_names: Iterable[str],
    panel_genes: Sequence[str],
) -> list[str]:
    """Resolve panel genes to exact symbols present in ``var_names``."""
    lookup = {str(name).upper(): str(name) for name in var_names}
    resolved = []
    for gene in panel_genes:
        match = lookup.get(str(gene).upper())
        if match is not None:
            resolved.append(match)
    return resolved


def score_expression_panel(
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    panel_genes: Sequence[str],
) -> np.ndarray:
    """Compute an additive panel score for each cell."""
    resolved = resolve_marker_panel_genes(var_names, panel_genes)
    if not resolved:
        return np.zeros(expression_matrix.shape[0], dtype=float)

    var_lookup = {str(name): idx for idx, name in enumerate(var_names)}
    indices = [var_lookup[gene] for gene in resolved]
    matrix = np.asarray(expression_matrix)
    return np.sum(matrix[:, indices], axis=1)


def score_expression_panels(
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    panels: Mapping[str, Sequence[str]],
) -> Dict[str, np.ndarray]:
    """Compute all configured marker-panel scores."""
    return {
        panel_name: score_expression_panel(expression_matrix, var_names, panel_genes)
        for panel_name, panel_genes in panels.items()
    }
