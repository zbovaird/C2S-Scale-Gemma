"""Report helpers for OSKM-adjacent regulatory screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from biology.regulatory_screening import (
    DEFAULT_OSKM_REGULATORY_PANELS,
    build_regulatory_screening_rows,
    summarize_regulatory_screening_rows,
)


def build_regulatory_screening_report(
    *,
    expression_matrix: np.ndarray,
    var_names: Sequence[str],
    cell_ids: Sequence[str] | None = None,
    cell_types: Sequence[str] | None = None,
    dataset_name: str | None = None,
    panels: Mapping[str, Sequence[str]] | None = None,
    high_signal_quantile: float = 0.75,
) -> dict:
    """Build a complete screening report from expression data."""
    active_panels = dict(panels or DEFAULT_OSKM_REGULATORY_PANELS)
    rows = build_regulatory_screening_rows(
        expression_matrix=expression_matrix,
        var_names=var_names,
        cell_ids=cell_ids,
        cell_types=cell_types,
        panels=active_panels,
        high_signal_quantile=high_signal_quantile,
    )
    return {
        "artifact_type": "regulatory_screening_report",
        "dataset_name": dataset_name,
        "high_signal_quantile": high_signal_quantile,
        "panels": {name: list(genes) for name, genes in active_panels.items()},
        "summary": summarize_regulatory_screening_rows(rows),
        "rows": rows,
    }


def write_regulatory_screening_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    """Write a regulatory screening report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
