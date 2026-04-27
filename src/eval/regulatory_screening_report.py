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


def select_regulatory_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_n: int | None = None,
    require_high_signal: bool = True,
    cell_types: Sequence[str] | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """Select candidate cells for downstream root-finding stress tests."""
    allowed_cell_types = {str(cell_type) for cell_type in cell_types or []}
    selected = []
    for row in rows:
        if require_high_signal and not bool(row.get("high_regulatory_signal", False)):
            continue
        if allowed_cell_types and str(row.get("cell_type", "unknown")) not in allowed_cell_types:
            continue
        score = float(row.get("regulatory_screen_score", 0.0))
        if min_score is not None and score < min_score:
            continue
        selected.append(dict(row))
    selected.sort(
        key=lambda row: float(row.get("regulatory_screen_score", 0.0)),
        reverse=True,
    )
    if top_n is not None:
        selected = selected[: max(0, int(top_n))]
    return selected


def build_regulatory_candidate_selection(
    screening_report: Mapping[str, Any],
    *,
    top_n: int | None = None,
    require_high_signal: bool = True,
    cell_types: Sequence[str] | None = None,
    min_score: float | None = None,
) -> dict:
    """Build a manifest of selected OSKM-adjacent candidate cells."""
    selected_rows = select_regulatory_candidate_rows(
        screening_report.get("rows", []),
        top_n=top_n,
        require_high_signal=require_high_signal,
        cell_types=cell_types,
        min_score=min_score,
    )
    return {
        "artifact_type": "regulatory_candidate_selection",
        "dataset_name": screening_report.get("dataset_name"),
        "source_high_signal_quantile": screening_report.get("high_signal_quantile"),
        "selection_criteria": {
            "top_n": top_n,
            "require_high_signal": require_high_signal,
            "cell_types": list(cell_types or []),
            "min_score": min_score,
        },
        "summary": {
            "n_source_cells": len(screening_report.get("rows", [])),
            "n_selected_cells": len(selected_rows),
            "selected_fraction": (
                len(selected_rows) / len(screening_report.get("rows", []))
                if screening_report.get("rows")
                else 0.0
            ),
            "mean_selected_score": (
                sum(float(row.get("regulatory_screen_score", 0.0)) for row in selected_rows)
                / len(selected_rows)
                if selected_rows
                else 0.0
            ),
        },
        "selected_cell_ids": [str(row.get("cell_id")) for row in selected_rows],
        "selected_cell_indices": [int(row.get("cell_index")) for row in selected_rows],
        "rows": selected_rows,
    }


def load_regulatory_screening_report(report_path: str | Path) -> dict:
    """Load a regulatory screening report JSON."""
    with Path(report_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_regulatory_candidate_selection(
    selection: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Write a regulatory candidate-cell selection manifest."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
