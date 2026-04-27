"""Inspect downloaded validation AnnData objects before expensive runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from biology.oskm import resolve_oskm_genes


def _to_list(values: Iterable[Any]) -> list[Any]:
    """Convert common AnnData/pandas containers to plain Python lists."""
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _obs_columns(adata: Any) -> list[str]:
    return [str(column) for column in getattr(adata, "obs").columns]


def _column_values(adata: Any, column_name: str | None) -> list[Any]:
    if not column_name or column_name not in _obs_columns(adata):
        return []
    return _to_list(getattr(adata, "obs")[column_name])


def summarize_column_values(values: Iterable[Any], *, limit: int = 20) -> dict:
    """Summarize distinct values and top counts for a metadata column."""
    value_strings = [str(value) for value in values]
    counts = Counter(value_strings)
    return {
        "n_unique": len(counts),
        "values": sorted(counts)[:limit],
        "top_counts": [
            {"value": value, "count": count}
            for value, count in counts.most_common(limit)
        ],
    }


def build_validation_dataset_inspection(
    adata: Any,
    *,
    dataset_name: str,
    species: str,
    cell_type_column: str | None = None,
    timepoint_column: str | None = None,
) -> dict:
    """Build a lightweight inspection report for an AnnData-like object."""
    obs_columns = _obs_columns(adata)
    var_names = [str(name) for name in _to_list(getattr(adata, "var_names"))]
    cell_type_values = _column_values(adata, cell_type_column)
    timepoint_values = _column_values(adata, timepoint_column)
    resolved_oskm = resolve_oskm_genes(var_names, species=species)

    return {
        "artifact_type": "validation_dataset_inspection",
        "dataset_name": dataset_name,
        "species": species,
        "n_cells": int(getattr(adata, "n_obs")),
        "n_genes": int(getattr(adata, "n_vars")),
        "obs_columns": obs_columns,
        "var_name_sample": var_names[:20],
        "cell_type_column": cell_type_column,
        "timepoint_column": timepoint_column,
        "cell_type_column_present": bool(cell_type_column in obs_columns)
        if cell_type_column
        else False,
        "timepoint_column_present": bool(timepoint_column in obs_columns)
        if timepoint_column
        else False,
        "cell_type_summary": summarize_column_values(cell_type_values)
        if cell_type_values
        else None,
        "timepoint_summary": summarize_column_values(timepoint_values)
        if timepoint_values
        else None,
        "resolved_oskm_genes": resolved_oskm,
        "missing_oskm_genes": [
            canonical
            for canonical in ("POU5F1", "SOX2", "KLF4", "MYC")
            if canonical not in {gene.upper(): gene for gene in resolved_oskm}
        ],
        "ready_for_profile_review": bool(
            cell_type_column
            and timepoint_column
            and cell_type_column in obs_columns
            and timepoint_column in obs_columns
            and resolved_oskm
        ),
    }


def write_validation_dataset_inspection(
    inspection: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """Write a validation dataset inspection report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")
