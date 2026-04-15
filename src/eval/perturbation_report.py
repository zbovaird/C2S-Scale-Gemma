"""Helpers for building OSKM perturbation reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def summarize_shift_by_category(
    rows: Sequence[dict],
    category_key: str,
    value_key: str = "l2_shift",
) -> list[dict]:
    """Aggregate shift metrics by a categorical column."""
    grouped_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        category = str(row.get(category_key, "unknown"))
        value = row.get(value_key)
        if value is None:
            continue
        grouped_values[category].append(float(value))

    summary_rows = []
    for category, values in grouped_values.items():
        if not values:
            continue
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        median = (
            values_sorted[mid]
            if len(values_sorted) % 2 == 1
            else (values_sorted[mid - 1] + values_sorted[mid]) / 2.0
        )
        summary_rows.append(
            {
                "category": category,
                "count": len(values),
                "mean_shift": sum(values) / len(values),
                "median_shift": median,
                "max_shift": max(values),
            }
        )

    return sorted(summary_rows, key=lambda row: row["mean_shift"], reverse=True)


def get_top_shift_rows(
    rows: Sequence[dict],
    top_n: int = 20,
    sort_key: str = "l2_shift",
) -> list[dict]:
    """Return the rows with the largest reported shifts."""
    return sorted(
        rows,
        key=lambda row: float(row.get(sort_key, 0.0)),
        reverse=True,
    )[:top_n]


def compute_shift_histogram(
    rows: Sequence[dict],
    value_key: str = "l2_shift",
    bins: int = 20,
) -> dict:
    """Compute a simple histogram without requiring numpy/pandas."""
    values = [float(row.get(value_key, 0.0)) for row in rows]
    if not values:
        return {"counts": [], "bin_edges": []}

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return {"counts": [len(values)], "bin_edges": [min_value, max_value]}

    width = (max_value - min_value) / bins
    counts = [0] * bins
    edges = [min_value + (width * idx) for idx in range(bins + 1)]

    for value in values:
        if value == max_value:
            counts[-1] += 1
            continue
        bin_index = int((value - min_value) / width)
        counts[bin_index] += 1

    return {"counts": counts, "bin_edges": edges}
