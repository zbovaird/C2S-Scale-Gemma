"""Lightweight reprogramming metrics for branch and risk overlays."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


DEFAULT_SOMATIC_LABELS = ("fibroblast", "somatic", "starting_state")
DEFAULT_PLURIPOTENT_LABELS = ("esc", "ipsc", "pluripotent", "stem_cell")
PARTIAL_WINDOW_PROXIMITY_MIN = 0.35
PARTIAL_WINDOW_PROXIMITY_MAX = 0.75
PARTIAL_WINDOW_MAX_RISK = 0.60
LONGEVITY_SAFE_PROXIMITY_MAX = 0.65
LONGEVITY_SAFE_MAX_RISK = 0.45
PLURIPOTENCY_RISK_PROXIMITY_MIN = 0.80
PLURIPOTENCY_RISK_SCORE_MIN = 0.60


def _normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        normalized = np.zeros_like(values, dtype=float)
    else:
        normalized = (values - min_value) / (max_value - min_value)
    return 1.0 - normalized if invert else normalized


def _match_labels(cell_types: Sequence[object], candidates: Sequence[str]) -> np.ndarray:
    lowered_candidates = {candidate.lower() for candidate in candidates}
    return np.array([str(label).lower() in lowered_candidates for label in cell_types])


def build_reprogramming_overlay_rows(
    baseline_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    cell_types: Sequence[object],
    baseline_oskm_scores: Sequence[float] | None = None,
    perturbed_oskm_scores: Sequence[float] | None = None,
    somatic_labels: Sequence[str] = DEFAULT_SOMATIC_LABELS,
    pluripotent_labels: Sequence[str] = DEFAULT_PLURIPOTENT_LABELS,
) -> list[dict]:
    """Build per-cell overlay rows for branch and risk interpretation."""
    baseline = np.asarray(baseline_embeddings, dtype=float)
    perturbed = np.asarray(perturbed_embeddings, dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline_embeddings and perturbed_embeddings must have matching shape.")
    if baseline.ndim != 2:
        raise ValueError("Embeddings must be 2D.")
    if len(cell_types) != baseline.shape[0]:
        raise ValueError("cell_types length must match number of embeddings.")

    somatic_mask = _match_labels(cell_types, somatic_labels)
    pluripotent_mask = _match_labels(cell_types, pluripotent_labels)
    if not np.any(somatic_mask) or not np.any(pluripotent_mask):
        return [
            {
                "cell_index": idx,
                "branch_label": "unavailable",
                "risk_score": None,
                "progress_delta": None,
            }
            for idx in range(baseline.shape[0])
        ]

    somatic_centroid = np.mean(baseline[somatic_mask], axis=0)
    pluripotent_centroid = np.mean(baseline[pluripotent_mask], axis=0)

    baseline_to_somatic = np.linalg.norm(baseline - somatic_centroid, axis=1)
    baseline_to_pluripotent = np.linalg.norm(baseline - pluripotent_centroid, axis=1)
    perturbed_to_somatic = np.linalg.norm(perturbed - somatic_centroid, axis=1)
    perturbed_to_pluripotent = np.linalg.norm(perturbed - pluripotent_centroid, axis=1)

    baseline_progress = baseline_to_somatic - baseline_to_pluripotent
    perturbed_progress = perturbed_to_somatic - perturbed_to_pluripotent
    progress_delta = perturbed_progress - baseline_progress
    pluripotency_gain = baseline_to_pluripotent - perturbed_to_pluripotent
    somatic_escape = perturbed_to_somatic - baseline_to_somatic
    l2_shift = np.linalg.norm(perturbed - baseline, axis=1)

    baseline_oskm = (
        np.asarray(list(baseline_oskm_scores), dtype=float)
        if baseline_oskm_scores is not None
        else np.zeros(baseline.shape[0], dtype=float)
    )
    perturbed_oskm = (
        np.asarray(list(perturbed_oskm_scores), dtype=float)
        if perturbed_oskm_scores is not None
        else baseline_oskm
    )

    risk_components = (
        _normalize(perturbed_to_pluripotent, invert=True)
        + _normalize(l2_shift)
        + _normalize(perturbed_oskm)
    ) / 3.0
    pluripotency_proximity = _normalize(perturbed_to_pluripotent, invert=True)

    rows = []
    for idx in range(baseline.shape[0]):
        if progress_delta[idx] > 0 and pluripotency_gain[idx] > 0:
            branch_label = "productive"
        elif progress_delta[idx] < 0 and somatic_escape[idx] <= 0:
            branch_label = "somatic_retention"
        elif progress_delta[idx] < 0:
            branch_label = "alternative"
        else:
            branch_label = "ambiguous"

        in_partial_window = (
            PARTIAL_WINDOW_PROXIMITY_MIN
            <= pluripotency_proximity[idx]
            <= PARTIAL_WINDOW_PROXIMITY_MAX
            and progress_delta[idx] > 0
            and risk_components[idx] <= PARTIAL_WINDOW_MAX_RISK
            and branch_label in {"productive", "ambiguous"}
        )
        in_longevity_safe_zone = (
            in_partial_window
            and pluripotency_proximity[idx] <= LONGEVITY_SAFE_PROXIMITY_MAX
            and risk_components[idx] <= LONGEVITY_SAFE_MAX_RISK
            and somatic_escape[idx] > 0
            and branch_label != "alternative"
        )
        pluripotency_risk_flag = (
            pluripotency_proximity[idx] >= PLURIPOTENCY_RISK_PROXIMITY_MIN
            and risk_components[idx] >= PLURIPOTENCY_RISK_SCORE_MIN
        )

        rows.append(
            {
                "cell_index": idx,
                "branch_label": branch_label,
                "risk_score": float(risk_components[idx]),
                "pluripotency_proximity": float(pluripotency_proximity[idx]),
                "baseline_to_somatic": float(baseline_to_somatic[idx]),
                "baseline_to_pluripotent": float(baseline_to_pluripotent[idx]),
                "perturbed_to_somatic": float(perturbed_to_somatic[idx]),
                "perturbed_to_pluripotent": float(perturbed_to_pluripotent[idx]),
                "baseline_progress": float(baseline_progress[idx]),
                "perturbed_progress": float(perturbed_progress[idx]),
                "progress_delta": float(progress_delta[idx]),
                "pluripotency_gain": float(pluripotency_gain[idx]),
                "somatic_escape": float(somatic_escape[idx]),
                "partial_reprogramming_window": bool(in_partial_window),
                "longevity_safe_zone": bool(in_longevity_safe_zone),
                "pluripotency_risk_flag": bool(pluripotency_risk_flag),
            }
        )

    return rows


def summarize_branch_counts(overlay_rows: Sequence[dict]) -> dict:
    """Summarize branch labels and average risk."""
    summary: dict[str, dict] = {}
    for row in overlay_rows:
        branch = str(row.get("branch_label", "unknown"))
        risk_score = row.get("risk_score")
        branch_state = summary.setdefault(
            branch,
            {"count": 0, "risk_scores": []},
        )
        branch_state["count"] += 1
        if risk_score is not None:
            branch_state["risk_scores"].append(float(risk_score))

    for branch, values in summary.items():
        risk_scores = values.pop("risk_scores")
        values["mean_risk_score"] = (
            float(np.mean(risk_scores)) if risk_scores else None
        )
    return summary


def summarize_zone_counts(overlay_rows: Sequence[dict]) -> dict:
    """Summarize partial-window, safe-zone, and pluripotency-risk cohorts."""
    total_rows = len(overlay_rows)
    partial_count = sum(bool(row.get("partial_reprogramming_window")) for row in overlay_rows)
    safe_count = sum(bool(row.get("longevity_safe_zone")) for row in overlay_rows)
    risk_count = sum(bool(row.get("pluripotency_risk_flag")) for row in overlay_rows)
    return {
        "n_cells": total_rows,
        "partial_reprogramming_window_count": partial_count,
        "partial_reprogramming_window_fraction": (partial_count / total_rows) if total_rows else 0.0,
        "longevity_safe_zone_count": safe_count,
        "longevity_safe_zone_fraction": (safe_count / total_rows) if total_rows else 0.0,
        "pluripotency_risk_count": risk_count,
        "pluripotency_risk_fraction": (risk_count / total_rows) if total_rows else 0.0,
    }
