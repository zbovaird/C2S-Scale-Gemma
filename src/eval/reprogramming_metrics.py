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
MIN_REJUVENATION_SCORE = 0.30
SAFE_MAX_PLURIPOTENCY_MARKER = 0.40
RISK_MIN_PLURIPOTENCY_MARKER = 0.60
MIN_PROGRESS_DELTA = 0.0
MIN_SOMATIC_ESCAPE = 0.0


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
    baseline_rejuvenation_scores: Sequence[float] | None = None,
    perturbed_rejuvenation_scores: Sequence[float] | None = None,
    baseline_pluripotency_marker_scores: Sequence[float] | None = None,
    perturbed_pluripotency_marker_scores: Sequence[float] | None = None,
    somatic_labels: Sequence[str] = DEFAULT_SOMATIC_LABELS,
    pluripotent_labels: Sequence[str] = DEFAULT_PLURIPOTENT_LABELS,
    window_profile: dict | None = None,
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

    profile = {
        "partial_window_proximity_min": PARTIAL_WINDOW_PROXIMITY_MIN,
        "partial_window_proximity_max": PARTIAL_WINDOW_PROXIMITY_MAX,
        "partial_window_max_risk": PARTIAL_WINDOW_MAX_RISK,
        "longevity_safe_proximity_max": LONGEVITY_SAFE_PROXIMITY_MAX,
        "longevity_safe_max_risk": LONGEVITY_SAFE_MAX_RISK,
        "pluripotency_risk_proximity_min": PLURIPOTENCY_RISK_PROXIMITY_MIN,
        "pluripotency_risk_score_min": PLURIPOTENCY_RISK_SCORE_MIN,
        "min_rejuvenation_score": MIN_REJUVENATION_SCORE,
        "safe_max_pluripotency_marker": SAFE_MAX_PLURIPOTENCY_MARKER,
        "risk_min_pluripotency_marker": RISK_MIN_PLURIPOTENCY_MARKER,
        "min_progress_delta": MIN_PROGRESS_DELTA,
        "min_somatic_escape": MIN_SOMATIC_ESCAPE,
    }
    if window_profile:
        profile.update(window_profile)

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
    baseline_rejuvenation = (
        np.asarray(list(baseline_rejuvenation_scores), dtype=float)
        if baseline_rejuvenation_scores is not None
        else np.zeros(baseline.shape[0], dtype=float)
    )
    perturbed_rejuvenation = (
        np.asarray(list(perturbed_rejuvenation_scores), dtype=float)
        if perturbed_rejuvenation_scores is not None
        else baseline_rejuvenation
    )
    baseline_pluripotency_markers = (
        np.asarray(list(baseline_pluripotency_marker_scores), dtype=float)
        if baseline_pluripotency_marker_scores is not None
        else np.zeros(baseline.shape[0], dtype=float)
    )
    perturbed_pluripotency_markers = (
        np.asarray(list(perturbed_pluripotency_marker_scores), dtype=float)
        if perturbed_pluripotency_marker_scores is not None
        else baseline_pluripotency_markers
    )

    rejuvenation_score = _normalize(perturbed_rejuvenation)
    pluripotency_marker_score = _normalize(perturbed_pluripotency_markers)
    risk_components = (
        _normalize(perturbed_to_pluripotent, invert=True)
        + _normalize(l2_shift)
        + _normalize(perturbed_oskm)
        + pluripotency_marker_score
    ) / 4.0
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
            profile["partial_window_proximity_min"]
            <= pluripotency_proximity[idx]
            <= profile["partial_window_proximity_max"]
            and progress_delta[idx] > profile["min_progress_delta"]
            and risk_components[idx] <= profile["partial_window_max_risk"]
            and branch_label in {"productive", "ambiguous"}
            and rejuvenation_score[idx] >= profile["min_rejuvenation_score"]
        )
        in_longevity_safe_zone = (
            in_partial_window
            and pluripotency_proximity[idx] <= profile["longevity_safe_proximity_max"]
            and risk_components[idx] <= profile["longevity_safe_max_risk"]
            and somatic_escape[idx] > profile["min_somatic_escape"]
            and branch_label != "alternative"
            and pluripotency_marker_score[idx] <= profile["safe_max_pluripotency_marker"]
        )
        pluripotency_risk_flag = (
            pluripotency_proximity[idx] >= profile["pluripotency_risk_proximity_min"]
            and (
                risk_components[idx] >= profile["pluripotency_risk_score_min"]
                or pluripotency_marker_score[idx] >= profile["risk_min_pluripotency_marker"]
            )
        )

        rows.append(
            {
                "cell_index": idx,
                "branch_label": branch_label,
                "risk_score": float(risk_components[idx]),
                "pluripotency_proximity": float(pluripotency_proximity[idx]),
                "rejuvenation_score": float(rejuvenation_score[idx]),
                "pluripotency_marker_score": float(pluripotency_marker_score[idx]),
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
