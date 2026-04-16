"""Helpers for exporting 2D validation trajectory projections."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def project_embedding_pair_to_2d(
    baseline_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project paired baseline/perturbed embeddings into a shared 2D PCA space."""
    baseline = np.asarray(baseline_embeddings, dtype=float)
    perturbed = np.asarray(perturbed_embeddings, dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("baseline_embeddings and perturbed_embeddings must match shape.")
    if baseline.ndim != 2:
        raise ValueError("Embedding arrays must be 2D.")

    combined = np.concatenate([baseline, perturbed], axis=0)
    centered = combined - np.mean(combined, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    projected = centered @ components
    n_rows = baseline.shape[0]
    return projected[:n_rows], projected[n_rows:]


def build_projection_rows(
    *,
    label: str,
    alignment_mode: str,
    baseline_projection: np.ndarray,
    perturbed_projection: np.ndarray,
    fused_shift_rows: Sequence[Dict[str, Any]],
    timepoint_column: str | None = None,
) -> list[dict]:
    """Build scatter/arrow-ready rows for one validation run."""
    if len(fused_shift_rows) != baseline_projection.shape[0]:
        raise ValueError("Projection rows and fused_shift_rows must have the same length.")

    projection_rows = []
    for idx, row in enumerate(fused_shift_rows):
        timepoint = str(row.get(timepoint_column, "unknown")) if timepoint_column else "unknown"
        base_x, base_y = baseline_projection[idx]
        pert_x, pert_y = perturbed_projection[idx]
        projection_rows.append(
            {
                "cell_id": row.get("cell_id", row.get("cell_index", idx)),
                "cell_type": row.get("cell_type"),
                "timepoint": timepoint,
                "label": label,
                "alignment_mode": alignment_mode,
                "branch_label": row.get("branch_label"),
                "partial_reprogramming_window": bool(
                    row.get("partial_reprogramming_window")
                ),
                "longevity_safe_zone": bool(row.get("longevity_safe_zone")),
                "pluripotency_risk_flag": bool(row.get("pluripotency_risk_flag")),
                "baseline_x": float(base_x),
                "baseline_y": float(base_y),
                "perturbed_x": float(pert_x),
                "perturbed_y": float(pert_y),
                "delta_x": float(pert_x - base_x),
                "delta_y": float(pert_y - base_y),
                "l2_shift": float(row.get("l2_shift", 0.0)),
                "progress_delta": float(row.get("progress_delta", 0.0)),
                "risk_score": float(row.get("risk_score", 0.0)),
            }
        )
    return projection_rows


def build_validation_trajectory_projection(
    validation_manifest: Dict[str, Any],
    run_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a 2D trajectory projection artifact for a validation bundle."""
    timepoint_column = validation_manifest.get("track", {}).get("timepoint_column")
    runs = []
    for payload in run_payloads:
        baseline_projection, perturbed_projection = project_embedding_pair_to_2d(
            payload["baseline_fused_embeddings"],
            payload["perturbed_fused_embeddings"],
        )
        rows = build_projection_rows(
            label=str(payload.get("label", "unknown")),
            alignment_mode=str(payload.get("alignment_mode", "unknown")),
            baseline_projection=baseline_projection,
            perturbed_projection=perturbed_projection,
            fused_shift_rows=payload.get("fused_shift_rows", []),
            timepoint_column=timepoint_column,
        )
        runs.append(
            {
                "label": payload.get("label", "unknown"),
                "alignment_mode": payload.get("alignment_mode", "unknown"),
                "projection_method": "pca",
                "rows": rows,
            }
        )
    return {
        "track_name": validation_manifest.get("track_name"),
        "dataset_profile": validation_manifest.get("dataset_profile"),
        "projection_method": "pca",
        "timepoint_column": timepoint_column,
        "runs": runs,
    }
