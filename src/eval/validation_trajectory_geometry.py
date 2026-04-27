"""Geometry-distance trajectory artifacts for validation bundles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:
    from uhg.projective import ProjectiveUHG
except Exception:  # pragma: no cover - optional dependency can recurse on import
    ProjectiveUHG = None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_pairwise_geometry_distances(
    baseline_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    *,
    prefer_projective: bool = True,
) -> tuple[list[float], str, bool]:
    """Compute paired baseline-to-perturbed distances with backend metadata."""
    baseline = np.asarray(baseline_embeddings, dtype=float)
    perturbed = np.asarray(perturbed_embeddings, dtype=float)
    if baseline.shape != perturbed.shape:
        raise ValueError("Baseline and perturbed embeddings must have the same shape.")

    if prefer_projective and ProjectiveUHG is not None:
        uhg = ProjectiveUHG()
        try:
            distances = [
                float(
                    uhg.distance(
                        torch.as_tensor(baseline_row, dtype=torch.float32),
                        torch.as_tensor(perturbed_row, dtype=torch.float32),
                    )
                )
                for baseline_row, perturbed_row in zip(baseline, perturbed)
            ]
            return distances, "projective_uhg_distance", False
        except Exception:
            pass

    distances = np.linalg.norm(perturbed - baseline, axis=1).astype(float).tolist()
    return distances, "euclidean_l2_fallback", prefer_projective


def build_validation_trajectory_geometry(
    validation_manifest: Mapping[str, Any],
    run_payloads: Sequence[Mapping[str, Any]],
) -> dict:
    """Build per-cell geometry-distance rows for validation trajectories."""
    runs = []
    for payload in run_payloads:
        overlay_summary = payload.get("overlay_summary", {})
        alignment_summary = overlay_summary.get("alignment", {})
        alignment_mode = str(payload.get("alignment_mode") or alignment_summary.get("alignment_mode"))
        prefer_projective = alignment_mode in {
            "projective_distance",
            "hyperbolic_distance",
            "projective_uhg",
        }
        distances, backend, fallback_used = compute_pairwise_geometry_distances(
            payload["baseline_fused_embeddings"],
            payload["perturbed_fused_embeddings"],
            prefer_projective=prefer_projective,
        )
        fused_shift_rows = list(payload.get("fused_shift_rows", []))
        cell_rows = []
        for idx, distance in enumerate(distances):
            source_row = fused_shift_rows[idx] if idx < len(fused_shift_rows) else {}
            cell_rows.append(
                {
                    "cell_index": idx,
                    "cell_id": source_row.get("cell_id", str(idx)),
                    "cell_type": source_row.get("cell_type", "unknown"),
                    "timepoint": source_row.get(
                        validation_manifest.get("track", {}).get("timepoint_column", "timepoint"),
                        source_row.get("timepoint", "unknown"),
                    ),
                    "geometry_distance": distance,
                    "l2_shift": _to_float(source_row.get("l2_shift"), distance),
                    "branch_label": source_row.get("branch_label", "unknown"),
                    "longevity_safe_zone": bool(source_row.get("longevity_safe_zone", False)),
                    "pluripotency_risk_flag": bool(
                        source_row.get("pluripotency_risk_flag", False)
                    ),
                }
            )
        runs.append(
            {
                "label": payload.get("label"),
                "alignment_mode": alignment_mode,
                "geometry_distance_backend": backend,
                "geometry_fallback_used": fallback_used,
                "n_cells": len(cell_rows),
                "mean_geometry_distance": float(np.mean(distances)) if distances else 0.0,
                "max_geometry_distance": float(np.max(distances)) if distances else 0.0,
                "cell_rows": cell_rows,
            }
        )
    return {
        "track_name": validation_manifest.get("track_name"),
        "dataset_profile": validation_manifest.get("dataset_profile"),
        "artifact_type": "validation_trajectory_geometry",
        "runs": runs,
    }


def summarize_validation_trajectory_geometry(geometry_payload: Mapping[str, Any]) -> list[dict]:
    """Return compact run-level geometry summaries for benchmark reports."""
    return [
        {
            "label": run.get("label"),
            "alignment_mode": run.get("alignment_mode"),
            "geometry_distance_backend": run.get("geometry_distance_backend"),
            "geometry_fallback_used": run.get("geometry_fallback_used"),
            "n_cells": run.get("n_cells", 0),
            "mean_geometry_distance": run.get("mean_geometry_distance", 0.0),
            "max_geometry_distance": run.get("max_geometry_distance", 0.0),
        }
        for run in geometry_payload.get("runs", [])
    ]
