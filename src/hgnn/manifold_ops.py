"""Explicit manifold-boundary operations for HGNN modules."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from uhg.projective import ProjectiveUHG
except Exception:  # pragma: no cover - optional runtime dependency
    ProjectiveUHG = None


PRIMARY_MANIFOLD = "projective_uhg"
SUPPORTED_PRIMARY_MANIFOLDS = {PRIMARY_MANIFOLD}


def normalize_primary_manifold(primary_manifold: str | None = None) -> str:
    """Resolve configured manifold aliases to the supported projective UHG backend."""
    value = (primary_manifold or PRIMARY_MANIFOLD).strip().lower()
    aliases = {
        "projective": PRIMARY_MANIFOLD,
        "projective_distance": PRIMARY_MANIFOLD,
        "projective_uhg": PRIMARY_MANIFOLD,
    }
    normalized = aliases.get(value, value)
    if normalized not in SUPPORTED_PRIMARY_MANIFOLDS:
        raise ValueError(
            f"Unsupported primary manifold '{primary_manifold}'. "
            f"Supported values: {sorted(SUPPORTED_PRIMARY_MANIFOLDS)}"
        )
    return normalized


class TangentSpaceLinear(nn.Module):
    """Linear map that explicitly marks a tangent-space operation.

    This adapter preserves the existing Euclidean linear behavior while making
    the geometry boundary visible in module names, audit reports, and tests.
    """

    geometry_role = "tangent_space_linear"

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProjectiveManifoldBackend:
    """Primary projective-UHG distance backend with explicit fallback metadata."""

    def __init__(
        self,
        primary_manifold: str | None = None,
        *,
        require_backend: bool = False,
    ):
        self.primary_manifold = normalize_primary_manifold(primary_manifold)
        self.require_backend = require_backend
        self.uhg = ProjectiveUHG() if ProjectiveUHG is not None else None

    @property
    def backend_available(self) -> bool:
        return self.uhg is not None

    def pairwise_distance(
        self,
        left_embeddings: torch.Tensor,
        right_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, str, bool]:
        """Compute pairwise distances and report the concrete backend used."""
        if self.uhg is None:
            if self.require_backend:
                raise RuntimeError(
                    "Geometry-aware alignment requires the projective UHG backend, "
                    "but it is unavailable. Disable require_backend to allow "
                    "the Euclidean fallback."
                )
            return (
                torch.cdist(left_embeddings, right_embeddings, p=2),
                "euclidean_cdist_fallback",
                True,
            )

        distances = torch.zeros(
            left_embeddings.size(0),
            right_embeddings.size(0),
            device=left_embeddings.device,
        )
        for i in range(left_embeddings.size(0)):
            for j in range(right_embeddings.size(0)):
                try:
                    distances[i, j] = self.uhg.distance(
                        left_embeddings[i],
                        right_embeddings[j],
                    )
                except ValueError:
                    if self.require_backend:
                        raise
                    return (
                        torch.cdist(left_embeddings, right_embeddings, p=2),
                        "euclidean_cdist_fallback",
                        True,
                    )
        return distances, "projective_uhg_distance", False
