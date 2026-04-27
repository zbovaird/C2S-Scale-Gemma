"""Explicit manifold-boundary operations for HGNN modules."""

from __future__ import annotations

import torch
import torch.nn as nn


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
