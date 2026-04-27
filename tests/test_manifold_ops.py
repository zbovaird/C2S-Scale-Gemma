from pathlib import Path

import torch

from hgnn.manifold_ops import (
    ProjectiveManifoldBackend,
    TangentSpaceLinear,
    normalize_primary_manifold,
)


def test_tangent_space_linear_preserves_linear_shape_and_metadata():
    layer = TangentSpaceLinear(3, 2)
    x = torch.ones(4, 3)

    output = layer(x)

    assert output.shape == (4, 2)
    assert layer.geometry_role == "tangent_space_linear"
    assert layer.in_features == 3
    assert layer.out_features == 2


def test_uhg_layers_source_uses_explicit_tangent_adapters_for_feature_maps():
    source = Path("src/hgnn/layers.py").read_text(encoding="utf-8")

    assert "self.self_linear = TangentSpaceLinear" in source
    assert "self.neighbor_linear = TangentSpaceLinear" in source
    assert "self.query_linear = TangentSpaceLinear" in source
    assert "self.output_linear = TangentSpaceLinear" in source
    assert "nn.Linear(" not in source


def test_projective_manifold_backend_reports_fallback_when_uhg_unavailable():
    backend = ProjectiveManifoldBackend(require_backend=False)
    backend.uhg = None

    distances, backend_name, fallback_used = backend.pairwise_distance(
        torch.zeros(2, 3),
        torch.ones(2, 3),
    )

    assert distances.shape == (2, 2)
    assert backend_name == "euclidean_cdist_fallback"
    assert fallback_used is True


def test_projective_manifold_backend_strict_mode_rejects_missing_uhg():
    backend = ProjectiveManifoldBackend(require_backend=True)
    backend.uhg = None

    try:
        backend.pairwise_distance(torch.zeros(1, 2), torch.zeros(1, 2))
    except RuntimeError as exc:
        assert "requires the projective UHG backend" in str(exc)
    else:
        raise AssertionError("Strict projective backend should reject fallback.")


def test_normalize_primary_manifold_rejects_unconfigured_manifold():
    assert normalize_primary_manifold("projective") == "projective_uhg"
    try:
        normalize_primary_manifold("lorentz")
    except ValueError as exc:
        assert "Unsupported primary manifold" in str(exc)
    else:
        raise AssertionError("Unsupported manifold names must be explicit failures.")
