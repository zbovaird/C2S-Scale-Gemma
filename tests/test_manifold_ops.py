import torch

from hgnn.manifold_ops import TangentSpaceLinear


def test_tangent_space_linear_preserves_linear_shape_and_metadata():
    layer = TangentSpaceLinear(3, 2)
    x = torch.ones(4, 3)

    output = layer(x)

    assert output.shape == (4, 2)
    assert layer.geometry_role == "tangent_space_linear"
    assert layer.in_features == 3
    assert layer.out_features == 2
