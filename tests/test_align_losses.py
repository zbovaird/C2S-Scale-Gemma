import torch

from fusion.align_losses import InfoNCELoss
from hgnn.manifold_ops import TangentSpaceLinear


def test_info_nce_accepts_hard_negative_weight():
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    graph = text.clone()
    labels = torch.tensor([0, 1])

    base_loss = InfoNCELoss(
        hard_negative_mining=True,
        hard_negative_weight=1.0,
    )(text, graph, labels=labels)
    weighted_loss = InfoNCELoss(
        hard_negative_mining=True,
        hard_negative_weight=2.0,
    )(text, graph, labels=labels)

    assert "total_loss" in base_loss
    assert weighted_loss["total_loss"].item() >= base_loss["total_loss"].item()


def test_info_nce_projective_distance_mode_returns_similarity_matrix():
    text = torch.tensor([[1.0, 0.2], [0.1, 1.0]], dtype=torch.float32)
    graph = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)

    loss_dict = InfoNCELoss(
        hard_negative_mining=False,
        alignment_mode="projective_distance",
        shared_dim=2,
        text_projection_type="linear",
    )(text, graph)

    assert loss_dict["alignment_mode"] == "projective_distance"
    assert loss_dict["primary_manifold"] == "projective_uhg"
    assert loss_dict["geometry_distance_backend"] in {
        "projective_uhg_distance",
        "euclidean_cdist_fallback",
    }
    assert loss_dict["similarity_matrix"].shape == (2, 2)
    assert torch.isfinite(loss_dict["total_loss"])


def test_info_nce_reports_euclidean_backend_for_cosine_mode():
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    graph = text.clone()

    loss_dict = InfoNCELoss(hard_negative_mining=False)(text, graph)

    assert loss_dict["geometry_distance_backend"] == "euclidean_cosine"
    assert loss_dict["geometry_fallback_used"] is False


def test_info_nce_uses_tangent_adapter_when_graph_dim_needs_projection():
    text = torch.tensor([[1.0, 0.2], [0.1, 1.0]], dtype=torch.float32)
    graph = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.4]], dtype=torch.float32)
    loss = InfoNCELoss(
        hard_negative_mining=False,
        alignment_mode="projective_distance",
        text_dim=2,
        graph_dim=3,
        shared_dim=2,
        text_projection_type="linear",
    )

    loss_dict = loss(text, graph)

    assert isinstance(loss.graph_to_geometry, TangentSpaceLinear)
    assert loss_dict["similarity_matrix"].shape == (2, 2)


def test_info_nce_strict_geometry_backend_raises_when_uhg_unavailable():
    text = torch.tensor([[1.0, 0.2], [0.1, 1.0]], dtype=torch.float32)
    graph = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)
    loss = InfoNCELoss(
        hard_negative_mining=False,
        alignment_mode="projective_distance",
        shared_dim=2,
        text_projection_type="linear",
        require_geometry_backend=True,
    )
    loss.uhg = None

    try:
        loss(text, graph)
    except RuntimeError as exc:
        assert "requires the projective UHG backend" in str(exc)
    else:
        raise AssertionError("Strict geometry backend should reject Euclidean fallback.")


def test_info_nce_rejects_unknown_primary_manifold():
    try:
        InfoNCELoss(
            hard_negative_mining=False,
            alignment_mode="projective_distance",
            primary_manifold="lorentz",
        )
    except ValueError as exc:
        assert "Unsupported primary manifold" in str(exc)
    else:
        raise AssertionError("Unknown primary manifolds should fail explicitly.")
