import torch

from fusion.align_losses import InfoNCELoss


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
