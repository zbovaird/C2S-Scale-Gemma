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
