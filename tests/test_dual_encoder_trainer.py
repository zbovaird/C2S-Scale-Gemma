import torch
import torch.nn as nn

from fusion.align_losses import InfoNCELoss
from fusion.trainer import DualEncoderTrainer, FusionTrainer


class DummyTextModel(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return {"last_hidden_state": self.embedding(input_ids)}


class DummyGraphModel(nn.Module):
    def __init__(self, output_dim: int = 4):
        super().__init__()
        self.input_dim = output_dim
        self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, **kwargs):
        return {"euclidean_embeddings": self.linear(x)}


class DummyFusionHead(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, text_embeddings, graph_embeddings):
        return self.linear(torch.cat([text_embeddings, graph_embeddings], dim=-1))


class DummyHyperbolicGraphModel(nn.Module):
    def __init__(self, output_dim: int = 4):
        super().__init__()
        self.input_dim = output_dim
        self.hyperbolic_linear = nn.Linear(output_dim, output_dim)
        self.euclidean_linear = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, **kwargs):
        return {
            "hyperbolic_embeddings": self.hyperbolic_linear(x),
            "euclidean_embeddings": self.euclidean_linear(x),
        }


def test_dual_encoder_trainer_compute_loss_shapes():
    trainer = DualEncoderTrainer(
        hgnn_encoder=DummyGraphModel(),
        text_model=DummyTextModel(),
        fusion_head=DummyFusionHead(),
        contrastive_loss=InfoNCELoss(hard_negative_mining=False),
        device=torch.device("cpu"),
        config={"model": {"hgnn": {"input_dim": 4}}},
    )

    batch = {
        "input_ids": torch.randint(0, 16, (3, 5)),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
    }

    loss_dict = trainer.compute_loss(batch)

    assert set(
        ["total_loss", "contrastive_loss", "alignment_loss", "fusion_loss"]
    ).issubset(loss_dict)
    assert loss_dict["text_embeddings"].shape == (3, 4)
    assert loss_dict["graph_embeddings"].shape == (3, 4)
    assert loss_dict["fused_embeddings"].shape == (3, 4)


def test_fusion_trainer_alias_behaves_like_dual_encoder():
    trainer = FusionTrainer(
        hgnn_encoder=DummyGraphModel(),
        text_model=DummyTextModel(),
        fusion_head=DummyFusionHead(),
        contrastive_loss=InfoNCELoss(hard_negative_mining=False),
        device=torch.device("cpu"),
        config={"model": {"hgnn": {"input_dim": 4}}},
    )
    assert isinstance(trainer, DualEncoderTrainer)


def test_dual_encoder_trainer_uses_hyperbolic_graph_space_for_alignment():
    trainer = DualEncoderTrainer(
        hgnn_encoder=DummyHyperbolicGraphModel(),
        text_model=DummyTextModel(),
        fusion_head=DummyFusionHead(),
        contrastive_loss=InfoNCELoss(
            hard_negative_mining=False,
            alignment_mode="projective_distance",
            shared_dim=4,
            text_projection_type="linear",
        ),
        device=torch.device("cpu"),
        config={"model": {"hgnn": {"input_dim": 4}}},
    )

    batch = {
        "input_ids": torch.randint(0, 16, (3, 5)),
        "attention_mask": torch.ones(3, 5, dtype=torch.long),
    }

    loss_dict = trainer.compute_loss(batch)

    assert loss_dict["alignment_graph_embeddings"].shape == (3, 4)
    assert loss_dict["graph_embeddings"].shape == (3, 4)
    assert torch.isfinite(loss_dict["alignment_loss"])
