"""Training utilities for the dual-encoder fusion stack."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DualEncoderTrainer(nn.Module):
    """Backward-compatible trainer module used by the repo scripts.

    The repo's scripts expect a PyTorch ``nn.Module`` that exposes
    ``compute_loss()``, ``parameters()``, ``state_dict()``, and embedding helper
    methods. Earlier refactors replaced that with a plain utility class, which
    broke the script entrypoints. This class restores the expected contract
    while keeping the implementation small and CPU-testable.
    """

    def __init__(
        self,
        hgnn_encoder: Optional[nn.Module] = None,
        text_model: Optional[nn.Module] = None,
        fusion_head: Optional[nn.Module] = None,
        contrastive_loss: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        radial_projector: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        text_encoder: Optional[nn.Module] = None,
        graph_encoder: Optional[nn.Module] = None,
        alignment_loss: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.text_model = text_model or text_encoder
        self.hgnn_encoder = hgnn_encoder or graph_encoder
        self.fusion_head = fusion_head
        self.contrastive_loss = contrastive_loss or alignment_loss
        self.radial_projector = radial_projector
        self.config = config or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.last_fusion_graph_source = "unknown"
        self.last_alignment_graph_source = "unknown"

        if self.hgnn_encoder is None or self.text_model is None:
            raise ValueError("Both graph and text models must be provided.")
        if self.fusion_head is None:
            raise ValueError("A fusion head must be provided.")
        if self.contrastive_loss is None:
            raise ValueError("A contrastive/alignment loss must be provided.")

        # Reuse the encoder's projector if it already owns one.
        if self.radial_projector is None:
            self.radial_projector = getattr(self.hgnn_encoder, "radial_projector", None)

        self.to(self.device)
        logger.info("Initialized DualEncoderTrainer on %s", self.device)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.compute_loss(batch)

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch = self._move_batch_to_device(batch)
        text_embeddings = self.get_text_representation(batch)
        graph_outputs = self._encode_graph_batch(batch)
        graph_embeddings = self._select_graph_embeddings_for_fusion(graph_outputs)
        alignment_graph_embeddings = self._select_graph_embeddings_for_alignment(graph_outputs)
        fused_embeddings = self.get_fused_representation(
            batch,
            precomputed_text=text_embeddings,
            precomputed_graph=graph_embeddings,
        )

        contrastive_loss_dict = self.contrastive_loss(
            text_embeddings,
            alignment_graph_embeddings,
        )
        contrastive_loss = contrastive_loss_dict.get(
            "total_loss",
            contrastive_loss_dict.get("contrastive_loss"),
        )
        if contrastive_loss is None:
            raise KeyError("Contrastive loss did not return a recognized loss key.")

        fusion_loss = torch.mean(torch.sum(fused_embeddings**2, dim=-1))
        total_loss = contrastive_loss + fusion_loss

        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "alignment_loss": contrastive_loss,
            "fusion_loss": fusion_loss,
            "text_embeddings": text_embeddings,
            "graph_embeddings": graph_embeddings,
            "alignment_graph_embeddings": alignment_graph_embeddings,
            "fusion_graph_source": self.last_fusion_graph_source,
            "alignment_graph_source": self.last_alignment_graph_source,
            "fused_embeddings": fused_embeddings,
        }

    def get_text_representation(self, batch: Dict[str, Any]) -> torch.Tensor:
        model_outputs = self._run_text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden_states = self._extract_text_hidden_states(model_outputs)
        return self._pool_text_embeddings(hidden_states, batch["attention_mask"])

    def get_graph_representation(self, batch: Dict[str, Any]) -> torch.Tensor:
        graph_outputs = self._encode_graph_batch(batch)
        return self._select_graph_embeddings_for_fusion(graph_outputs)

    def get_fused_representation(
        self,
        batch: Dict[str, Any],
        precomputed_text: Optional[torch.Tensor] = None,
        precomputed_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_embeddings = (
            precomputed_text
            if precomputed_text is not None
            else self.get_text_representation(batch)
        )
        graph_embeddings = (
            precomputed_graph
            if precomputed_graph is not None
            else self.get_graph_representation(batch)
        )
        return self.fusion_head(text_embeddings, graph_embeddings)

    def get_embeddings(
        self,
        dataloader: DataLoader,
        return_fused: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.eval()

        all_text_embeddings = []
        all_graph_embeddings = []
        all_alignment_graph_embeddings = []
        all_fused_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                text_embeddings = self.get_text_representation(batch)
                graph_outputs = self._encode_graph_batch(batch)
                graph_embeddings = self._select_graph_embeddings_for_fusion(graph_outputs)
                alignment_graph_embeddings = self._select_graph_embeddings_for_alignment(
                    graph_outputs
                )
                all_text_embeddings.append(text_embeddings)
                all_graph_embeddings.append(graph_embeddings)
                all_alignment_graph_embeddings.append(alignment_graph_embeddings)

                if return_fused:
                    all_fused_embeddings.append(
                        self.get_fused_representation(
                            batch,
                            precomputed_text=text_embeddings,
                            precomputed_graph=graph_embeddings,
                        )
                    )

        result = {
            "text_embeddings": torch.cat(all_text_embeddings, dim=0),
            "graph_embeddings": torch.cat(all_graph_embeddings, dim=0),
            "alignment_graph_embeddings": torch.cat(all_alignment_graph_embeddings, dim=0),
            "fusion_graph_source": self.last_fusion_graph_source,
            "alignment_graph_source": self.last_alignment_graph_source,
        }
        if return_fused and all_fused_embeddings:
            result["fused_embeddings"] = torch.cat(all_fused_embeddings, dim=0)
        return result

    def _run_text_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Any:
        common_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Hugging Face models prefer output_hidden_states/return_dict.
        for extra_kwargs in (
            {"output_hidden_states": True, "return_dict": True},
            {"return_hidden_states": True},
            {},
        ):
            try:
                return self.text_model(**common_kwargs, **extra_kwargs)
            except TypeError:
                continue
        return self.text_model(**common_kwargs)

    def _encode_graph_batch(self, batch: Dict[str, Any]) -> Any:
        batch_size = batch["input_ids"].size(0)
        input_dim = (
            self.config.get("model", {})
            .get("hgnn", {})
            .get("input_dim", getattr(self.hgnn_encoder, "input_dim", 64))
        )
        node_features = torch.randn(batch_size, input_dim, device=self.device)
        edge_index = self._create_edge_index(batch_size)
        return self.hgnn_encoder(node_features, edge_index)

    def _select_graph_embeddings_for_fusion(self, graph_outputs: Any) -> torch.Tensor:
        if isinstance(graph_outputs, dict):
            if graph_outputs.get("euclidean_embeddings") is not None:
                self.last_fusion_graph_source = "euclidean_embeddings"
                return graph_outputs["euclidean_embeddings"]
            if graph_outputs.get("graph_embeddings") is not None:
                self.last_fusion_graph_source = "graph_embeddings"
                return graph_outputs["graph_embeddings"]
            graph_embeddings = self._extract_hyperbolic_graph_embeddings(
                graph_outputs,
                source_attr="last_fusion_graph_source",
            )
        else:
            graph_embeddings = graph_outputs
            self.last_fusion_graph_source = "direct_graph_outputs"

        if self.radial_projector is not None:
            self.last_fusion_graph_source = f"{self.last_fusion_graph_source}:radial_projected"
            return self.radial_projector(graph_embeddings)
        return graph_embeddings

    def _select_graph_embeddings_for_alignment(self, graph_outputs: Any) -> torch.Tensor:
        if getattr(self.contrastive_loss, "requires_hyperbolic_graph_space", False):
            if isinstance(graph_outputs, dict):
                return self._extract_hyperbolic_graph_embeddings(
                    graph_outputs,
                    fallback_to_projected=True,
                    source_attr="last_alignment_graph_source",
                )
            self.last_alignment_graph_source = "direct_graph_outputs"
            return graph_outputs
        alignment_embeddings = self._select_graph_embeddings_for_fusion(graph_outputs)
        self.last_alignment_graph_source = self.last_fusion_graph_source
        return alignment_embeddings

    def _extract_hyperbolic_graph_embeddings(
        self,
        graph_outputs: Dict[str, torch.Tensor],
        fallback_to_projected: bool = False,
        source_attr: str | None = None,
    ) -> torch.Tensor:
        if graph_outputs.get("hyperbolic_embeddings") is not None:
            if source_attr:
                setattr(self, source_attr, "hyperbolic_embeddings")
            return graph_outputs["hyperbolic_embeddings"]
        if graph_outputs.get("node_embeddings") is not None:
            if source_attr:
                setattr(self, source_attr, "node_embeddings")
            return graph_outputs["node_embeddings"]
        if fallback_to_projected:
            if graph_outputs.get("euclidean_embeddings") is not None:
                if source_attr:
                    setattr(self, source_attr, "euclidean_embeddings:fallback")
                return graph_outputs["euclidean_embeddings"]
            if graph_outputs.get("graph_embeddings") is not None:
                if source_attr:
                    setattr(self, source_attr, "graph_embeddings:fallback")
                return graph_outputs["graph_embeddings"]
        raise KeyError("Graph encoder outputs missing requested embedding tensors.")

    def _extract_text_hidden_states(self, model_outputs: Any) -> torch.Tensor:
        if isinstance(model_outputs, torch.Tensor):
            return model_outputs
        if isinstance(model_outputs, dict):
            if "last_hidden_state" in model_outputs:
                return model_outputs["last_hidden_state"]
            if "hidden_states" in model_outputs and model_outputs["hidden_states"]:
                return model_outputs["hidden_states"][-1]
            if "logits" in model_outputs:
                return model_outputs["logits"]
        if hasattr(model_outputs, "last_hidden_state"):
            return model_outputs.last_hidden_state
        if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states:
            return model_outputs.hidden_states[-1]
        if hasattr(model_outputs, "logits"):
            return model_outputs.logits
        raise TypeError("Unsupported text model output type.")

    def _pool_text_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        masked_hidden_states = hidden_states * attention_mask
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        seq_lengths = torch.sum(attention_mask, dim=1).clamp_min(1e-8)
        return sum_hidden_states / seq_lengths

    def _create_edge_index(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        edges = [
            [source, target]
            for source in range(batch_size)
            for target in range(batch_size)
            if source != target
        ]
        return torch.tensor(edges, dtype=torch.long, device=self.device).t()

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }


class FusionTrainer(DualEncoderTrainer):
    """Alias retained for compatibility with newer imports."""

