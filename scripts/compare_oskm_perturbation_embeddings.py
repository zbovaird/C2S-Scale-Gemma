#!/usr/bin/env python3
"""Compare baseline and perturbed datasets in representation space."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.collate import GraphTextCollator
from data.dataset import Cell2SentenceDataset
from eval.embedding_shift import build_embedding_shift_frame, summarize_embedding_shift
from eval.reprogramming_metrics import (
    DEFAULT_PLURIPOTENT_LABELS,
    DEFAULT_SOMATIC_LABELS,
    build_reprogramming_overlay_rows,
    summarize_branch_counts,
    summarize_zone_counts,
)
from eval.reprogramming_profiles import resolve_dataset_profile
from fusion.align_losses import InfoNCELoss
from fusion.trainer import DualEncoderTrainer
from fusion.heads import FusionHead
from hgnn.encoder import UHGHGNNEncoder
from text.adapters import LoRAAdapter
from text.gemma_loader import GemmaLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def load_models(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", config)

    hgnn_encoder = UHGHGNNEncoder(
        input_dim=checkpoint_config["model"]["hgnn"]["input_dim"],
        hidden_dim=checkpoint_config["model"]["hgnn"]["hidden_dim"],
        output_dim=checkpoint_config["model"]["hgnn"]["output_dim"],
        num_layers=checkpoint_config["model"]["hgnn"]["num_layers"],
        dropout=checkpoint_config["model"]["hgnn"]["dropout"],
        curvature=checkpoint_config["model"]["hgnn"].get("curvature", -1.0),
        device=device,
    )

    gemma_loader = GemmaLoader(
        model_name=checkpoint_config["model"]["text"]["model_name"],
        device=device,
        torch_dtype=torch.bfloat16,
        quantization_config=checkpoint_config["model"]["text"].get("quantization"),
    )
    gemma_model, tokenizer = gemma_loader.load_model()

    lora_adapter = LoRAAdapter(
        model=gemma_model,
        r=checkpoint_config["model"]["text"]["lora"]["r"],
        lora_alpha=checkpoint_config["model"]["text"]["lora"]["alpha"],
        lora_dropout=checkpoint_config["model"]["text"]["lora"]["dropout"],
        target_modules=checkpoint_config["model"]["text"]["lora"]["target_modules"],
    )

    fusion_head = FusionHead(
        graph_dim=checkpoint_config["model"]["hgnn"]["output_dim"],
        text_dim=checkpoint_config["model"]["text"]["hidden_size"],
        fusion_dim=checkpoint_config["model"]["fusion"]["dim"],
        dropout=checkpoint_config["model"]["fusion"]["dropout"],
    )

    trainer_state_dict = checkpoint["trainer_state_dict"]
    hgnn_encoder.load_state_dict(
        {
            key.replace("hgnn_encoder.", ""): value
            for key, value in trainer_state_dict.items()
            if key.startswith("hgnn_encoder.")
        },
        strict=False,
    )
    lora_adapter.load_state_dict(
        {
            key.replace("text_model.", ""): value
            for key, value in trainer_state_dict.items()
            if key.startswith("text_model.")
        },
        strict=False,
    )
    fusion_head.load_state_dict(
        {
            key.replace("fusion_head.", ""): value
            for key, value in trainer_state_dict.items()
            if key.startswith("fusion_head.")
        },
        strict=False,
    )

    return hgnn_encoder, lora_adapter, fusion_head, tokenizer


def create_dataset(
    data_path: str,
    tokenizer: Any,
    config: Dict[str, Any],
    device: torch.device,
) -> Cell2SentenceDataset:
    data_config = config.get("data", {})
    oskm_config = data_config.get("oskm", {})
    reprogramming_config = config.get("reprogramming", {})
    marker_panels = reprogramming_config.get("marker_panels", {})
    return Cell2SentenceDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config["model"]["text"]["max_length"],
        device=device,
        top_genes=data_config.get("top_genes", 1000),
        oskm_anchor_mode=oskm_config.get("anchor_mode", "none"),
        oskm_species=oskm_config.get("species", "human"),
        marker_panels=marker_panels,
    )


def create_dataloader(
    dataset: Cell2SentenceDataset,
    tokenizer: Any,
    batch_size: int,
) -> DataLoader:
    collator = GraphTextCollator(
        tokenizer=tokenizer,
        max_length=dataset.max_seq_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def _get_column_values(dataset: Cell2SentenceDataset, column: str, default_value: float | str):
    if column in dataset.cell_data:
        return dataset.cell_data[column].tolist()
    return [default_value] * len(dataset.cell_data)


def _parse_label_list(raw_value: str | None, default_values: tuple[str, ...]) -> list[str]:
    if raw_value is None:
        return list(default_values)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_reference_labels(
    config: Dict[str, Any],
    cli_value: str | None,
    config_key: str,
    defaults: tuple[str, ...],
) -> list[str]:
    if cli_value is not None:
        return _parse_label_list(cli_value, defaults)
    configured = config.get("reprogramming", {}).get("references", {}).get(config_key)
    if configured:
        return [str(value).strip() for value in configured if str(value).strip()]
    return list(defaults)


def _resolve_window_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("reprogramming", {}).get("window_profile", {}))


def run_embedding_comparison(
    *,
    config_path: str,
    checkpoint_path: str,
    baseline_data_path: str,
    perturbed_data_path: str,
    output_dir: str,
    batch_size: int = 8,
    dataset_profile: str | None = None,
    dataset_profile_config: str = "configs/reprogramming_profiles.toml",
    somatic_labels_arg: str | None = None,
    pluripotent_labels_arg: str | None = None,
) -> dict:
    config = load_config(config_path)
    config, dataset_manifest = resolve_dataset_profile(
        config,
        profile_name=dataset_profile,
        profile_config_path=dataset_profile_config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hgnn_encoder, lora_adapter, fusion_head, tokenizer = load_models(
        checkpoint_path,
        config,
        device,
    )
    trainer = DualEncoderTrainer(
        hgnn_encoder=hgnn_encoder,
        text_model=lora_adapter,
        fusion_head=fusion_head,
        contrastive_loss=InfoNCELoss(
            hard_negative_mining=False,
            alignment_mode=config.get("fusion", {}).get(
                "alignment_mode",
                "euclidean_cosine",
            ),
            shared_dim=config.get("fusion", {}).get("alignment_dim"),
            text_projection_type=config.get("fusion", {}).get(
                "text_projection_type",
                "learned",
            ),
        ),
        device=device,
        config=config,
    )
    trainer.eval()

    baseline_dataset = create_dataset(baseline_data_path, tokenizer, config, device)
    perturbed_dataset = create_dataset(perturbed_data_path, tokenizer, config, device)
    if len(baseline_dataset) != len(perturbed_dataset):
        raise ValueError("Baseline and perturbed datasets must contain the same number of cells.")
    baseline_loader = create_dataloader(baseline_dataset, tokenizer, batch_size)
    perturbed_loader = create_dataloader(perturbed_dataset, tokenizer, batch_size)

    baseline_embeddings = trainer.get_embeddings(baseline_loader)
    perturbed_embeddings = trainer.get_embeddings(perturbed_loader)

    comparison_summary = {
        representation: summarize_embedding_shift(
            baseline_embeddings[representation].cpu().numpy(),
            perturbed_embeddings[representation].cpu().numpy(),
        )
        for representation in ("text_embeddings", "graph_embeddings", "fused_embeddings")
        if representation in baseline_embeddings and representation in perturbed_embeddings
    }

    metadata = {
        "cell_id": baseline_dataset.cell_data["cell_id"].tolist(),
        "cell_type": baseline_dataset.cell_data["cell_type"].tolist(),
        "baseline_oskm_score": _get_column_values(baseline_dataset, "oskm_score", 0.0),
        "perturbed_oskm_score": _get_column_values(perturbed_dataset, "oskm_score", 0.0),
        "baseline_rejuvenation_score": _get_column_values(
            baseline_dataset,
            "rejuvenation_score",
            0.0,
        ),
        "perturbed_rejuvenation_score": _get_column_values(
            perturbed_dataset,
            "rejuvenation_score",
            0.0,
        ),
        "baseline_pluripotency_marker_score": _get_column_values(
            baseline_dataset,
            "pluripotency_risk_score",
            0.0,
        ),
        "perturbed_pluripotency_marker_score": _get_column_values(
            perturbed_dataset,
            "pluripotency_risk_score",
            0.0,
        ),
    }
    somatic_labels = _resolve_reference_labels(
        config, somatic_labels_arg, "somatic_labels", DEFAULT_SOMATIC_LABELS
    )
    pluripotent_labels = _resolve_reference_labels(
        config, pluripotent_labels_arg, "pluripotent_labels", DEFAULT_PLURIPOTENT_LABELS
    )
    window_profile = _resolve_window_profile(config)
    fused_shift_frame = build_embedding_shift_frame(
        baseline_embeddings["fused_embeddings"].cpu().numpy(),
        perturbed_embeddings["fused_embeddings"].cpu().numpy(),
        metadata=metadata,
    )
    overlay_rows = build_reprogramming_overlay_rows(
        baseline_embeddings["fused_embeddings"].cpu().numpy(),
        perturbed_embeddings["fused_embeddings"].cpu().numpy(),
        cell_types=metadata["cell_type"],
        baseline_oskm_scores=metadata["baseline_oskm_score"],
        perturbed_oskm_scores=metadata["perturbed_oskm_score"],
        baseline_rejuvenation_scores=metadata["baseline_rejuvenation_score"],
        perturbed_rejuvenation_scores=metadata["perturbed_rejuvenation_score"],
        baseline_pluripotency_marker_scores=metadata["baseline_pluripotency_marker_score"],
        perturbed_pluripotency_marker_scores=metadata["perturbed_pluripotency_marker_score"],
        somatic_labels=somatic_labels,
        pluripotent_labels=pluripotent_labels,
        window_profile=window_profile,
    )
    branch_summary = summarize_branch_counts(overlay_rows)
    zone_summary = summarize_zone_counts(overlay_rows)
    for row, overlay_row in zip(fused_shift_frame, overlay_rows):
        row.update(overlay_row)

    np.save(output_path / "baseline_text_embeddings.npy", baseline_embeddings["text_embeddings"].cpu().numpy())
    np.save(output_path / "baseline_graph_embeddings.npy", baseline_embeddings["graph_embeddings"].cpu().numpy())
    np.save(output_path / "baseline_fused_embeddings.npy", baseline_embeddings["fused_embeddings"].cpu().numpy())
    np.save(output_path / "perturbed_text_embeddings.npy", perturbed_embeddings["text_embeddings"].cpu().numpy())
    np.save(output_path / "perturbed_graph_embeddings.npy", perturbed_embeddings["graph_embeddings"].cpu().numpy())
    np.save(output_path / "perturbed_fused_embeddings.npy", perturbed_embeddings["fused_embeddings"].cpu().numpy())

    with (output_path / "embedding_shift_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison_summary, handle, indent=2)
    with (output_path / "fused_embedding_shift_frame.json").open("w", encoding="utf-8") as handle:
        json.dump(fused_shift_frame, handle, indent=2)
    overlay_summary = {
        "branch_summary": branch_summary,
        "zone_summary": zone_summary,
        "dataset_profile": config.get("reprogramming", {}).get("dataset_profile"),
        "dataset_manifest": dataset_manifest,
        "alignment": {
            "alignment_mode": config.get("fusion", {}).get(
                "alignment_mode",
                "euclidean_cosine",
            ),
            "alignment_dim": config.get("fusion", {}).get("alignment_dim"),
            "text_projection_type": config.get("fusion", {}).get(
                "text_projection_type",
                "learned",
            ),
        },
        "reference_labels": {
            "somatic_labels": somatic_labels,
            "pluripotent_labels": pluripotent_labels,
        },
        "window_profile": window_profile,
    }
    with (output_path / "reprogramming_overlay_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(overlay_summary, handle, indent=2)

    logger.info("Saved embedding comparison artifacts to %s", output_path)
    return {
        "output_dir": str(output_path),
        "embedding_summary": comparison_summary,
        "overlay_summary": overlay_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs perturbed embeddings")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--baseline-data-path", type=str, required=True, help="Baseline .h5ad path")
    parser.add_argument("--perturbed-data-path", type=str, required=True, help="Perturbed .h5ad path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/oskm_embedding_comparison",
        help="Directory for comparison outputs",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding extraction batch size")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default=None,
        help="Optional named dataset profile from configs/reprogramming_profiles.toml",
    )
    parser.add_argument(
        "--dataset-profile-config",
        type=str,
        default="configs/reprogramming_profiles.toml",
        help="Path to the dataset profile registry",
    )
    parser.add_argument(
        "--somatic-labels",
        type=str,
        default=None,
        help="Comma-separated cell_type labels defining the somatic reference",
    )
    parser.add_argument(
        "--pluripotent-labels",
        type=str,
        default=None,
        help="Comma-separated cell_type labels defining the pluripotent reference",
    )
    args = parser.parse_args()

    run_embedding_comparison(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        baseline_data_path=args.baseline_data_path,
        perturbed_data_path=args.perturbed_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dataset_profile=args.dataset_profile,
        dataset_profile_config=args.dataset_profile_config,
        somatic_labels_arg=args.somatic_labels,
        pluripotent_labels_arg=args.pluripotent_labels,
    )


if __name__ == "__main__":
    main()
