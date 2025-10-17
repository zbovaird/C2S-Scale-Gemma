#!/usr/bin/env python3
"""
Script for aligning dual-encoder (HGNN + Gemma) with contrastive loss.

This script trains the dual-encoder model to align graph and text representations
using InfoNCE contrastive loss with hard negative mining.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import mlflow.pytorch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import Cell2SentenceDataset
from data.collate import GraphTextCollator
from hgnn.encoder import UHGHGNNEncoder
from text.gemma_loader import GemmaLoader
from text.adapters import LoRAAdapter
from text.pooling import AttentionPooling
from fusion.align_losses import InfoNCELoss
from fusion.heads import FusionHead
from fusion.trainer import DualEncoderTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/align_dual_encoder.log"),
        ],
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def create_data_loaders(
    config: Dict[str, Any], 
    tokenizer: Any,
    device: torch.device
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Create datasets
    train_dataset = Cell2SentenceDataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_length=config["model"]["text"]["max_length"],
        device=device,
    )
    
    val_dataset = Cell2SentenceDataset(
        data_path=config["data"]["val_path"],
        tokenizer=tokenizer,
        max_length=config["model"]["text"]["max_length"],
        device=device,
    )
    
    test_dataset = Cell2SentenceDataset(
        data_path=config["data"]["test_path"],
        tokenizer=tokenizer,
        max_length=config["model"]["text"]["max_length"],
        device=device,
    )
    
    # Create collator
    collator = GraphTextCollator(
        tokenizer=tokenizer,
        max_length=config["model"]["text"]["max_length"],
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def create_models(
    config: Dict[str, Any], 
    device: torch.device
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Create HGNN encoder, Gemma model, LoRA adapter, and fusion head."""
    
    # Create HGNN encoder
    hgnn_encoder = UHGHGNNEncoder(
        input_dim=config["model"]["hgnn"]["input_dim"],
        hidden_dim=config["model"]["hgnn"]["hidden_dim"],
        output_dim=config["model"]["hgnn"]["output_dim"],
        num_layers=config["model"]["hgnn"]["num_layers"],
        dropout=config["model"]["hgnn"]["dropout"],
        curvature=config["model"]["hgnn"]["curvature"],
        device=device,
    )
    
    # Create Gemma loader
    gemma_loader = GemmaLoader(
        model_name=config["model"]["text"]["model_name"],
        device=device,
        torch_dtype=torch.bfloat16,
        quantization_config=config["model"]["text"]["quantization"],
    )
    
    # Load Gemma model and tokenizer
    gemma_model, tokenizer = gemma_loader.load_model()
    
    # Create LoRA adapter
    lora_adapter = LoRAAdapter(
        model=gemma_model,
        r=config["model"]["text"]["lora"]["r"],
        lora_alpha=config["model"]["text"]["lora"]["alpha"],
        lora_dropout=config["model"]["text"]["lora"]["dropout"],
        target_modules=config["model"]["text"]["lora"]["target_modules"],
    )
    
    # Create fusion head
    fusion_head = FusionHead(
        graph_dim=config["model"]["hgnn"]["output_dim"],
        text_dim=config["model"]["text"]["hidden_size"],
        fusion_dim=config["model"]["fusion"]["dim"],
        dropout=config["model"]["fusion"]["dropout"],
    )
    
    return hgnn_encoder, lora_adapter, fusion_head, tokenizer


def create_optimizer_and_scheduler(
    models: tuple[nn.Module, nn.Module, nn.Module],
    config: Dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    
    hgnn_encoder, lora_adapter, fusion_head = models
    
    # Collect parameters
    params = []
    params.extend(hgnn_encoder.parameters())
    params.extend(lora_adapter.parameters())
    params.extend(fusion_head.parameters())
    
    # Create optimizer
    optimizer = AdamW(
        params,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config["training"]["min_lr"],
    )
    
    return optimizer, scheduler


def train_epoch(
    trainer: DualEncoderTrainer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train for one epoch."""
    
    trainer.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_fusion_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Forward pass
        loss_dict = trainer.compute_loss(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        
        # Gradient clipping
        if config["training"]["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                trainer.parameters(), 
                config["training"]["grad_clip_norm"]
            )
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss_dict["total_loss"].item()
        total_contrastive_loss += loss_dict["contrastive_loss"].item()
        total_fusion_loss += loss_dict["fusion_loss"].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss_dict['total_loss'].item():.4f}",
            "contrastive": f"{loss_dict['contrastive_loss'].item():.4f}",
            "fusion": f"{loss_dict['fusion_loss'].item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
        
        # Log to MLflow
        if batch_idx % config["training"]["log_interval"] == 0:
            mlflow.log_metrics({
                "train/loss": loss_dict["total_loss"].item(),
                "train/contrastive_loss": loss_dict["contrastive_loss"].item(),
                "train/fusion_loss": loss_dict["fusion_loss"].item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch * len(train_loader) + batch_idx)
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_contrastive_loss = total_contrastive_loss / num_batches
    avg_fusion_loss = total_fusion_loss / num_batches
    
    return {
        "loss": avg_loss,
        "contrastive_loss": avg_contrastive_loss,
        "fusion_loss": avg_fusion_loss,
    }


def validate(
    trainer: DualEncoderTrainer,
    val_loader: DataLoader,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model."""
    
    trainer.eval()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_fusion_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            loss_dict = trainer.compute_loss(batch)
            
            total_loss += loss_dict["total_loss"].item()
            total_contrastive_loss += loss_dict["contrastive_loss"].item()
            total_fusion_loss += loss_dict["fusion_loss"].item()
            num_batches += 1
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_contrastive_loss = total_contrastive_loss / num_batches
    avg_fusion_loss = total_fusion_loss / num_batches
    
    # Log to MLflow
    mlflow.log_metrics({
        "val/loss": avg_loss,
        "val/contrastive_loss": avg_contrastive_loss,
        "val/fusion_loss": avg_fusion_loss,
    }, step=epoch)
    
    return {
        "loss": avg_loss,
        "contrastive_loss": avg_contrastive_loss,
        "fusion_loss": avg_fusion_loss,
    }


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Align dual-encoder with contrastive loss")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="artifacts/align_dual_encoder", help="Output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file://" + str(output_dir / "mlruns"))
    mlflow.set_experiment("align_dual_encoder")
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(config)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config, None, device)
        
        # Create models
        logger.info("Creating models...")
        hgnn_encoder, lora_adapter, fusion_head, tokenizer = create_models(config, device)
        
        # Create trainer
        trainer = DualEncoderTrainer(
            hgnn_encoder=hgnn_encoder,
            text_model=lora_adapter,
            fusion_head=fusion_head,
            contrastive_loss=InfoNCELoss(
                temperature=config["training"]["contrastive_temperature"],
                hard_negative_weight=config["training"]["hard_negative_weight"],
            ),
            device=device,
        )
        
        # Create optimizer and scheduler
        total_steps = len(train_loader) * config["training"]["num_epochs"]
        optimizer, scheduler = create_optimizer_and_scheduler(
            (hgnn_encoder, lora_adapter, fusion_head),
            config,
            total_steps,
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location=device)
            trainer.load_state_dict(checkpoint["trainer_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(start_epoch, config["training"]["num_epochs"]):
            # Train
            train_metrics = train_epoch(
                trainer, train_loader, optimizer, scheduler, epoch, config
            )
            
            # Validate
            val_metrics = validate(trainer, val_loader, epoch)
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # Save checkpoint if best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = output_dir / f"best_checkpoint_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "trainer_state_dict": trainer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                }, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
            
            # Save regular checkpoint
            if epoch % config["training"]["save_interval"] == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    "epoch": epoch,
                    "trainer_state_dict": trainer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                }, checkpoint_path)
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = validate(trainer, test_loader, config["training"]["num_epochs"])
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        
        # Log final metrics
        mlflow.log_metrics({
            "test/loss": test_metrics["loss"],
            "test/contrastive_loss": test_metrics["contrastive_loss"],
            "test/fusion_loss": test_metrics["fusion_loss"],
        })
        
        # Save final model
        final_model_path = output_dir / "final_model.pt"
        torch.save({
            "trainer_state_dict": trainer.state_dict(),
            "config": config,
            "test_metrics": test_metrics,
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()