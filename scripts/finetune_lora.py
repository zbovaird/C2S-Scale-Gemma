#!/usr/bin/env python3
"""
Script for fine-tuning the fused model with LoRA adapters.

This script fine-tunes the aligned dual-encoder model on downstream tasks
using LoRA adapters for parameter-efficient fine-tuning.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
            logging.FileHandler("logs/finetune_lora.log"),
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


def load_pretrained_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    """Load pretrained dual-encoder model."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    
    # Create HGNN encoder
    hgnn_encoder = UHGHGNNEncoder(
        input_dim=checkpoint_config["model"]["hgnn"]["input_dim"],
        hidden_dim=checkpoint_config["model"]["hgnn"]["hidden_dim"],
        output_dim=checkpoint_config["model"]["hgnn"]["output_dim"],
        num_layers=checkpoint_config["model"]["hgnn"]["num_layers"],
        dropout=checkpoint_config["model"]["hgnn"]["dropout"],
        curvature=checkpoint_config["model"]["hgnn"]["curvature"],
        device=device,
    )
    
    # Create Gemma loader
    gemma_loader = GemmaLoader(
        model_name=checkpoint_config["model"]["text"]["model_name"],
        device=device,
        torch_dtype=torch.bfloat16,
        quantization_config=checkpoint_config["model"]["text"]["quantization"],
    )
    
    # Load Gemma model and tokenizer
    gemma_model, tokenizer = gemma_loader.load_model()
    
    # Create LoRA adapter
    lora_adapter = LoRAAdapter(
        model=gemma_model,
        r=checkpoint_config["model"]["text"]["lora"]["r"],
        lora_alpha=checkpoint_config["model"]["text"]["lora"]["alpha"],
        lora_dropout=checkpoint_config["model"]["text"]["lora"]["dropout"],
        target_modules=checkpoint_config["model"]["text"]["lora"]["target_modules"],
    )
    
    # Create fusion head
    fusion_head = FusionHead(
        graph_dim=checkpoint_config["model"]["hgnn"]["output_dim"],
        text_dim=checkpoint_config["model"]["text"]["hidden_size"],
        fusion_dim=checkpoint_config["model"]["fusion"]["dim"],
        dropout=checkpoint_config["model"]["fusion"]["dropout"],
    )
    
    # Load pretrained weights
    trainer_state_dict = checkpoint["trainer_state_dict"]
    hgnn_encoder.load_state_dict({
        k.replace("hgnn_encoder.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("hgnn_encoder.")
    })
    lora_adapter.load_state_dict({
        k.replace("text_model.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("text_model.")
    })
    fusion_head.load_state_dict({
        k.replace("fusion_head.", ""): v 
        for k, v in trainer_state_dict.items() 
        if k.startswith("fusion_head.")
    })
    
    return hgnn_encoder, lora_adapter, fusion_head, tokenizer


def create_task_head(
    task_type: str,
    input_dim: int,
    num_classes: int,
    dropout: float = 0.1
) -> nn.Module:
    """Create task-specific head."""
    
    if task_type == "classification":
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )
    elif task_type == "regression":
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )
    elif task_type == "multilabel":
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
            nn.Sigmoid(),
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def create_optimizer_and_scheduler(
    models: List[nn.Module],
    config: Dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    
    # Collect parameters
    params = []
    for model in models:
        params.extend(model.parameters())
    
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


def compute_task_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str
) -> torch.Tensor:
    """Compute task-specific loss."""
    
    if task_type == "classification":
        return nn.CrossEntropyLoss()(outputs, targets)
    elif task_type == "regression":
        return nn.MSELoss()(outputs.squeeze(), targets.float())
    elif task_type == "multilabel":
        return nn.BCELoss()(outputs, targets.float())
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_task_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str
) -> Dict[str, float]:
    """Compute task-specific metrics."""
    
    metrics = {}
    
    if task_type == "classification":
        predictions = torch.argmax(outputs, dim=1)
        metrics["accuracy"] = accuracy_score(targets.cpu(), predictions.cpu())
        metrics["f1"] = f1_score(targets.cpu(), predictions.cpu(), average="weighted")
        
        # Compute ROC-AUC for binary classification
        if outputs.shape[1] == 2:
            probs = torch.softmax(outputs, dim=1)[:, 1]
            metrics["roc_auc"] = roc_auc_score(targets.cpu(), probs.cpu())
    
    elif task_type == "regression":
        predictions = outputs.squeeze()
        mse = nn.MSELoss()(predictions, targets.float())
        metrics["mse"] = mse.item()
        metrics["rmse"] = torch.sqrt(mse).item()
        
        # Compute R²
        ss_res = torch.sum((targets.float() - predictions) ** 2)
        ss_tot = torch.sum((targets.float() - torch.mean(targets.float())) ** 2)
        metrics["r2"] = 1 - (ss_res / ss_tot).item()
    
    elif task_type == "multilabel":
        predictions = (outputs > 0.5).float()
        metrics["accuracy"] = accuracy_score(targets.cpu(), predictions.cpu())
        metrics["f1"] = f1_score(targets.cpu(), predictions.cpu(), average="weighted")
    
    return metrics


def train_epoch(
    trainer: DualEncoderTrainer,
    task_head: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    config: Dict[str, Any],
    task_type: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    
    trainer.train()
    task_head.train()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_metrics = {}
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Forward pass
        with torch.no_grad():
            # Get fused representations
            fused_repr = trainer.get_fused_representation(batch)
        
        # Task-specific forward pass
        task_outputs = task_head(fused_repr)
        
        # Compute loss
        task_loss = compute_task_loss(task_outputs, batch["labels"], task_type)
        
        # Backward pass
        optimizer.zero_grad()
        task_loss.backward()
        
        # Gradient clipping
        if config["training"]["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                list(trainer.parameters()) + list(task_head.parameters()), 
                config["training"]["grad_clip_norm"]
            )
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += task_loss.item()
        total_task_loss += task_loss.item()
        num_batches += 1
        
        # Compute metrics
        batch_metrics = compute_task_metrics(task_outputs, batch["labels"], task_type)
        for key, value in batch_metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{task_loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })
        
        # Log to MLflow
        if batch_idx % config["training"]["log_interval"] == 0:
            mlflow.log_metrics({
                "train/task_loss": task_loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch * len(train_loader) + batch_idx)
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_task_loss = total_task_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    return {
        "loss": avg_loss,
        "task_loss": avg_task_loss,
        **avg_metrics,
    }


def validate(
    trainer: DualEncoderTrainer,
    task_head: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    task_type: str,
) -> Dict[str, float]:
    """Validate the model."""
    
    trainer.eval()
    task_head.eval()
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Forward pass
            fused_repr = trainer.get_fused_representation(batch)
            task_outputs = task_head(fused_repr)
            
            # Compute loss
            task_loss = compute_task_loss(task_outputs, batch["labels"], task_type)
            
            total_loss += task_loss.item()
            total_task_loss += task_loss.item()
            num_batches += 1
            
            # Compute metrics
            batch_metrics = compute_task_metrics(task_outputs, batch["labels"], task_type)
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_task_loss = total_task_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    # Log to MLflow
    mlflow.log_metrics({
        "val/task_loss": avg_task_loss,
        **{f"val/{key}": value for key, value in avg_metrics.items()},
    }, step=epoch)
    
    return {
        "loss": avg_loss,
        "task_loss": avg_task_loss,
        **avg_metrics,
    }


def main():
    """Main fine-tuning function."""
    
    parser = argparse.ArgumentParser(description="Fine-tune fused model with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="artifacts/finetune_lora", help="Output directory")
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
    mlflow.set_experiment("finetune_lora")
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(config)
        
        # Load pretrained model
        logger.info("Loading pretrained model...")
        hgnn_encoder, lora_adapter, fusion_head, tokenizer = load_pretrained_model(
            args.checkpoint, config, device
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config, tokenizer, device)
        
        # Create task head
        task_type = config["task"]["type"]
        num_classes = config["task"]["num_classes"]
        task_head = create_task_head(
            task_type=task_type,
            input_dim=config["model"]["fusion"]["dim"],
            num_classes=num_classes,
            dropout=config["model"]["fusion"]["dropout"],
        ).to(device)
        
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
            [trainer, task_head],
            config,
            total_steps,
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location=device)
            trainer.load_state_dict(checkpoint["trainer_state_dict"])
            task_head.load_state_dict(checkpoint["task_head_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        logger.info("Starting fine-tuning...")
        best_val_loss = float("inf")
        
        for epoch in range(start_epoch, config["training"]["num_epochs"]):
            # Train
            train_metrics = train_epoch(
                trainer, task_head, train_loader, optimizer, scheduler, epoch, config, task_type
            )
            
            # Validate
            val_metrics = validate(trainer, task_head, val_loader, epoch, task_type)
            
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
                    "task_head_state_dict": task_head.state_dict(),
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
                    "task_head_state_dict": task_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                }, checkpoint_path)
        
        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_metrics = validate(trainer, task_head, test_loader, config["training"]["num_epochs"], task_type)
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        
        # Log final metrics
        mlflow.log_metrics({
            "test/task_loss": test_metrics["task_loss"],
            **{f"test/{key}": value for key, value in test_metrics.items() if key != "task_loss"},
        })
        
        # Save final model
        final_model_path = output_dir / "final_model.pt"
        torch.save({
            "trainer_state_dict": trainer.state_dict(),
            "task_head_state_dict": task_head.state_dict(),
            "config": config,
            "test_metrics": test_metrics,
        }, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()