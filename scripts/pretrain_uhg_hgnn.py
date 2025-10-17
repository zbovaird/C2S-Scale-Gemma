#!/usr/bin/env python3
"""
UHG-HGNN Pretraining Script for C2S-Scale-Gemma Hybrid Model.

This script performs self-supervised pretraining of the UHG-HGNN encoder
using contrastive learning in hyperbolic space.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hgnn.uhg_hgnn_encoder import create_uhg_hgnn_encoder, create_colab_hgnn_encoder, create_vertex_hgnn_encoder
from data.dataset import CellSentenceDataset, GraphDataset, PairedDataset
from data.collate import GraphTextCollator
from graphs.build_knn import build_knn_graph
from graphs.build_lr_bipartite import build_lr_bipartite_graph
from graphs.build_grn import build_grn_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UHGHGNNTrainer:
    """
    Trainer for UHG-HGNN encoder with self-supervised contrastive learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        use_mlflow: bool = True,
        experiment_name: str = "uhg-hgnn-pretraining"
    ):
        """
        Initialize UHG-HGNN trainer.
        
        Args:
            model: UHG-HGNN encoder model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            use_mlflow: Whether to use MLflow for logging
            experiment_name: MLflow experiment name
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler (will be set up in training loop)
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # MLflow setup
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run()
            logger.info(f"Started MLflow run: {self.run.info.run_id}")
        
        logger.info(f"Initialized UHG-HGNN trainer on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler."""
        from transformers import get_linear_schedule_with_warmup
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Setup scheduler: {self.warmup_steps} warmup steps, {total_steps} total steps")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        gradient_accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        self.epoch = epoch
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_monotonicity_loss = 0.0
        total_angular_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                x=batch['node_features'],
                edge_index=batch['edge_index'],
                edge_weight=batch.get('edge_weight'),
                batch=batch.get('batch'),
                return_projections=True
            )
            
            # Get hyperbolic embeddings for contrastive loss
            hyperbolic_embeddings = outputs['hyperbolic_embeddings']
            
            # Compute contrastive loss
            contrastive_loss_dict = self.model.compute_contrastive_loss(
                embeddings=hyperbolic_embeddings,
                labels=batch['labels']
            )
            
            # Compute projection losses
            monotonicity_loss = self.model.get_monotonicity_loss(batch['node_features'])
            angular_loss = self.model.get_angular_preservation_loss(batch['node_features'])
            
            # Total loss
            loss = (
                contrastive_loss_dict['total_loss'] +
                0.1 * monotonicity_loss +
                0.1 * angular_loss
            )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Update global step
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * gradient_accumulation_steps
            total_contrastive_loss += contrastive_loss_dict['contrastive_loss'].item()
            total_monotonicity_loss += monotonicity_loss.item()
            total_angular_loss += angular_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'contrastive': f"{contrastive_loss_dict['contrastive_loss'].item():.4f}",
                'monotonicity': f"{monotonicity_loss.item():.4f}",
                'angular': f"{angular_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'step': self.global_step
            })
            
            # Log to MLflow
            if self.use_mlflow and self.global_step % 10 == 0:
                mlflow.log_metrics({
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/contrastive_loss': contrastive_loss_dict['contrastive_loss'].item(),
                    'train/monotonicity_loss': monotonicity_loss.item(),
                    'train/angular_loss': angular_loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_monotonicity_loss = total_monotonicity_loss / num_batches
        avg_angular_loss = total_angular_loss / num_batches
        
        metrics = {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'monotonicity_loss': avg_monotonicity_loss,
            'angular_loss': avg_angular_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_monotonicity_loss = 0.0
        total_angular_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    x=batch['node_features'],
                    edge_index=batch['edge_index'],
                    edge_weight=batch.get('edge_weight'),
                    batch=batch.get('batch'),
                    return_projections=True
                )
                
                # Get hyperbolic embeddings for contrastive loss
                hyperbolic_embeddings = outputs['hyperbolic_embeddings']
                
                # Compute contrastive loss
                contrastive_loss_dict = self.model.compute_contrastive_loss(
                    embeddings=hyperbolic_embeddings,
                    labels=batch['labels']
                )
                
                # Compute projection losses
                monotonicity_loss = self.model.get_monotonicity_loss(batch['node_features'])
                angular_loss = self.model.get_angular_preservation_loss(batch['node_features'])
                
                # Total loss
                loss = (
                    contrastive_loss_dict['total_loss'] +
                    0.1 * monotonicity_loss +
                    0.1 * angular_loss
                )
                
                # Update metrics
                total_loss += loss.item()
                total_contrastive_loss += contrastive_loss_dict['contrastive_loss'].item()
                total_monotonicity_loss += monotonicity_loss.item()
                total_angular_loss += angular_loss.item()
                num_batches += 1
        
        # Calculate average metrics
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_monotonicity_loss = total_monotonicity_loss / num_batches
        avg_angular_loss = total_angular_loss / num_batches
        
        metrics = {
            'val_loss': avg_loss,
            'val_contrastive_loss': avg_contrastive_loss,
            'val_monotonicity_loss': avg_monotonicity_loss,
            'val_angular_loss': avg_angular_loss
        }
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, save_path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'metrics': metrics,
            'model_info': self.model.get_model_info()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        # Log to MLflow
        if self.use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_artifact(save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def create_dummy_graph_data(num_nodes: int = 1000, num_features: int = 2000) -> Dict[str, torch.Tensor]:
    """Create dummy graph data for testing."""
    # Create random node features
    node_features = torch.randn(num_nodes, num_features)
    
    # Create random edge indices (kNN-like graph)
    k = 15
    edge_list = []
    for i in range(num_nodes):
        # Random k neighbors
        neighbors = torch.randperm(num_nodes)[:k]
        for neighbor in neighbors:
            if neighbor != i:
                edge_list.append([i, neighbor])
    
    edge_index = torch.tensor(edge_list).T
    
    # Create random edge weights
    edge_weight = torch.rand(edge_index.size(1))
    
    # Create random labels
    labels = torch.randint(0, 10, (num_nodes,))
    
    # Create batch assignment (each node is its own graph for now)
    batch = torch.arange(num_nodes)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'labels': labels,
        'batch': batch
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train UHG-HGNN encoder")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--use_dummy_data", action="store_true", help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create model
    if config.model.encoder_type == "colab":
        model = create_colab_hgnn_encoder(**config.model)
    elif config.model.encoder_type == "vertex":
        model = create_vertex_hgnn_encoder(**config.model)
    else:
        model = create_uhg_hgnn_encoder(**config.model)
    
    # Create trainer
    trainer = UHGHGNNTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_mlflow=config.training.use_mlflow,
        experiment_name=config.training.experiment_name
    )
    
    # Create data loaders
    if args.use_dummy_data:
        logger.info("Using dummy data for testing")
        
        # Create dummy data
        train_data = create_dummy_graph_data(
            num_nodes=config.data.train_size,
            num_features=config.model.input_dim
        )
        val_data = create_dummy_graph_data(
            num_nodes=config.data.val_size,
            num_features=config.model.input_dim
        )
        
        # Create data loaders
        train_loader = DataLoader(
            [train_data] * config.training.batch_size,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0]
        )
        val_loader = DataLoader(
            [val_data] * config.training.batch_size,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0]
        )
    else:
        # TODO: Implement real data loading
        raise NotImplementedError("Real data loading not implemented yet")
    
    # Setup scheduler
    total_steps = len(train_loader) * config.training.num_epochs
    trainer.setup_scheduler(total_steps)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    logger.info(f"Starting training for {config.training.num_epochs} epochs")
    logger.info(f"Total steps: {total_steps:,}")
    
    for epoch in range(trainer.epoch, config.training.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps
        )
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"Epoch Time: {epoch_time:.1f}s")
        
        # Log to MLflow
        if trainer.use_mlflow:
            mlflow.log_metrics({
                **train_metrics,
                **val_metrics,
                'epoch_time': epoch_time
            }, step=trainer.global_step)
        
        # Save checkpoint
        if val_metrics['val_loss'] < trainer.best_loss:
            trainer.best_loss = val_metrics['val_loss']
            checkpoint_path = output_dir / f"best_model_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
        
        # Save regular checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)
    
    # Save final model
    final_checkpoint_path = output_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_checkpoint_path), epoch, val_metrics)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
