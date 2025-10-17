#!/usr/bin/env python3
"""
HGNN pretraining script for C2S-Scale-Gemma Hybrid project.

Performs self-supervised pretraining of the UHG Graph Neural Network encoder.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from src.hgnn.encoder import create_uhg_encoder
from src.hgnn.losses import create_uhg_loss
from src.data.dataset import GraphDataset
from src.data.collate import collate_graph_batch
from src.uhg_adapters.sampling import create_sampler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HGNNPretrainer:
    """
    HGNN pretraining trainer.
    
    Handles self-supervised pretraining of UHG Graph Neural Network encoder.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize HGNN pretrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.encoder = self._create_encoder()
        self.loss_fn = self._create_loss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Move to device
        self.encoder.to(self.device)
        
        # Initialize sampling
        self.sampler = create_sampler(
            sampler_type=config.get('sampling_strategy', 'radial_band'),
            num_bands=config.get('num_bands', 5),
            band_size=config.get('band_size', 20)
        )
        
        logger.info(f"Initialized HGNN pretrainer on {self.device}")
    
    def _create_encoder(self) -> nn.Module:
        """Create UHG encoder."""
        encoder_config = self.config['model']['hgnn']
        
        encoder = create_uhg_encoder(
            encoder_type=encoder_config.get('encoder_type', 'single_scale'),
            input_dim=encoder_config.get('input_dim', 64),
            hidden_dim=encoder_config.get('hidden_dim', 256),
            output_dim=encoder_config.get('output_dim', 128),
            num_layers=encoder_config.get('num_layers', 3),
            layer_type=encoder_config.get('layer_type', 'graphsage'),
            dropout=encoder_config.get('dropout', 0.1),
            use_uhg_norm=encoder_config.get('use_uhg_norm', True)
        )
        
        return encoder
    
    def _create_loss(self) -> nn.Module:
        """Create UHG loss function."""
        loss_config = self.config['model']['hgnn']
        
        loss_fn = create_uhg_loss(
            loss_type=loss_config.get('loss_type', 'contrastive'),
            temperature=loss_config.get('contrastive_tau', 0.07),
            margin=loss_config.get('margin', 1.0),
            hard_negative_mining=loss_config.get('hard_negative_mining', True)
        )
        
        return loss_fn
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['train']
        
        optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=train_config.get('lr', 2e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        train_config = self.config['train']
        
        if train_config.get('use_scheduler', True):
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('epochs', 100),
                eta_min=train_config.get('min_lr', 1e-6)
            )
            return scheduler
        
        return None
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.encoder.train()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_hard_negative_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss_dict = self._forward_pass(batch)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            if self.config['train'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(),
                    self.config['train']['grad_clip']
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_contrastive_loss += loss_dict['contrastive_loss'].item()
            total_hard_negative_loss += loss_dict['hard_negative_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'contrastive': f"{loss_dict['contrastive_loss'].item():.4f}",
                'hard_neg': f"{loss_dict['hard_negative_loss'].item():.4f}"
            })
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_hard_negative_loss = total_hard_negative_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_contrastive_loss': avg_contrastive_loss,
            'train_hard_negative_loss': avg_hard_negative_loss
        }
    
    def _forward_pass(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary with loss components
        """
        # Extract data
        cell_ids = batch['cell_ids']
        cell_types = batch['cell_types']
        tissues = batch['tissues']
        
        # Get graph data
        knn_neighbors = batch.get('knn_neighbors', None)
        lr_interactions = batch.get('lr_interactions', None)
        grn_edges = batch.get('grn_edges', None)
        
        # Create node features (simplified - in practice, you'd use actual features)
        batch_size = len(cell_ids)
        input_dim = self.config['model']['hgnn'].get('input_dim', 64)
        node_features = torch.randn(batch_size, input_dim, device=self.device)
        
        # Create edge indices (simplified - in practice, you'd use actual graph structure)
        edge_index = self._create_edge_index(batch_size)
        
        # Encode nodes
        encoder_outputs = self.encoder(node_features, edge_index)
        node_embeddings = encoder_outputs['node_embeddings']
        
        # Create labels for contrastive learning
        labels = self._create_labels(cell_types, tissues)
        
        # Compute loss
        loss_dict = self.loss_fn(node_embeddings, labels)
        
        return loss_dict
    
    def _create_edge_index(self, batch_size: int) -> torch.Tensor:
        """
        Create edge indices for the batch.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Edge indices [2, E]
        """
        # Create a simple fully connected graph for each sample
        edges = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    edges.append([i, j])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return edge_index
    
    def _create_labels(self, cell_types: list, tissues: list) -> torch.Tensor:
        """
        Create labels for contrastive learning.
        
        Args:
            cell_types: List of cell types
            tissues: List of tissues
            
        Returns:
            Labels tensor
        """
        # Create labels based on cell type and tissue
        labels = []
        for i, (cell_type, tissue) in enumerate(zip(cell_types, tissues)):
            # Combine cell type and tissue for label
            label = f"{cell_type}_{tissue}"
            labels.append(label)
        
        # Convert to tensor
        unique_labels = list(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        label_indices = [label_to_idx[label] for label in labels]
        labels_tensor = torch.tensor(label_indices, device=self.device)
        
        return labels_tensor
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.encoder.eval()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_hard_negative_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                loss_dict = self._forward_pass(batch)
                
                # Accumulate losses
                total_loss += loss_dict['total_loss'].item()
                total_contrastive_loss += loss_dict['contrastive_loss'].item()
                total_hard_negative_loss += loss_dict['hard_negative_loss'].item()
                num_batches += 1
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_hard_negative_loss = total_hard_negative_loss / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_contrastive_loss': avg_contrastive_loss,
            'val_hard_negative_loss': avg_hard_negative_loss
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], checkpoint_dir: Path):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            checkpoint_dir: Checkpoint directory
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Pretrain HGNN encoder")
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="configs/colab_7b.toml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to graph data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/hgnn_pretrain",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.cfg):
        cfg = OmegaConf.load(args.cfg)
        logger.info(f"Loaded configuration from {args.cfg}")
    else:
        logger.error(f"Configuration file {args.cfg} not found")
        return
    
    # Override configuration with command line arguments
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.get('logging', {}).get('mlflow_tracking_uri', 'file:./mlruns'))
    mlflow.set_experiment(cfg.get('logging', {}).get('experiment_name', 'hgnn-pretrain'))
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Create datasets
        logger.info("Creating datasets...")
        
        # Create mock dataset for now (in practice, you'd load real data)
        train_dataset = GraphDataset(
            graph_paths={
                'knn': Path(args.data_path) / 'knn.parquet',
                'lr': Path(args.data_path) / 'lr.parquet',
                'grn': Path(args.data_path) / 'grn.parquet'
            },
            cell_data_path=Path(args.data_path) / 'cells.h5ad'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg['train'].get('batch_size', 8),
            shuffle=True,
            collate_fn=collate_graph_batch,
            num_workers=cfg['train'].get('num_workers', 0)
        )
        
        # Create pretrainer
        pretrainer = HGNNPretrainer(cfg)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch = pretrainer.load_checkpoint(Path(args.resume))
        
        # Training loop
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, cfg['train']['epochs']):
            # Train
            train_metrics = pretrainer.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = pretrainer.validate(train_loader)  # Using train_loader for now
            
            # Log metrics
            mlflow.log_metrics({
                **train_metrics,
                **val_metrics,
                'epoch': epoch
            }, step=epoch)
            
            # Update learning rate
            if pretrainer.scheduler is not None:
                pretrainer.scheduler.step()
            
            # Save checkpoint
            if epoch % cfg['train'].get('save_every', 10) == 0:
                pretrainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, output_dir)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                pretrainer.save_checkpoint(epoch, {**train_metrics, **val_metrics}, output_dir / 'best_model.pt')
            
            logger.info(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, val_loss={val_metrics['val_loss']:.4f}")
        
        # Save final model
        pretrainer.save_checkpoint(cfg['train']['epochs'] - 1, {**train_metrics, **val_metrics}, output_dir / 'final_model.pt')
        
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
