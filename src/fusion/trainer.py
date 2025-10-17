"""
Fusion trainer for dual-encoder alignment.

Handles training of the hybrid model with text and graph encoders.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.text.gemma_loader import GemmaTextEncoder
from src.hgnn.encoder import UHGEncoder
from src.uhg_adapters.radial_projector import RadialProjector
from src.fusion.align_losses import create_alignment_loss
from src.fusion.heads import create_fusion_head

logger = logging.getLogger(__name__)


class FusionTrainer:
    """
    Fusion trainer for dual-encoder alignment.
    
    Handles training of the hybrid model with text and graph encoders.
    """
    
    def __init__(
        self,
        text_encoder: GemmaTextEncoder,
        graph_encoder: UHGEncoder,
        radial_projector: RadialProjector,
        fusion_head: nn.Module,
        alignment_loss: nn.Module,
        config: Dict
    ):
        """
        Initialize fusion trainer.
        
        Args:
            text_encoder: Text encoder (Gemma)
            graph_encoder: Graph encoder (UHG)
            radial_projector: Radial projector for UHG -> Euclidean
            fusion_head: Fusion head for combining embeddings
            alignment_loss: Alignment loss function
            config: Configuration dictionary
        """
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.radial_projector = radial_projector
        self.fusion_head = fusion_head
        self.alignment_loss = alignment_loss
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        self.text_encoder.to(self.device)
        self.graph_encoder.to(self.device)
        self.radial_projector.to(self.device)
        self.fusion_head.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        logger.info(f"Initialized fusion trainer on {self.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['train']
        
        # Get trainable parameters
        trainable_params = []
        
        # Text encoder parameters (LoRA only)
        for param in self.text_encoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Graph encoder parameters
        for param in self.graph_encoder.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Radial projector parameters
        for param in self.radial_projector.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Fusion head parameters
        for param in self.fusion_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=train_config.get('lr', 2e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        logger.info(f"Created optimizer with {len(trainable_params)} parameter groups")
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
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.text_encoder.train()
        self.graph_encoder.train()
        self.radial_projector.train()
        self.fusion_head.train()
        
        total_loss = 0.0
        total_alignment_loss = 0.0
        total_fusion_loss = 0.0
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
                    self._get_trainable_parameters(),
                    self.config['train']['grad_clip']
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_alignment_loss += loss_dict['alignment_loss'].item()
            total_fusion_loss += loss_dict['fusion_loss'].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'align': f"{loss_dict['alignment_loss'].item():.4f}",
                'fusion': f"{loss_dict['fusion_loss'].item():.4f}"
            })
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_alignment_loss = total_alignment_loss / num_batches
        avg_fusion_loss = total_fusion_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_alignment_loss': avg_alignment_loss,
            'train_fusion_loss': avg_fusion_loss
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
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Get graph data
        knn_neighbors = batch.get('knn_neighbors', None)
        lr_interactions = batch.get('lr_interactions', None)
        grn_edges = batch.get('grn_edges', None)
        
        # Create node features (simplified - in practice, you'd use actual features)
        batch_size = input_ids.size(0)
        input_dim = self.config['model']['hgnn'].get('input_dim', 64)
        node_features = torch.randn(batch_size, input_dim, device=self.device)
        
        # Create edge indices (simplified - in practice, you'd use actual graph structure)
        edge_index = self._create_edge_index(batch_size)
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=True
        )
        text_embeddings = text_outputs['last_hidden_state']
        
        # Pool text embeddings
        text_embeddings = self._pool_text_embeddings(text_embeddings, attention_mask)
        
        # Encode graph
        graph_outputs = self.graph_encoder(node_features, edge_index)
        graph_embeddings = graph_outputs['node_embeddings']
        
        # Project graph embeddings to Euclidean space
        graph_embeddings_euclidean = self.radial_projector(graph_embeddings)
        
        # Compute alignment loss
        alignment_loss_dict = self.alignment_loss(text_embeddings, graph_embeddings_euclidean)
        alignment_loss = alignment_loss_dict['total_loss']
        
        # Fuse embeddings
        fused_embeddings = self.fusion_head(text_embeddings, graph_embeddings_euclidean)
        
        # Compute fusion loss (simplified - in practice, you'd use task-specific loss)
        fusion_loss = torch.mean(torch.sum(fused_embeddings**2, dim=-1))
        
        # Total loss
        total_loss = alignment_loss + fusion_loss
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'fusion_loss': fusion_loss,
            'text_embeddings': text_embeddings,
            'graph_embeddings': graph_embeddings_euclidean,
            'fused_embeddings': fused_embeddings
        }
    
    def _pool_text_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool text embeddings.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        # Simple mean pooling
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden_states = hidden_states * attention_mask
        
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        seq_lengths = torch.sum(attention_mask, dim=1)
        
        pooled = sum_hidden_states / (seq_lengths + 1e-8)
        
        return pooled
    
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
    
    def _get_trainable_parameters(self) -> list:
        """Get trainable parameters."""
        trainable_params = []
        
        for model in [self.text_encoder, self.graph_encoder, self.radial_projector, self.fusion_head]:
            for param in model.parameters():
                if param.requires_grad:
                    trainable_params.append(param)
        
        return trainable_params
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.text_encoder.eval()
        self.graph_encoder.eval()
        self.radial_projector.eval()
        self.fusion_head.eval()
        
        total_loss = 0.0
        total_alignment_loss = 0.0
        total_fusion_loss = 0.0
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
                total_alignment_loss += loss_dict['alignment_loss'].item()
                total_fusion_loss += loss_dict['fusion_loss'].item()
                num_batches += 1
        
        # Compute average losses
        avg_loss = total_loss / num_batches
        avg_alignment_loss = total_alignment_loss / num_batches
        avg_fusion_loss = total_fusion_loss / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_alignment_loss': avg_alignment_loss,
            'val_fusion_loss': avg_fusion_loss
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], checkpoint_dir: str):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            checkpoint_dir: Checkpoint directory
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'graph_encoder_state_dict': self.graph_encoder.state_dict(),
            'radial_projector_state_dict': self.radial_projector.state_dict(),
            'fusion_head_state_dict': self.fusion_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder_state_dict'])
        self.radial_projector.load_state_dict(checkpoint['radial_projector_state_dict'])
        self.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def get_embeddings(
        self,
        dataloader: DataLoader,
        return_fused: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get embeddings from the model.
        
        Args:
            dataloader: Data loader
            return_fused: Whether to return fused embeddings
            
        Returns:
            Dictionary containing embeddings
        """
        self.text_encoder.eval()
        self.graph_encoder.eval()
        self.radial_projector.eval()
        self.fusion_head.eval()
        
        all_text_embeddings = []
        all_graph_embeddings = []
        all_fused_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Extract data
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                
                # Create node features
                batch_size = input_ids.size(0)
                input_dim = self.config['model']['hgnn'].get('input_dim', 64)
                node_features = torch.randn(batch_size, input_dim, device=self.device)
                
                # Create edge indices
                edge_index = self._create_edge_index(batch_size)
                
                # Encode text
                text_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden_states=True
                )
                text_embeddings = text_outputs['last_hidden_state']
                text_embeddings = self._pool_text_embeddings(text_embeddings, attention_mask)
                
                # Encode graph
                graph_outputs = self.graph_encoder(node_features, edge_index)
                graph_embeddings = graph_outputs['node_embeddings']
                graph_embeddings_euclidean = self.radial_projector(graph_embeddings)
                
                # Store embeddings
                all_text_embeddings.append(text_embeddings)
                all_graph_embeddings.append(graph_embeddings_euclidean)
                
                if return_fused:
                    fused_embeddings = self.fusion_head(text_embeddings, graph_embeddings_euclidean)
                    all_fused_embeddings.append(fused_embeddings)
        
        # Concatenate embeddings
        result = {
            'text_embeddings': torch.cat(all_text_embeddings, dim=0),
            'graph_embeddings': torch.cat(all_graph_embeddings, dim=0)
        }
        
        if return_fused:
            result['fused_embeddings'] = torch.cat(all_fused_embeddings, dim=0)
        
        return result
