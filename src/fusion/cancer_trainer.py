#!/usr/bin/env python3
"""
Cancer-Specific Hybrid Trainer

Enhanced trainer for cancer research tasks with:
- Cancer-specific loss functions
- Multi-task learning
- Clinical validation
- Drug response modeling
- Prognosis prediction
- Biomarker discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import mlflow
import wandb

# Import our modules
from ..hgnn.cancer_encoder import CancerUHGEncoder
from ..text.cancer_gemma_loader import CancerGemmaLoader
from ..fusion.trainer import DualEncoderTrainer
from ..fusion.align_losses import InfoNCELoss
from ..fusion.heads import FusionHead

logger = logging.getLogger(__name__)

class CancerSpecificLoss(nn.Module):
    """Cancer-specific loss functions."""
    
    def __init__(
        self,
        drug_response_weight: float = 0.3,
        prognosis_weight: float = 0.2,
        biomarker_weight: float = 0.2,
        classification_weight: float = 0.3,
        clinical_relevance_weight: float = 0.1
    ):
        super().__init__()
        
        self.drug_response_weight = drug_response_weight
        self.prognosis_weight = prognosis_weight
        self.biomarker_weight = biomarker_weight
        self.classification_weight = classification_weight
        self.clinical_relevance_weight = clinical_relevance_weight
        
        # Loss functions
        self.drug_response_loss = nn.BCELoss()
        self.prognosis_loss = nn.MSELoss()
        self.biomarker_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.clinical_relevance_loss = nn.MSELoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute cancer-specific losses."""
        
        losses = {}
        
        # Drug response loss
        if 'drug_response' in predictions and 'drug_response' in targets:
            drug_loss = self.drug_response_loss(
                predictions['drug_response'].squeeze(),
                targets['drug_response'].float()
            )
            losses['drug_response'] = drug_loss
            losses['weighted_drug_response'] = self.drug_response_weight * drug_loss
        
        # Prognosis loss
        if 'prognosis' in predictions and 'prognosis' in targets:
            prognosis_loss = self.prognosis_loss(
                predictions['prognosis'].squeeze(),
                targets['prognosis'].float()
            )
            losses['prognosis'] = prognosis_loss
            losses['weighted_prognosis'] = self.prognosis_weight * prognosis_loss
        
        # Biomarker loss
        if 'biomarkers' in predictions and 'biomarkers' in targets:
            biomarker_loss = self.biomarker_loss(
                predictions['biomarkers'],
                targets['biomarkers'].float()
            )
            losses['biomarkers'] = biomarker_loss
            losses['weighted_biomarkers'] = self.biomarker_weight * biomarker_loss
        
        # Classification loss
        if 'classification' in predictions and 'classification' in targets:
            classification_loss = self.classification_loss(
                predictions['classification'],
                targets['classification'].long()
            )
            losses['classification'] = classification_loss
            losses['weighted_classification'] = self.classification_weight * classification_loss
        
        # Clinical relevance loss
        if 'clinical_relevance' in predictions and 'clinical_relevance' in targets:
            clinical_loss = self.clinical_relevance_loss(
                predictions['clinical_relevance'].squeeze(),
                targets['clinical_relevance'].float()
            )
            losses['clinical_relevance'] = clinical_loss
            losses['weighted_clinical_relevance'] = self.clinical_relevance_weight * clinical_loss
        
        # Total weighted loss
        total_loss = sum(
            loss for key, loss in losses.items() 
            if key.startswith('weighted_')
        )
        losses['total_loss'] = total_loss
        
        return losses


class CancerHybridTrainer(DualEncoderTrainer):
    """Cancer-specific hybrid trainer."""
    
    def __init__(
        self,
        hgnn_encoder: CancerUHGEncoder,
        text_encoder: CancerGemmaLoader,
        device: torch.device,
        cancer_specific: bool = True,
        drug_response_weight: float = 0.3,
        prognosis_weight: float = 0.2,
        biomarker_weight: float = 0.2,
        classification_weight: float = 0.3,
        clinical_relevance_weight: float = 0.1
    ):
        # Initialize base trainer
        super().__init__(
            hgnn_encoder=hgnn_encoder,
            text_model=text_encoder,
            fusion_head=FusionHead(
                graph_dim=hgnn_encoder.output_dim,
                text_dim=text_encoder.model.config.hidden_size,
                fusion_dim=768,
                dropout=0.1
            ),
            contrastive_loss=InfoNCELoss(temperature=0.07),
            device=device
        )
        
        self.cancer_specific = cancer_specific
        self.device = device
        
        # Cancer-specific loss
        self.cancer_loss = CancerSpecificLoss(
            drug_response_weight=drug_response_weight,
            prognosis_weight=prognosis_weight,
            biomarker_weight=biomarker_weight,
            classification_weight=classification_weight,
            clinical_relevance_weight=clinical_relevance_weight
        )
        
        # Cancer-specific task heads
        if cancer_specific:
            self.drug_response_head = nn.Linear(768, 1).to(device)
            self.prognosis_head = nn.Linear(768, 1).to(device)
            self.biomarker_head = nn.Linear(768, 20000).to(device)  # 20k genes
            self.classification_head = nn.Linear(768, 5).to(device)  # 5 cancer types
            self.clinical_relevance_head = nn.Linear(768, 1).to(device)
        
        # Training metrics
        self.training_metrics = {
            'drug_response_auroc': [],
            'prognosis_correlation': [],
            'biomarker_overlap': [],
            'classification_accuracy': [],
            'clinical_relevance': []
        }
        
    def compute_cancer_loss(
        self,
        batch: Dict[str, torch.Tensor],
        cancer_metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute cancer-specific loss."""
        
        # Get graph and text representations
        graph_repr = self.get_graph_representation(batch)
        text_repr = self.get_text_representation(batch)
        fused_repr = self.get_fused_representation(batch)
        
        # Cancer-specific predictions
        predictions = {}
        
        if self.cancer_specific:
            # Drug response prediction
            if 'drug_response' in batch:
                drug_response_pred = torch.sigmoid(self.drug_response_head(fused_repr))
                predictions['drug_response'] = drug_response_pred
            
            # Prognosis prediction
            if 'prognosis' in batch:
                prognosis_pred = self.prognosis_head(fused_repr)
                predictions['prognosis'] = prognosis_pred
            
            # Biomarker discovery
            if 'biomarkers' in batch:
                biomarker_pred = torch.sigmoid(self.biomarker_head(fused_repr))
                predictions['biomarkers'] = biomarker_pred
            
            # Cancer classification
            if 'classification' in batch:
                classification_pred = F.softmax(self.classification_head(fused_repr), dim=-1)
                predictions['classification'] = classification_pred
            
            # Clinical relevance
            if 'clinical_relevance' in batch:
                clinical_pred = torch.sigmoid(self.clinical_relevance_head(fused_repr))
                predictions['clinical_relevance'] = clinical_pred
        
        # Compute cancer-specific losses
        cancer_losses = self.cancer_loss(predictions, batch)
        
        # Add base contrastive loss
        base_losses = self.compute_loss(batch)
        cancer_losses.update(base_losses)
        
        return cancer_losses
    
    def train_cancer_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch on cancer-specific tasks."""
        
        self.train()
        epoch_losses = []
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Cancer Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Compute cancer-specific loss
            loss_dict = self.compute_cancer_loss(batch)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Update metrics
            epoch_losses.append(total_loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'drug_response': f"{loss_dict.get('drug_response', 0):.4f}",
                'prognosis': f"{loss_dict.get('prognosis', 0):.4f}",
                'biomarkers': f"{loss_dict.get('biomarkers', 0):.4f}",
                'classification': f"{loss_dict.get('classification', 0):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        epoch_metrics['avg_loss'] = avg_loss
        
        return epoch_metrics
    
    def evaluate_cancer_tasks(
        self,
        test_loader: DataLoader,
        task_evaluator: Optional[object] = None
    ) -> Dict[str, float]:
        """Evaluate cancer-specific tasks."""
        
        self.eval()
        evaluation_results = {}
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                graph_repr = self.get_graph_representation(batch)
                text_repr = self.get_text_representation(batch)
                fused_repr = self.get_fused_representation(batch)
                
                # Cancer-specific predictions
                if self.cancer_specific:
                    # Drug response prediction
                    drug_response_pred = torch.sigmoid(self.drug_response_head(fused_repr))
                    
                    # Prognosis prediction
                    prognosis_pred = self.prognosis_head(fused_repr)
                    
                    # Biomarker discovery
                    biomarker_pred = torch.sigmoid(self.biomarker_head(fused_repr))
                    
                    # Cancer classification
                    classification_pred = F.softmax(self.classification_head(fused_repr), dim=-1)
                    
                    # Clinical relevance
                    clinical_pred = torch.sigmoid(self.clinical_relevance_head(fused_repr))
                    
                    # Store predictions for evaluation
                    if 'predictions' not in evaluation_results:
                        evaluation_results['predictions'] = {
                            'drug_response': [],
                            'prognosis': [],
                            'biomarkers': [],
                            'classification': [],
                            'clinical_relevance': []
                        }
                    
                    evaluation_results['predictions']['drug_response'].append(drug_response_pred.cpu())
                    evaluation_results['predictions']['prognosis'].append(prognosis_pred.cpu())
                    evaluation_results['predictions']['biomarkers'].append(biomarker_pred.cpu())
                    evaluation_results['predictions']['classification'].append(classification_pred.cpu())
                    evaluation_results['predictions']['clinical_relevance'].append(clinical_pred.cpu())
        
        return evaluation_results
    
    def train_cancer_model(
        self,
        train_dataset,
        test_dataset=None,
        config: Dict = None
    ) -> Dict[str, any]:
        """Train cancer-specific hybrid model."""
        
        logger.info("Starting cancer-specific model training")
        
        # Training configuration
        batch_size = config.get('batch_size', 8)
        learning_rate = config.get('learning_rate', 1.5e-4)
        num_epochs = config.get('num_epochs', 10)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        warmup_steps = config.get('warmup_steps', 100)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        # Training loop
        training_results = {
            'epoch_losses': [],
            'epoch_metrics': [],
            'best_metrics': {},
            'final_metrics': {}
        }
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train epoch
            epoch_metrics = self.train_cancer_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch
            )
            
            training_results['epoch_losses'].append(epoch_metrics['avg_loss'])
            training_results['epoch_metrics'].append(epoch_metrics)
            
            # Evaluate on test set
            if test_loader:
                test_results = self.evaluate_cancer_tasks(test_loader)
                training_results['test_results'] = test_results
            
            # Log metrics
            if mlflow.active_run():
                mlflow.log_metrics({
                    f'epoch_{epoch}_loss': epoch_metrics['avg_loss'],
                    f'epoch_{epoch}_lr': scheduler.get_last_lr()[0]
                })
            
            # Save best model
            if epoch_metrics['avg_loss'] < best_loss:
                best_loss = epoch_metrics['avg_loss']
                training_results['best_metrics'] = epoch_metrics
                
                # Save model checkpoint
                self.save_checkpoint(f"best_cancer_model_epoch_{epoch}.pt")
        
        # Final evaluation
        if test_loader:
            final_results = self.evaluate_cancer_tasks(test_loader)
            training_results['final_metrics'] = final_results
        
        logger.info("Cancer-specific model training completed")
        return training_results
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        
        checkpoint = {
            'hgnn_encoder_state_dict': self.hgnn_encoder.state_dict(),
            'text_encoder_state_dict': self.text_encoder.model.state_dict(),
            'fusion_head_state_dict': self.fusion_head.state_dict(),
            'drug_response_head_state_dict': self.drug_response_head.state_dict() if self.cancer_specific else None,
            'prognosis_head_state_dict': self.prognosis_head.state_dict() if self.cancer_specific else None,
            'biomarker_head_state_dict': self.biomarker_head.state_dict() if self.cancer_specific else None,
            'classification_head_state_dict': self.classification_head.state_dict() if self.cancer_specific else None,
            'clinical_relevance_head_state_dict': self.clinical_relevance_head.state_dict() if self.cancer_specific else None,
            'cancer_specific': self.cancer_specific
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.hgnn_encoder.load_state_dict(checkpoint['hgnn_encoder_state_dict'])
        self.text_encoder.model.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
        
        if self.cancer_specific and checkpoint['cancer_specific']:
            self.drug_response_head.load_state_dict(checkpoint['drug_response_head_state_dict'])
            self.prognosis_head.load_state_dict(checkpoint['prognosis_head_state_dict'])
            self.biomarker_head.load_state_dict(checkpoint['biomarker_head_state_dict'])
            self.classification_head.load_state_dict(checkpoint['classification_head_state_dict'])
            self.clinical_relevance_head.load_state_dict(checkpoint['clinical_relevance_head_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def export_for_deployment(
        self,
        output_dir: str,
        vertex_config: Dict = None
    ):
        """Export model for Vertex AI deployment."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        self.save_checkpoint(output_path / "cancer_model_checkpoint.pt")
        
        # Save model configuration
        model_config = {
            'hgnn_config': self.hgnn_encoder.get_model_info(),
            'text_config': {
                'model_name': self.text_encoder.model_name,
                'cancer_specific': self.cancer_specific
            },
            'fusion_config': {
                'graph_dim': self.hgnn_encoder.output_dim,
                'text_dim': self.text_encoder.model.config.hidden_size,
                'fusion_dim': 768
            },
            'cancer_config': {
                'drug_response_weight': self.cancer_loss.drug_response_weight,
                'prognosis_weight': self.cancer_loss.prognosis_weight,
                'biomarker_weight': self.cancer_loss.biomarker_weight,
                'classification_weight': self.cancer_loss.classification_weight,
                'clinical_relevance_weight': self.cancer_loss.clinical_relevance_weight
            }
        }
        
        with open(output_path / "model_config.json", 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save Vertex AI deployment configuration
        if vertex_config:
            with open(output_path / "vertex_config.json", 'w') as f:
                json.dump(vertex_config, f, indent=2)
        
        logger.info(f"Model exported for deployment to {output_dir}")
