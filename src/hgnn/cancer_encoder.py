#!/usr/bin/env python3
"""
Cancer-Specific UHG-HGNN Encoder

Enhanced UHG-HGNN encoder with cancer-specific capabilities:
- Hierarchical cancer taxonomy (Tumor → Tissue → Cell type)
- Temporal evolution modeling (cancer progression, drug resistance)
- Spatial-temporal graphs (metastasis, tumor growth)
- Cancer-specific gene prioritization
- Drug resistance modeling
- Metastasis prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

# Import base UHG-HGNN components
from ..hgnn.encoder import UHGEncoder
from ..hgnn.layers import UHGGraphSAGELayer, UHGGINLayer, UHGAttentionLayer
from ..uhg_adapters.radial_projector import RadialProjector
from ..uhg_adapters.sampling import RadialBandSampler

logger = logging.getLogger(__name__)

class CancerHierarchicalTaxonomy(nn.Module):
    """Hierarchical cancer taxonomy encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Define cancer hierarchy levels
        self.hierarchy_levels = {
            'tumor': 0,      # Primary tumor type
            'tissue': 1,      # Tissue of origin
            'cell_type': 2   # Specific cell type
        }
        
        # Level-specific encoders
        self.level_encoders = nn.ModuleDict({
            'tumor': nn.Linear(input_dim, hidden_dim),
            'tissue': nn.Linear(input_dim, hidden_dim),
            'cell_type': nn.Linear(input_dim, hidden_dim)
        })
        
        # Hierarchy aggregation
        self.hierarchy_aggregator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Cancer-specific gene importance
        self.gene_importance = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x: torch.Tensor, cancer_metadata: Dict) -> torch.Tensor:
        """Forward pass through hierarchical taxonomy."""
        
        # Extract cancer metadata
        tumor_type = cancer_metadata.get('tumor_type', 'unknown')
        tissue_type = cancer_metadata.get('tissue_type', 'unknown')
        cell_type = cancer_metadata.get('cell_type', 'unknown')
        
        # Encode each hierarchy level
        level_embeddings = []
        
        # Tumor level encoding
        tumor_embedding = self.level_encoders['tumor'](x)
        tumor_embedding = tumor_embedding + self._get_tumor_embedding(tumor_type)
        level_embeddings.append(tumor_embedding)
        
        # Tissue level encoding
        tissue_embedding = self.level_encoders['tissue'](x)
        tissue_embedding = tissue_embedding + self._get_tissue_embedding(tissue_type)
        level_embeddings.append(tissue_embedding)
        
        # Cell type level encoding
        cell_embedding = self.level_encoders['cell_type'](x)
        cell_embedding = cell_embedding + self._get_cell_type_embedding(cell_type)
        level_embeddings.append(cell_embedding)
        
        # Stack level embeddings
        level_stack = torch.stack(level_embeddings, dim=1)  # [batch, levels, hidden_dim]
        
        # Aggregate hierarchy using attention
        aggregated, attention_weights = self.hierarchy_aggregator(
            level_stack, level_stack, level_stack
        )
        
        # Apply cancer-specific gene importance
        gene_weights = torch.sigmoid(self.gene_importance)
        x_weighted = x * gene_weights.unsqueeze(0)
        
        # Combine aggregated hierarchy with weighted input
        output = aggregated.mean(dim=1) + x_weighted
        
        return output, attention_weights
    
    def _get_tumor_embedding(self, tumor_type: str) -> torch.Tensor:
        """Get tumor type embedding."""
        tumor_embeddings = {
            'breast': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            'lung': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
            'colon': torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
            'prostate': torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
            'melanoma': torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
            'unknown': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        }
        return tumor_embeddings.get(tumor_type, tumor_embeddings['unknown']).to(self.gene_importance.device)
    
    def _get_tissue_embedding(self, tissue_type: str) -> torch.Tensor:
        """Get tissue type embedding."""
        tissue_embeddings = {
            'epithelial': torch.tensor([1.0, 0.0, 0.0]),
            'mesenchymal': torch.tensor([0.0, 1.0, 0.0]),
            'hematopoietic': torch.tensor([0.0, 0.0, 1.0]),
            'unknown': torch.tensor([0.33, 0.33, 0.33])
        }
        return tissue_embeddings.get(tissue_type, tissue_embeddings['unknown']).to(self.gene_importance.device)
    
    def _get_cell_type_embedding(self, cell_type: str) -> torch.Tensor:
        """Get cell type embedding."""
        cell_type_embeddings = {
            'cancer_cell': torch.tensor([1.0, 0.0, 0.0, 0.0]),
            'immune_cell': torch.tensor([0.0, 1.0, 0.0, 0.0]),
            'stromal_cell': torch.tensor([0.0, 0.0, 1.0, 0.0]),
            'endothelial_cell': torch.tensor([0.0, 0.0, 0.0, 1.0]),
            'unknown': torch.tensor([0.25, 0.25, 0.25, 0.25])
        }
        return cell_type_embeddings.get(cell_type, cell_type_embeddings['unknown']).to(self.gene_importance.device)


class CancerTemporalEvolution(nn.Module):
    """Temporal evolution modeling for cancer progression."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_timesteps: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Temporal evolution layers
        self.temporal_layers = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True)
            for _ in range(num_timesteps)
        ])
        
        # Drug resistance modeling
        self.drug_resistance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Cancer progression stages
        self.progression_stages = {
            'stage_i': 0,
            'stage_ii': 1,
            'stage_iii': 2,
            'stage_iv': 3,
            'metastatic': 4
        }
        
        # Stage-specific encoders
        self.stage_encoders = nn.ModuleDict({
            stage: nn.Linear(input_dim, hidden_dim)
            for stage in self.progression_stages.keys()
        })
        
    def forward(self, x: torch.Tensor, temporal_data: Dict) -> torch.Tensor:
        """Forward pass through temporal evolution."""
        
        # Extract temporal information
        current_stage = temporal_data.get('stage', 'stage_i')
        time_since_diagnosis = temporal_data.get('time_since_diagnosis', 0)
        drug_exposure = temporal_data.get('drug_exposure', {})
        
        # Encode current stage
        stage_embedding = self.stage_encoders[current_stage](x)
        
        # Model drug resistance evolution
        resistance_embedding = self.drug_resistance_encoder(x)
        
        # Combine stage and resistance information
        temporal_embedding = stage_embedding * resistance_embedding
        
        # Add time-dependent modulation
        time_modulation = torch.sin(torch.tensor(time_since_diagnosis) * 0.1)
        temporal_embedding = temporal_embedding * (1 + time_modulation)
        
        return temporal_embedding


class CancerSpatialTemporal(nn.Module):
    """Spatial-temporal modeling for metastasis and tumor growth."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Spatial encoders
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Metastasis prediction
        self.metastasis_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Tumor growth modeling
        self.growth_modeler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # x, y, z coordinates
            nn.Tanh()
        )
        
        # Metastasis sites
        self.metastasis_sites = {
            'bone': 0,
            'liver': 1,
            'lung': 2,
            'brain': 3,
            'adrenal': 4
        }
        
        # Site-specific encoders
        self.site_encoders = nn.ModuleDict({
            site: nn.Linear(hidden_dim, hidden_dim)
            for site in self.metastasis_sites.keys()
        })
        
    def forward(self, x: torch.Tensor, spatial_data: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass through spatial-temporal modeling."""
        
        # Encode spatial information
        spatial_embedding = self.spatial_encoder(x)
        
        # Predict metastasis probability
        metastasis_prob = self.metastasis_predictor(spatial_embedding)
        
        # Model tumor growth
        growth_coordinates = self.growth_modeler(spatial_embedding)
        
        # Predict metastasis to specific sites
        site_predictions = {}
        for site, site_embedding in self.site_encoders.items():
            site_pred = site_embedding(spatial_embedding)
            site_predictions[site] = torch.sigmoid(site_pred.mean(dim=-1))
        
        return {
            'metastasis_probability': metastasis_prob,
            'growth_coordinates': growth_coordinates,
            'site_predictions': site_predictions,
            'spatial_embedding': spatial_embedding
        }


class CancerUHGEncoder(nn.Module):
    """Cancer-specific UHG-HGNN encoder with enhanced capabilities."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        curvature: float = -1.0,
        cancer_specific: bool = True,
        hierarchical_taxonomy: bool = True,
        temporal_evolution: bool = True,
        spatial_temporal: bool = True,
        metastasis_prediction: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.curvature = curvature
        self.cancer_specific = cancer_specific
        
        # Base UHG-HGNN encoder
        self.base_encoder = UHGEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            curvature=curvature
        )
        
        # Cancer-specific enhancements
        if cancer_specific:
            if hierarchical_taxonomy:
                self.hierarchical_taxonomy = CancerHierarchicalTaxonomy(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim
                )
            
            if temporal_evolution:
                self.temporal_evolution = CancerTemporalEvolution(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim
                )
            
            if spatial_temporal:
                self.spatial_temporal = CancerSpatialTemporal(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim
                )
        
        # Radial projector for UHG to Euclidean
        self.radial_projector = RadialProjector(
            input_dim=output_dim,
            output_dim=output_dim,
            projection_type="monotone_radial",
            preserve_angular=True
        )
        
        # Cancer-specific output heads
        if cancer_specific:
            self.drug_response_head = nn.Linear(output_dim, 1)
            self.prognosis_head = nn.Linear(output_dim, 1)
            self.biomarker_head = nn.Linear(output_dim, input_dim)
            self.classification_head = nn.Linear(output_dim, 5)  # 5 cancer types
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        cancer_metadata: Optional[Dict] = None,
        temporal_data: Optional[Dict] = None,
        spatial_data: Optional[Dict] = None,
        return_projections: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through cancer-specific UHG-HGNN encoder."""
        
        # Base UHG-HGNN encoding
        base_outputs = self.base_encoder(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            return_projections=return_projections
        )
        
        hyperbolic_embeddings = base_outputs['hyperbolic_embeddings']
        euclidean_embeddings = base_outputs['euclidean_embeddings']
        
        # Cancer-specific enhancements
        cancer_outputs = {}
        
        if self.cancer_specific:
            # Hierarchical taxonomy
            if hasattr(self, 'hierarchical_taxonomy') and cancer_metadata:
                hierarchy_embedding, attention_weights = self.hierarchical_taxonomy(
                    x, cancer_metadata
                )
                cancer_outputs['hierarchy_embedding'] = hierarchy_embedding
                cancer_outputs['hierarchy_attention'] = attention_weights
            
            # Temporal evolution
            if hasattr(self, 'temporal_evolution') and temporal_data:
                temporal_embedding = self.temporal_evolution(x, temporal_data)
                cancer_outputs['temporal_embedding'] = temporal_embedding
            
            # Spatial-temporal modeling
            if hasattr(self, 'spatial_temporal') and spatial_data:
                spatial_outputs = self.spatial_temporal(x, spatial_data)
                cancer_outputs.update(spatial_outputs)
            
            # Cancer-specific predictions
            if return_projections:
                # Drug response prediction
                drug_response = self.drug_response_head(euclidean_embeddings)
                cancer_outputs['drug_response'] = torch.sigmoid(drug_response)
                
                # Prognosis prediction
                prognosis = self.prognosis_head(euclidean_embeddings)
                cancer_outputs['prognosis'] = prognosis
                
                # Biomarker discovery
                biomarkers = self.biomarker_head(euclidean_embeddings)
                cancer_outputs['biomarkers'] = torch.sigmoid(biomarkers)
                
                # Cancer classification
                classification = self.classification_head(euclidean_embeddings)
                cancer_outputs['classification'] = F.softmax(classification, dim=-1)
        
        # Combine all outputs
        outputs = {
            **base_outputs,
            **cancer_outputs
        }
        
        return outputs
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cancer_metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """Encode individual nodes with cancer-specific information."""
        
        outputs = self.forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cancer_metadata=cancer_metadata,
            return_projections=True
        )
        
        return outputs['euclidean_embeddings']
    
    def encode_graphs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cancer_metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """Encode entire graphs with cancer-specific information."""
        
        outputs = self.forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            cancer_metadata=cancer_metadata,
            return_projections=True
        )
        
        return outputs['graph_embeddings']
    
    def predict_drug_response(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        drug_name: str,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict drug response for cancer cells."""
        
        outputs = self.forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            return_projections=True
        )
        
        return outputs['drug_response']
    
    def predict_metastasis(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        spatial_data: Dict,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict metastasis probability and sites."""
        
        outputs = self.forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            spatial_data=spatial_data,
            return_projections=True
        )
        
        return {
            'metastasis_probability': outputs['metastasis_probability'],
            'site_predictions': outputs['site_predictions']
        }
    
    def discover_biomarkers(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Discover cancer biomarkers."""
        
        outputs = self.forward(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            return_projections=True
        )
        
        return outputs['biomarkers']
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'cancer_specific': self.cancer_specific,
            'hierarchical_taxonomy': hasattr(self, 'hierarchical_taxonomy'),
            'temporal_evolution': hasattr(self, 'temporal_evolution'),
            'spatial_temporal': hasattr(self, 'spatial_temporal'),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
