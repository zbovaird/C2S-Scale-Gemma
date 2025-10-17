"""
UHG contrastive losses for self-supervised learning.

Implements contrastive losses in hyperbolic space using UHG operations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from uhg.projective import ProjectiveUHG

logger = logging.getLogger(__name__)


class UHGContrastiveLoss(nn.Module):
    """
    UHG contrastive loss for self-supervised learning.
    
    Implements contrastive loss in hyperbolic space using UHG distance.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 1.0,
        hard_negative_mining: bool = True,
        num_hard_negatives: int = 5
    ):
        """
        Initialize UHG contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for hard negative mining
            hard_negative_mining: Whether to use hard negative mining
            num_hard_negatives: Number of hard negatives to use
        """
        super().__init__()
        
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_mining = hard_negative_mining
        self.num_hard_negatives = num_hard_negatives
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        logger.info(f"Initialized UHG contrastive loss: temp={temperature}, margin={margin}")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute UHG contrastive loss.
        
        Args:
            embeddings: Node embeddings [N, D]
            labels: Node labels [N] (for hard negative mining)
            positive_pairs: Positive pairs [P, 2] (optional)
            negative_pairs: Negative pairs [N, 2] (optional)
            
        Returns:
            Dictionary containing loss components
        """
        # Compute UHG distances between all pairs
        distances = self._compute_uhg_distances(embeddings)
        
        # Create positive and negative masks
        if positive_pairs is not None and negative_pairs is not None:
            pos_mask, neg_mask = self._create_pair_masks(
                distances.size(0), positive_pairs, negative_pairs
            )
        else:
            pos_mask, neg_mask = self._create_label_masks(labels)
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(distances, pos_mask, neg_mask)
        
        # Hard negative mining
        hard_negative_loss = torch.tensor(0.0, device=embeddings.device)
        if self.hard_negative_mining:
            hard_negative_loss = self._compute_hard_negative_loss(
                distances, pos_mask, neg_mask, labels
            )
        
        # Total loss
        total_loss = contrastive_loss + hard_negative_loss
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'hard_negative_loss': hard_negative_loss,
            'distances': distances,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask
        }
    
    def _compute_uhg_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute UHG distances between all pairs of embeddings.
        
        Args:
            embeddings: Node embeddings [N, D]
            
        Returns:
            Distance matrix [N, N]
        """
        n = embeddings.size(0)
        distances = torch.zeros(n, n, device=embeddings.device)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = self.uhg.distance(embeddings[i], embeddings[j])
        
        return distances
    
    def _create_pair_masks(
        self,
        n: int,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive and negative masks from pair indices.
        
        Args:
            n: Number of nodes
            positive_pairs: Positive pairs [P, 2]
            negative_pairs: Negative pairs [N, 2]
            
        Returns:
            Tuple of (pos_mask, neg_mask)
        """
        pos_mask = torch.zeros(n, n, dtype=torch.bool, device=positive_pairs.device)
        neg_mask = torch.zeros(n, n, dtype=torch.bool, device=negative_pairs.device)
        
        # Set positive pairs
        for pair in positive_pairs:
            i, j = pair[0].item(), pair[1].item()
            pos_mask[i, j] = True
            pos_mask[j, i] = True
        
        # Set negative pairs
        for pair in negative_pairs:
            i, j = pair[0].item(), pair[1].item()
            neg_mask[i, j] = True
            neg_mask[j, i] = True
        
        return pos_mask, neg_mask
    
    def _create_label_masks(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive and negative masks based on labels.
        
        Args:
            labels: Node labels [N]
            
        Returns:
            Tuple of (pos_mask, neg_mask)
        """
        n = labels.size(0)
        pos_mask = torch.zeros(n, n, dtype=torch.bool, device=labels.device)
        neg_mask = torch.zeros(n, n, dtype=torch.bool, device=labels.device)
        
        # Create masks based on label similarity
        for i in range(n):
            for j in range(n):
                if i != j:
                    if labels[i] == labels[j]:
                        pos_mask[i, j] = True
                    else:
                        neg_mask[i, j] = True
        
        return pos_mask, neg_mask
    
    def _compute_contrastive_loss(
        self,
        distances: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss using UHG distances.
        
        Args:
            distances: Distance matrix [N, N]
            pos_mask: Positive mask [N, N]
            neg_mask: Negative mask [N, N]
            
        Returns:
            Contrastive loss
        """
        # Convert distances to similarities (negative distances)
        similarities = -distances / self.temperature
        
        # Compute positive similarities
        pos_similarities = similarities[pos_mask]
        
        # Compute negative similarities
        neg_similarities = similarities[neg_mask]
        
        if len(pos_similarities) == 0:
            return torch.tensor(0.0, device=distances.device)
        
        # Contrastive loss: maximize positive similarities, minimize negative similarities
        pos_loss = -torch.mean(pos_similarities)
        neg_loss = torch.mean(neg_similarities)
        
        contrastive_loss = pos_loss + neg_loss
        
        return contrastive_loss
    
    def _compute_hard_negative_loss(
        self,
        distances: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard negative mining loss.
        
        Args:
            distances: Distance matrix [N, N]
            pos_mask: Positive mask [N, N]
            neg_mask: Negative mask [N, N]
            labels: Node labels [N]
            
        Returns:
            Hard negative loss
        """
        hard_negative_loss = torch.tensor(0.0, device=distances.device)
        
        for i in range(distances.size(0)):
            # Find hard negatives for node i
            node_neg_mask = neg_mask[i]
            if node_neg_mask.sum() == 0:
                continue
            
            # Get negative distances for this node
            neg_distances = distances[i][node_neg_mask]
            
            # Find hardest negatives (closest negative neighbors)
            if len(neg_distances) > self.num_hard_negatives:
                hard_neg_indices = torch.topk(neg_distances, self.num_hard_negatives, largest=False)[1]
                hard_neg_distances = neg_distances[hard_neg_indices]
            else:
                hard_neg_distances = neg_distances
            
            # Hard negative loss: penalize close negative neighbors
            hard_neg_loss = torch.mean(torch.relu(self.margin - hard_neg_distances))
            hard_negative_loss += hard_neg_loss
        
        # Average over nodes
        hard_negative_loss = hard_negative_loss / distances.size(0)
        
        return hard_negative_loss


class UHGInfoNCELoss(nn.Module):
    """
    UHG InfoNCE loss for contrastive learning.
    
    Implements InfoNCE loss in hyperbolic space using UHG distance.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize UHG InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        
        self.temperature = temperature
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        logger.info(f"Initialized UHG InfoNCE loss: temp={temperature}")
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        positive_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute UHG InfoNCE loss.
        
        Args:
            query_embeddings: Query embeddings [N, D]
            key_embeddings: Key embeddings [N, D]
            positive_indices: Positive key indices for each query [N]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = query_embeddings.size(0)
        
        # Compute UHG distances between queries and keys
        distances = self._compute_uhg_distances(query_embeddings, key_embeddings)
        
        # Convert distances to similarities
        similarities = -distances / self.temperature
        
        # Compute InfoNCE loss
        infonce_loss = 0.0
        
        for i in range(batch_size):
            # Get similarities for query i
            query_similarities = similarities[i]
            
            # Get positive similarity
            pos_idx = positive_indices[i].item()
            pos_similarity = query_similarities[pos_idx]
            
            # Compute InfoNCE loss for this query
            # log(exp(pos_sim) / sum(exp(all_sims)))
            logits = query_similarities - pos_similarity  # Normalize by positive similarity
            infonce_loss += -logits[0] + torch.logsumexp(logits, dim=0)
        
        infonce_loss = infonce_loss / batch_size
        
        return {
            'infonce_loss': infonce_loss,
            'similarities': similarities,
            'distances': distances
        }
    
    def _compute_uhg_distances(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute UHG distances between queries and keys.
        
        Args:
            query_embeddings: Query embeddings [N, D]
            key_embeddings: Key embeddings [N, D]
            
        Returns:
            Distance matrix [N, N]
        """
        n_queries = query_embeddings.size(0)
        n_keys = key_embeddings.size(0)
        distances = torch.zeros(n_queries, n_keys, device=query_embeddings.device)
        
        for i in range(n_queries):
            for j in range(n_keys):
                distances[i, j] = self.uhg.distance(query_embeddings[i], key_embeddings[j])
        
        return distances


class UHGTripletLoss(nn.Module):
    """
    UHG triplet loss for metric learning.
    
    Implements triplet loss in hyperbolic space using UHG distance.
    """
    
    def __init__(self, margin: float = 1.0, distance_metric: str = "uhg"):
        """
        Initialize UHG triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ("uhg", "euclidean")
        """
        super().__init__()
        
        self.margin = margin
        self.distance_metric = distance_metric
        
        # UHG operations
        self.uhg = ProjectiveUHG()
        
        logger.info(f"Initialized UHG triplet loss: margin={margin}, metric={distance_metric}")
    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute UHG triplet loss.
        
        Args:
            anchor_embeddings: Anchor embeddings [N, D]
            positive_embeddings: Positive embeddings [N, D]
            negative_embeddings: Negative embeddings [N, D]
            
        Returns:
            Dictionary containing loss components
        """
        # Compute distances
        if self.distance_metric == "uhg":
            pos_distances = self._compute_uhg_distances(anchor_embeddings, positive_embeddings)
            neg_distances = self._compute_uhg_distances(anchor_embeddings, negative_embeddings)
        else:
            pos_distances = torch.norm(anchor_embeddings - positive_embeddings, dim=1)
            neg_distances = torch.norm(anchor_embeddings - negative_embeddings, dim=1)
        
        # Compute triplet loss
        triplet_loss = torch.mean(torch.relu(pos_distances - neg_distances + self.margin))
        
        return {
            'triplet_loss': triplet_loss,
            'pos_distances': pos_distances,
            'neg_distances': neg_distances
        }
    
    def _compute_uhg_distances(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute UHG distances between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [N, D]
            
        Returns:
            Distances [N]
        """
        distances = torch.zeros(embeddings1.size(0), device=embeddings1.device)
        
        for i in range(embeddings1.size(0)):
            distances[i] = self.uhg.distance(embeddings1[i], embeddings2[i])
        
        return distances


def create_uhg_loss(
    loss_type: str = "contrastive",
    **kwargs
) -> Union[UHGContrastiveLoss, UHGInfoNCELoss, UHGTripletLoss]:
    """
    Create a UHG loss instance.
    
    Args:
        loss_type: Type of loss ("contrastive", "infonce", "triplet")
        **kwargs: Additional arguments for loss
        
    Returns:
        UHG loss instance
    """
    if loss_type == "contrastive":
        return UHGContrastiveLoss(**kwargs)
    elif loss_type == "infonce":
        return UHGInfoNCELoss(**kwargs)
    elif loss_type == "triplet":
        return UHGTripletLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
