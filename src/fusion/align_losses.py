"""
Alignment losses for dual-encoder training.

Implements InfoNCE and other contrastive losses for aligning text and graph embeddings.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for dual-encoder alignment.
    
    Implements InfoNCE loss with hard negative mining for aligning
    text and graph embeddings.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_mining: bool = True,
        num_hard_negatives: int = 5,
        margin: float = 0.5
    ):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax
            hard_negative_mining: Whether to use hard negative mining
            num_hard_negatives: Number of hard negatives to use
            margin: Margin for hard negative mining
        """
        super().__init__()
        
        self.temperature = temperature
        self.hard_negative_mining = hard_negative_mining
        self.num_hard_negatives = num_hard_negatives
        self.margin = margin
        
        logger.info(f"Initialized InfoNCE loss: temp={temperature}, hard_neg={hard_negative_mining}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hard_negative_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.
        
        Args:
            text_embeddings: Text embeddings [batch_size, dim]
            graph_embeddings: Graph embeddings [batch_size, dim]
            labels: Labels for hard negative mining [batch_size]
            hard_negative_indices: Pre-computed hard negative indices [batch_size, num_hard_negatives]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = text_embeddings.size(0)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(text_embeddings, graph_embeddings.t()) / self.temperature
        
        # Create positive mask (diagonal)
        positive_mask = torch.eye(batch_size, device=text_embeddings.device)
        
        # Compute InfoNCE loss
        infonce_loss = self._compute_infonce_loss(similarity_matrix, positive_mask)
        
        # Hard negative mining
        hard_negative_loss = torch.tensor(0.0, device=text_embeddings.device)
        if self.hard_negative_mining and labels is not None:
            hard_negative_loss = self._compute_hard_negative_loss(
                similarity_matrix, labels, hard_negative_indices
            )
        
        # Total loss
        total_loss = infonce_loss + hard_negative_loss
        
        return {
            'total_loss': total_loss,
            'infonce_loss': infonce_loss,
            'hard_negative_loss': hard_negative_loss,
            'similarity_matrix': similarity_matrix
        }
    
    def _compute_infonce_loss(
        self,
        similarity_matrix: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            similarity_matrix: Similarity matrix [batch_size, batch_size]
            positive_mask: Positive mask [batch_size, batch_size]
            
        Returns:
            InfoNCE loss
        """
        # Compute log probabilities
        log_prob_matrix = F.log_softmax(similarity_matrix, dim=1)
        
        # Extract positive log probabilities
        positive_log_probs = log_prob_matrix * positive_mask
        
        # Sum over positive pairs
        positive_log_probs = torch.sum(positive_log_probs, dim=1)
        
        # Average over batch
        infonce_loss = -torch.mean(positive_log_probs)
        
        return infonce_loss
    
    def _compute_hard_negative_loss(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
        hard_negative_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hard negative mining loss.
        
        Args:
            similarity_matrix: Similarity matrix [batch_size, batch_size]
            labels: Labels for hard negative mining [batch_size]
            hard_negative_indices: Pre-computed hard negative indices
            
        Returns:
            Hard negative loss
        """
        batch_size = similarity_matrix.size(0)
        hard_negative_loss = 0.0
        
        for i in range(batch_size):
            # Get similarities for sample i
            sample_similarities = similarity_matrix[i]
            
            # Find hard negatives
            if hard_negative_indices is not None:
                # Use pre-computed hard negatives
                hard_neg_similarities = sample_similarities[hard_negative_indices[i]]
            else:
                # Find hard negatives based on labels
                hard_neg_mask = labels != labels[i]
                if hard_neg_mask.sum() == 0:
                    continue
                
                hard_neg_similarities = sample_similarities[hard_neg_mask]
                
                # Select hardest negatives
                if len(hard_neg_similarities) > self.num_hard_negatives:
                    hard_neg_similarities = torch.topk(
                        hard_neg_similarities, self.num_hard_negatives, largest=True
                    )[0]
            
            # Hard negative loss: penalize high similarities with negative samples
            hard_neg_loss = torch.mean(torch.relu(hard_neg_similarities - self.margin))
            hard_negative_loss += hard_neg_loss
        
        # Average over batch
        hard_negative_loss = hard_negative_loss / batch_size
        
        return hard_negative_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for dual-encoder alignment.
    
    Implements triplet loss with hard negative mining.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "cosine",
        hard_negative_mining: bool = True
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ("cosine", "euclidean")
            hard_negative_mining: Whether to use hard negative mining
        """
        super().__init__()
        
        self.margin = margin
        self.distance_metric = distance_metric
        self.hard_negative_mining = hard_negative_mining
        
        logger.info(f"Initialized triplet loss: margin={margin}, metric={distance_metric}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute triplet loss.
        
        Args:
            text_embeddings: Text embeddings [batch_size, dim]
            graph_embeddings: Graph embeddings [batch_size, dim]
            labels: Labels for hard negative mining [batch_size]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = text_embeddings.size(0)
        
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
        
        # Compute distances
        if self.distance_metric == "cosine":
            distances = 1 - torch.matmul(text_embeddings, graph_embeddings.t())
        else:  # euclidean
            distances = torch.cdist(text_embeddings, graph_embeddings, p=2)
        
        # Compute triplet loss
        triplet_loss = self._compute_triplet_loss(distances, labels)
        
        return {
            'triplet_loss': triplet_loss,
            'distances': distances
        }
    
    def _compute_triplet_loss(
        self,
        distances: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            distances: Distance matrix [batch_size, batch_size]
            labels: Labels for hard negative mining [batch_size]
            
        Returns:
            Triplet loss
        """
        batch_size = distances.size(0)
        triplet_loss = 0.0
        
        for i in range(batch_size):
            # Positive distance (diagonal)
            pos_distance = distances[i, i]
            
            # Find negative distances
            if labels is not None:
                # Use labels for hard negative mining
                neg_mask = labels != labels[i]
                if neg_mask.sum() == 0:
                    continue
                
                neg_distances = distances[i][neg_mask]
            else:
                # Use all other samples as negatives
                neg_distances = torch.cat([distances[i, :i], distances[i, i+1:]])
            
            # Find hardest negative
            if self.hard_negative_mining:
                hard_neg_distance = torch.min(neg_distances)
            else:
                hard_neg_distance = torch.mean(neg_distances)
            
            # Triplet loss
            triplet_loss += torch.relu(pos_distance - hard_neg_distance + self.margin)
        
        # Average over batch
        triplet_loss = triplet_loss / batch_size
        
        return triplet_loss


class ContrastiveLoss(nn.Module):
    """
    General contrastive loss for dual-encoder alignment.
    
    Implements contrastive loss with various similarity functions.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        similarity_function: str = "cosine",
        hard_negative_mining: bool = True
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter
            similarity_function: Similarity function ("cosine", "dot", "euclidean")
            hard_negative_mining: Whether to use hard negative mining
        """
        super().__init__()
        
        self.temperature = temperature
        self.similarity_function = similarity_function
        self.hard_negative_mining = hard_negative_mining
        
        logger.info(f"Initialized contrastive loss: temp={temperature}, sim={similarity_function}")
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss.
        
        Args:
            text_embeddings: Text embeddings [batch_size, dim]
            graph_embeddings: Graph embeddings [batch_size, dim]
            labels: Labels for hard negative mining [batch_size]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = text_embeddings.size(0)
        
        # Compute similarities
        similarities = self._compute_similarities(text_embeddings, graph_embeddings)
        
        # Create positive mask
        positive_mask = torch.eye(batch_size, device=text_embeddings.device)
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(similarities, positive_mask, labels)
        
        return {
            'contrastive_loss': contrastive_loss,
            'similarities': similarities
        }
    
    def _compute_similarities(
        self,
        text_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarities between text and graph embeddings.
        
        Args:
            text_embeddings: Text embeddings [batch_size, dim]
            graph_embeddings: Graph embeddings [batch_size, dim]
            
        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        if self.similarity_function == "cosine":
            # Normalize embeddings
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            graph_embeddings = F.normalize(graph_embeddings, p=2, dim=1)
            
            # Cosine similarity
            similarities = torch.matmul(text_embeddings, graph_embeddings.t())
        elif self.similarity_function == "dot":
            # Dot product
            similarities = torch.matmul(text_embeddings, graph_embeddings.t())
        elif self.similarity_function == "euclidean":
            # Euclidean distance (negative)
            distances = torch.cdist(text_embeddings, graph_embeddings, p=2)
            similarities = -distances
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
        
        # Apply temperature
        similarities = similarities / self.temperature
        
        return similarities
    
    def _compute_contrastive_loss(
        self,
        similarities: torch.Tensor,
        positive_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            similarities: Similarity matrix [batch_size, batch_size]
            positive_mask: Positive mask [batch_size, batch_size]
            labels: Labels for hard negative mining [batch_size]
            
        Returns:
            Contrastive loss
        """
        # Compute log probabilities
        log_prob_matrix = F.log_softmax(similarities, dim=1)
        
        # Extract positive log probabilities
        positive_log_probs = log_prob_matrix * positive_mask
        
        # Sum over positive pairs
        positive_log_probs = torch.sum(positive_log_probs, dim=1)
        
        # Average over batch
        contrastive_loss = -torch.mean(positive_log_probs)
        
        return contrastive_loss


def create_alignment_loss(
    loss_type: str = "infonce",
    **kwargs
) -> Union[InfoNCELoss, TripletLoss, ContrastiveLoss]:
    """
    Create alignment loss instance.
    
    Args:
        loss_type: Type of loss ("infonce", "triplet", "contrastive")
        **kwargs: Additional arguments for loss
        
    Returns:
        Alignment loss instance
    """
    if loss_type == "infonce":
        return InfoNCELoss(**kwargs)
    elif loss_type == "triplet":
        return TripletLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
