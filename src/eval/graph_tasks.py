"""
Graph-based evaluation tasks for C2S-Scale-Gemma hybrid model.

This module contains graph-specific evaluation tasks including graph classification,
graph regression, and graph generation tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform


class GraphClassificationTask:
    """Graph-level classification evaluation task."""
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pooling = pooling
        
        # Create graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def pool_graph_embeddings(
        self, 
        node_embeddings: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Pool node embeddings to graph-level embeddings."""
        
        if self.pooling == "mean":
            return torch.mean(node_embeddings, dim=1)
        elif self.pooling == "max":
            return torch.max(node_embeddings, dim=1)[0]
        elif self.pooling == "sum":
            return torch.sum(node_embeddings, dim=1)
        elif self.pooling == "attention":
            # Simple attention pooling
            attention_weights = torch.softmax(
                torch.sum(node_embeddings, dim=-1, keepdim=True), dim=1
            )
            return torch.sum(node_embeddings * attention_weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        graph_embeddings = self.pool_graph_embeddings(node_embeddings, batch)
        return self.classifier(graph_embeddings)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        return nn.CrossEntropyLoss()(outputs, targets)
    
    def compute_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        predictions = torch.argmax(outputs, dim=1)
        
        metrics = {
            "accuracy": accuracy_score(targets.cpu(), predictions.cpu()),
            "f1_macro": f1_score(targets.cpu(), predictions.cpu(), average="macro"),
            "f1_weighted": f1_score(targets.cpu(), predictions.cpu(), average="weighted"),
        }
        
        # Compute ROC-AUC for binary classification
        if self.num_classes == 2:
            probs = torch.softmax(outputs, dim=1)[:, 1]
            metrics["roc_auc"] = roc_auc_score(targets.cpu(), probs.cpu())
        
        return metrics


class GraphRegressionTask:
    """Graph-level regression evaluation task."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pooling = pooling
        
        # Create graph-level regressor
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def pool_graph_embeddings(
        self, 
        node_embeddings: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Pool node embeddings to graph-level embeddings."""
        
        if self.pooling == "mean":
            return torch.mean(node_embeddings, dim=1)
        elif self.pooling == "max":
            return torch.max(node_embeddings, dim=1)[0]
        elif self.pooling == "sum":
            return torch.sum(node_embeddings, dim=1)
        elif self.pooling == "attention":
            # Simple attention pooling
            attention_weights = torch.softmax(
                torch.sum(node_embeddings, dim=-1, keepdim=True), dim=1
            )
            return torch.sum(node_embeddings * attention_weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        graph_embeddings = self.pool_graph_embeddings(node_embeddings, batch)
        return self.regressor(graph_embeddings)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute regression loss."""
        return nn.MSELoss()(outputs.squeeze(), targets.float())
    
    def compute_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        outputs_np = outputs.squeeze().detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        metrics = {
            "mse": mean_squared_error(targets_np, outputs_np),
            "mae": mean_absolute_error(targets_np, outputs_np),
            "r2": r2_score(targets_np, outputs_np),
        }
        
        return metrics


class GraphSimilarityTask:
    """Graph similarity evaluation task."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
    
    def compute_graph_similarity(
        self,
        graph1_embeddings: torch.Tensor,
        graph2_embeddings: torch.Tensor,
        method: str = "cosine",
    ) -> torch.Tensor:
        """Compute similarity between two graphs."""
        
        if method == "cosine":
            # Cosine similarity
            norm1 = torch.norm(graph1_embeddings, dim=-1, keepdim=True)
            norm2 = torch.norm(graph2_embeddings, dim=-1, keepdim=True)
            similarity = torch.sum(graph1_embeddings * graph2_embeddings, dim=-1) / (norm1 * norm2)
            return torch.mean(similarity)
        
        elif method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = torch.norm(graph1_embeddings - graph2_embeddings, dim=-1)
            similarity = 1.0 / (1.0 + distance)
            return torch.mean(similarity)
        
        elif method == "dot_product":
            # Dot product similarity
            similarity = torch.sum(graph1_embeddings * graph2_embeddings, dim=-1)
            return torch.mean(similarity)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def evaluate_similarity_prediction(
        self,
        graph_embeddings: np.ndarray,
        similarity_labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate graph similarity prediction."""
        
        # Prepare data
        X_train = graph_embeddings[train_mask]
        y_train = similarity_labels[train_mask]
        X_val = graph_embeddings[val_mask]
        y_val = similarity_labels[val_mask]
        X_test = graph_embeddings[test_mask]
        y_test = similarity_labels[test_mask]
        
        # Train regressor
        regressor = Ridge(alpha=1.0)
        regressor.fit(X_train, y_train)
        
        # Evaluate
        val_pred = regressor.predict(X_val)
        test_pred = regressor.predict(X_test)
        
        metrics = {
            "val_mse": mean_squared_error(y_val, val_pred),
            "test_mse": mean_squared_error(y_test, test_pred),
            "val_r2": r2_score(y_val, val_pred),
            "test_r2": r2_score(y_test, test_pred),
        }
        
        return metrics


class GraphGenerationTask:
    """Graph generation evaluation task."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
    
    def evaluate_graph_reconstruction(
        self,
        original_graphs: List[nx.Graph],
        reconstructed_graphs: List[nx.Graph],
    ) -> Dict[str, float]:
        """Evaluate graph reconstruction quality."""
        
        metrics = {
            "node_count_accuracy": 0.0,
            "edge_count_accuracy": 0.0,
            "density_correlation": 0.0,
            "clustering_correlation": 0.0,
        }
        
        node_count_errors = []
        edge_count_errors = []
        density_correlations = []
        clustering_correlations = []
        
        for orig, recon in zip(original_graphs, reconstructed_graphs):
            # Node count accuracy
            node_count_error = abs(orig.number_of_nodes() - recon.number_of_nodes())
            node_count_errors.append(node_count_error)
            
            # Edge count accuracy
            edge_count_error = abs(orig.number_of_edges() - recon.number_of_edges())
            edge_count_errors.append(edge_count_error)
            
            # Density correlation
            orig_density = nx.density(orig)
            recon_density = nx.density(recon)
            density_correlations.append((orig_density, recon_density))
            
            # Clustering coefficient correlation
            orig_clustering = nx.average_clustering(orig)
            recon_clustering = nx.average_clustering(recon)
            clustering_correlations.append((orig_clustering, recon_clustering))
        
        # Compute metrics
        metrics["node_count_accuracy"] = 1.0 - np.mean(node_count_errors) / np.mean([g.number_of_nodes() for g in original_graphs])
        metrics["edge_count_accuracy"] = 1.0 - np.mean(edge_count_errors) / np.mean([g.number_of_edges() for g in original_graphs])
        
        if density_correlations:
            orig_densities, recon_densities = zip(*density_correlations)
            metrics["density_correlation"] = np.corrcoef(orig_densities, recon_densities)[0, 1]
        
        if clustering_correlations:
            orig_clustering, recon_clustering = zip(*clustering_correlations)
            metrics["clustering_correlation"] = np.corrcoef(orig_clustering, recon_clustering)[0, 1]
        
        return metrics
    
    def evaluate_graph_diversity(
        self,
        generated_graphs: List[nx.Graph],
    ) -> Dict[str, float]:
        """Evaluate diversity of generated graphs."""
        
        metrics = {
            "node_count_std": 0.0,
            "edge_count_std": 0.0,
            "density_std": 0.0,
            "clustering_std": 0.0,
        }
        
        if not generated_graphs:
            return metrics
        
        node_counts = [g.number_of_nodes() for g in generated_graphs]
        edge_counts = [g.number_of_edges() for g in generated_graphs]
        densities = [nx.density(g) for g in generated_graphs]
        clustering_coeffs = [nx.average_clustering(g) for g in generated_graphs]
        
        metrics["node_count_std"] = np.std(node_counts)
        metrics["edge_count_std"] = np.std(edge_counts)
        metrics["density_std"] = np.std(densities)
        metrics["clustering_std"] = np.std(clustering_coeffs)
        
        return metrics


class GraphTaskEvaluator:
    """Comprehensive graph task evaluator."""
    
    def __init__(
        self,
        fusion_dim: int,
        num_graph_classes: int = 10,
    ):
        self.fusion_dim = fusion_dim
        self.num_graph_classes = num_graph_classes
        
        # Initialize task evaluators
        self.graph_classification = GraphClassificationTask(
            num_classes=num_graph_classes,
            input_dim=fusion_dim,
        )
        
        self.graph_regression = GraphRegressionTask(
            input_dim=fusion_dim,
        )
        
        self.graph_similarity = GraphSimilarityTask(
            input_dim=fusion_dim,
        )
        
        self.graph_generation = GraphGenerationTask(
            input_dim=fusion_dim,
        )
    
    def evaluate_all_graph_tasks(
        self,
        graph_embeddings: torch.Tensor,
        graph_labels: Optional[torch.Tensor] = None,
        graph_targets: Optional[torch.Tensor] = None,
        similarity_data: Optional[Dict[str, Any]] = None,
        generation_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all graph-based tasks."""
        
        results = {}
        
        # Graph classification
        if graph_labels is not None and self.num_graph_classes > 1:
            # Assume batch dimension is first
            batch_size = graph_embeddings.shape[0]
            batch = torch.arange(batch_size, device=graph_embeddings.device)
            
            classification_outputs = self.graph_classification.forward(graph_embeddings, batch)
            classification_metrics = self.graph_classification.compute_metrics(
                classification_outputs, graph_labels
            )
            results["graph_classification"] = classification_metrics
        
        # Graph regression
        if graph_targets is not None:
            batch_size = graph_embeddings.shape[0]
            batch = torch.arange(batch_size, device=graph_embeddings.device)
            
            regression_outputs = self.graph_regression.forward(graph_embeddings, batch)
            regression_metrics = self.graph_regression.compute_metrics(
                regression_outputs, graph_targets
            )
            results["graph_regression"] = regression_metrics
        
        # Graph similarity
        if similarity_data is not None:
            similarity_metrics = self.graph_similarity.evaluate_similarity_prediction(
                graph_embeddings.detach().cpu().numpy(),
                similarity_data["labels"],
                similarity_data["train_mask"],
                similarity_data["val_mask"],
                similarity_data["test_mask"],
            )
            results["graph_similarity"] = similarity_metrics
        
        # Graph generation
        if generation_data is not None:
            if "original_graphs" in generation_data and "reconstructed_graphs" in generation_data:
                reconstruction_metrics = self.graph_generation.evaluate_graph_reconstruction(
                    generation_data["original_graphs"],
                    generation_data["reconstructed_graphs"],
                )
                results["graph_reconstruction"] = reconstruction_metrics
            
            if "generated_graphs" in generation_data:
                diversity_metrics = self.graph_generation.evaluate_graph_diversity(
                    generation_data["generated_graphs"],
                )
                results["graph_diversity"] = diversity_metrics
        
        return results
