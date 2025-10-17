"""
Evaluation tasks for C2S-Scale-Gemma hybrid model.

This module contains various evaluation tasks including cell type classification,
gene expression prediction, and graph-based tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import pandas as pd


class CellTypeClassificationTask:
    """Cell type classification evaluation task."""
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Create classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)
    
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


class GeneExpressionPredictionTask:
    """Gene expression prediction evaluation task."""
    
    def __init__(
        self,
        num_genes: int,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        self.num_genes = num_genes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Create regression head
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.predictor(x)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute regression loss."""
        return nn.MSELoss()(outputs, targets)
    
    def compute_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        metrics = {
            "mse": mean_squared_error(targets_np, outputs_np),
            "mae": mean_absolute_error(targets_np, outputs_np),
            "r2": r2_score(targets_np, outputs_np),
        }
        
        # Compute per-gene metrics
        per_gene_r2 = []
        for i in range(self.num_genes):
            if np.var(targets_np[:, i]) > 0:  # Skip genes with no variance
                r2 = r2_score(targets_np[:, i], outputs_np[:, i])
                per_gene_r2.append(r2)
        
        if per_gene_r2:
            metrics["mean_per_gene_r2"] = np.mean(per_gene_r2)
            metrics["median_per_gene_r2"] = np.median(per_gene_r2)
        
        return metrics


class GraphTaskEvaluator:
    """Graph-based task evaluator."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
    
    def evaluate_node_classification(
        self,
        node_embeddings: np.ndarray,
        node_labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate node classification task."""
        
        # Prepare data
        X_train = node_embeddings[train_mask]
        y_train = node_labels[train_mask]
        X_val = node_embeddings[val_mask]
        y_val = node_labels[val_mask]
        X_test = node_embeddings[test_mask]
        y_test = node_labels[test_mask]
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Evaluate
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        
        metrics = {
            "val_accuracy": accuracy_score(y_val, val_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "val_f1": f1_score(y_val, val_pred, average="weighted"),
            "test_f1": f1_score(y_test, test_pred, average="weighted"),
        }
        
        return metrics
    
    def evaluate_link_prediction(
        self,
        node_embeddings: np.ndarray,
        edge_index: np.ndarray,
        edge_labels: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate link prediction task."""
        
        # Compute edge embeddings
        edge_embeddings = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            edge_emb = np.concatenate([
                node_embeddings[src],
                node_embeddings[dst]
            ])
            edge_embeddings.append(edge_emb)
        
        edge_embeddings = np.array(edge_embeddings)
        
        # Prepare data
        X_train = edge_embeddings[train_mask]
        y_train = edge_labels[train_mask]
        X_val = edge_embeddings[val_mask]
        y_val = edge_labels[val_mask]
        X_test = edge_embeddings[test_mask]
        y_test = edge_labels[test_mask]
        
        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Evaluate
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        val_proba = clf.predict_proba(X_val)[:, 1]
        test_proba = clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            "val_accuracy": accuracy_score(y_val, val_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "test_roc_auc": roc_auc_score(y_test, test_proba),
        }
        
        return metrics


class ClusteringEvaluator:
    """Clustering evaluation."""
    
    def __init__(self):
        pass
    
    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate clustering performance."""
        
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute metrics
        metrics = {
            "adjusted_rand_index": adjusted_rand_score(true_labels, cluster_labels),
            "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels),
            "n_clusters": n_clusters,
        }
        
        return metrics


class DownstreamTaskEvaluator:
    """Comprehensive downstream task evaluator."""
    
    def __init__(
        self,
        fusion_dim: int,
        num_cell_types: int,
        num_genes: int,
    ):
        self.fusion_dim = fusion_dim
        self.num_cell_types = num_cell_types
        self.num_genes = num_genes
        
        # Initialize task evaluators
        self.cell_type_task = CellTypeClassificationTask(
            num_classes=num_cell_types,
            input_dim=fusion_dim,
        )
        
        self.gene_expr_task = GeneExpressionPredictionTask(
            num_genes=num_genes,
            input_dim=fusion_dim,
        )
        
        self.graph_evaluator = GraphTaskEvaluator(input_dim=fusion_dim)
        self.clustering_evaluator = ClusteringEvaluator()
    
    def evaluate_all_tasks(
        self,
        fused_embeddings: torch.Tensor,
        labels: torch.Tensor,
        gene_expressions: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all downstream tasks."""
        
        results = {}
        
        # Cell type classification
        if self.num_cell_types > 1:
            cell_type_outputs = self.cell_type_task.forward(fused_embeddings)
            cell_type_metrics = self.cell_type_task.compute_metrics(
                cell_type_outputs, labels
            )
            results["cell_type_classification"] = cell_type_metrics
        
        # Gene expression prediction
        if gene_expressions is not None:
            gene_expr_outputs = self.gene_expr_task.forward(fused_embeddings)
            gene_expr_metrics = self.gene_expr_task.compute_metrics(
                gene_expr_outputs, gene_expressions
            )
            results["gene_expression_prediction"] = gene_expr_metrics
        
        # Clustering evaluation
        clustering_metrics = self.clustering_evaluator.evaluate_clustering(
            fused_embeddings.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        )
        results["clustering"] = clustering_metrics
        
        # Graph-based tasks
        if graph_data is not None:
            node_embeddings = fused_embeddings.detach().cpu().numpy()
            
            # Node classification
            if "node_labels" in graph_data and "masks" in graph_data:
                node_class_metrics = self.graph_evaluator.evaluate_node_classification(
                    node_embeddings,
                    graph_data["node_labels"],
                    graph_data["masks"]["train"],
                    graph_data["masks"]["val"],
                    graph_data["masks"]["test"],
                )
                results["node_classification"] = node_class_metrics
            
            # Link prediction
            if "edge_index" in graph_data and "edge_labels" in graph_data:
                link_pred_metrics = self.graph_evaluator.evaluate_link_prediction(
                    node_embeddings,
                    graph_data["edge_index"],
                    graph_data["edge_labels"],
                    graph_data["masks"]["train"],
                    graph_data["masks"]["val"],
                    graph_data["masks"]["test"],
                )
                results["link_prediction"] = link_pred_metrics
        
        return results


class CrossModalEvaluator:
    """Cross-modal evaluation."""
    
    def __init__(self):
        pass
    
    def evaluate_alignment(
        self,
        graph_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        fused_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate cross-modal alignment."""
        
        # Normalize embeddings
        graph_norm = graph_embeddings / np.linalg.norm(graph_embeddings, axis=1, keepdims=True)
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        fused_norm = fused_embeddings / np.linalg.norm(fused_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        graph_text_sim = np.mean(np.diag(graph_norm @ text_norm.T))
        graph_fused_sim = np.mean(np.diag(graph_norm @ fused_norm.T))
        text_fused_sim = np.mean(np.diag(text_norm @ fused_norm.T))
        
        # Compute alignment metrics
        alignment_score = (graph_fused_sim + text_fused_sim) / 2
        cross_modal_coherence = graph_text_sim
        
        metrics = {
            "graph_text_similarity": graph_text_sim,
            "graph_fused_similarity": graph_fused_sim,
            "text_fused_similarity": text_fused_sim,
            "alignment_score": alignment_score,
            "cross_modal_coherence": cross_modal_coherence,
        }
        
        return metrics
    
    def evaluate_retrieval(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        query_labels: np.ndarray,
        candidate_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Evaluate cross-modal retrieval."""
        
        # Compute similarities
        similarities = query_embeddings @ candidate_embeddings.T
        
        # Compute retrieval metrics
        metrics = {}
        
        for k in k_values:
            # Get top-k candidates
            top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
            
            # Compute recall@k
            recall_at_k = 0
            for i, top_k in enumerate(top_k_indices):
                if query_labels[i] in candidate_labels[top_k]:
                    recall_at_k += 1
            
            recall_at_k /= len(query_labels)
            metrics[f"recall_at_{k}"] = recall_at_k
        
        return metrics
