"""
Metrics and evaluation utilities for C2S-Scale-Gemma hybrid model.

This module contains various metrics and utilities for evaluating the model
performance across different tasks and modalities.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    adjusted_rand_index, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import networkx as nx


class RepresentationMetrics:
    """Metrics for evaluating representation quality."""
    
    def __init__(self):
        pass
    
    def compute_intrinsic_dimension(
        self, 
        embeddings: np.ndarray, 
        method: str = "twonn"
    ) -> float:
        """Compute intrinsic dimension of embeddings."""
        
        if method == "twonn":
            # Two-NN method
            distances = pdist(embeddings)
            distances = np.sort(distances)
            
            # Find the two nearest neighbors
            r1 = distances[1]  # First nearest neighbor
            r2 = distances[2]  # Second nearest neighbor
            
            # Compute intrinsic dimension
            intrinsic_dim = 1.0 / np.mean(np.log(r2 / r1))
            return intrinsic_dim
        
        elif method == "correlation":
            # Correlation-based method
            corr_matrix = np.corrcoef(embeddings.T)
            eigenvals = np.linalg.eigvals(corr_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Find number of significant eigenvalues
            cumsum = np.cumsum(eigenvals) / np.sum(eigenvals)
            intrinsic_dim = np.argmax(cumsum >= 0.95) + 1
            return intrinsic_dim
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_separation_ratio(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Compute separation ratio between classes."""
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        # Compute within-class and between-class distances
        within_class_distances = []
        between_class_distances = []
        
        for label in unique_labels:
            class_mask = labels == label
            class_embeddings = embeddings[class_mask]
            
            # Within-class distances
            if len(class_embeddings) > 1:
                within_distances = pdist(class_embeddings)
                within_class_distances.extend(within_distances)
            
            # Between-class distances
            other_mask = labels != label
            other_embeddings = embeddings[other_mask]
            if len(other_embeddings) > 0:
                between_distances = pdist(np.vstack([class_embeddings, other_embeddings]))
                between_class_distances.extend(between_distances)
        
        if not within_class_distances or not between_class_distances:
            return 0.0
        
        # Compute separation ratio
        mean_within = np.mean(within_class_distances)
        mean_between = np.mean(between_class_distances)
        
        separation_ratio = mean_between / mean_within
        return separation_ratio
    
    def compute_clustering_metrics(
        self, 
        embeddings: np.ndarray, 
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        
        # Perform clustering
        n_clusters = len(np.unique(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute metrics
        metrics = {
            "silhouette_score": silhouette_score(embeddings, cluster_labels),
            "calinski_harabasz_score": calinski_harabasz_score(embeddings, cluster_labels),
            "davies_bouldin_score": davies_bouldin_score(embeddings, cluster_labels),
            "adjusted_rand_index": adjusted_rand_index(true_labels, cluster_labels),
            "normalized_mutual_info": normalized_mutual_info_score(true_labels, cluster_labels),
        }
        
        return metrics


class CrossModalMetrics:
    """Metrics for evaluating cross-modal alignment."""
    
    def __init__(self):
        pass
    
    def compute_alignment_metrics(
        self,
        graph_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        fused_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """Compute cross-modal alignment metrics."""
        
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
        
        # Compute correlation
        graph_text_corr = pearsonr(
            np.mean(graph_embeddings, axis=1), 
            np.mean(text_embeddings, axis=1)
        )[0]
        
        metrics = {
            "graph_text_similarity": graph_text_sim,
            "graph_fused_similarity": graph_fused_sim,
            "text_fused_similarity": text_fused_sim,
            "alignment_score": alignment_score,
            "cross_modal_coherence": cross_modal_coherence,
            "graph_text_correlation": graph_text_corr,
        }
        
        return metrics
    
    def compute_retrieval_metrics(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        query_labels: np.ndarray,
        candidate_labels: np.ndarray,
        k_values: List[int] = [1, 5, 10, 20, 50],
    ) -> Dict[str, float]:
        """Compute cross-modal retrieval metrics."""
        
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
        
        # Compute mean reciprocal rank (MRR)
        mrr = 0
        for i in range(len(query_labels)):
            sorted_indices = np.argsort(similarities[i])[::-1]
            for rank, idx in enumerate(sorted_indices):
                if candidate_labels[idx] == query_labels[i]:
                    mrr += 1.0 / (rank + 1)
                    break
        
        mrr /= len(query_labels)
        metrics["mean_reciprocal_rank"] = mrr
        
        return metrics


class HyperbolicMetrics:
    """Metrics for evaluating hyperbolic geometry properties."""
    
    def __init__(self, curvature: float = -1.0):
        self.curvature = curvature
    
    def compute_hyperbolic_distance(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """Compute hyperbolic distance between points."""
        
        # Poincaré ball model
        # d(x,y) = arccosh(1 + 2 * ||x-y||² / ((1-||x||²)(1-||y||²)))
        
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        
        # Ensure points are within the unit ball
        x = x / (x_norm[:, np.newaxis] + 1e-8)
        y = y / (y_norm[:, np.newaxis] + 1e-8)
        
        # Compute hyperbolic distance
        diff_norm_sq = np.sum((x - y) ** 2, axis=1)
        x_norm_sq = np.sum(x ** 2, axis=1)
        y_norm_sq = np.sum(y ** 2, axis=1)
        
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
        hyperbolic_dist = np.arccosh(1 + 2 * diff_norm_sq / denominator)
        
        return hyperbolic_dist
    
    def compute_hyperbolic_curvature(
        self, 
        embeddings: np.ndarray
    ) -> float:
        """Estimate hyperbolic curvature from embeddings."""
        
        # Simple estimation based on distance distribution
        distances = pdist(embeddings)
        mean_distance = np.mean(distances)
        
        # Estimate curvature based on mean distance
        estimated_curvature = -1.0 / (mean_distance ** 2)
        return estimated_curvature
    
    def compute_hyperbolic_volume(
        self, 
        embeddings: np.ndarray
    ) -> float:
        """Compute hyperbolic volume of embedding space."""
        
        # Compute volume in hyperbolic space
        # V = (4π/3) * (sinh(r))³ for 3D hyperbolic space
        
        # Estimate radius from embeddings
        max_norm = np.max(np.linalg.norm(embeddings, axis=1))
        
        # Compute volume
        volume = (4 * np.pi / 3) * (np.sinh(max_norm) ** 3)
        return volume


class VisualizationMetrics:
    """Metrics for evaluating visualization quality."""
    
    def __init__(self):
        pass
    
    def compute_visualization_quality(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        n_components: int = 2,
    ) -> Dict[str, float]:
        """Compute visualization quality metrics."""
        
        # Create visualization
        if method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == "umap":
            reducer = UMAP(n_components=n_components, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedded = reducer.fit_transform(embeddings)
        
        # Compute metrics
        metrics = {
            "silhouette_score": silhouette_score(embedded, labels),
            "calinski_harabasz_score": calinski_harabasz_score(embedded, labels),
            "davies_bouldin_score": davies_bouldin_score(embedded, labels),
        }
        
        return metrics
    
    def create_visualization(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_path: str,
        method: str = "tsne",
        n_components: int = 2,
    ) -> None:
        """Create visualization of embeddings."""
        
        # Create visualization
        if method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == "umap":
            reducer = UMAP(n_components=n_components, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedded = reducer.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedded[:, 0], 
            embedded[:, 1], 
            c=labels, 
            cmap="tab10",
            alpha=0.7,
            s=20
        )
        plt.colorbar(scatter)
        plt.title(f"{method.upper()} Visualization of Embeddings")
        plt.xlabel(f"{method.upper()} 1")
        plt.ylabel(f"{method.upper()} 2")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


class ComprehensiveEvaluator:
    """Comprehensive evaluator combining all metrics."""
    
    def __init__(self, curvature: float = -1.0):
        self.representation_metrics = RepresentationMetrics()
        self.cross_modal_metrics = CrossModalMetrics()
        self.hyperbolic_metrics = HyperbolicMetrics(curvature)
        self.visualization_metrics = VisualizationMetrics()
    
    def evaluate_all_metrics(
        self,
        graph_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        fused_embeddings: np.ndarray,
        labels: np.ndarray,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all metrics comprehensively."""
        
        results = {}
        
        # Representation quality
        results["representation"] = {
            "intrinsic_dimension": self.representation_metrics.compute_intrinsic_dimension(fused_embeddings),
            "separation_ratio": self.representation_metrics.compute_separation_ratio(fused_embeddings, labels),
            **self.representation_metrics.compute_clustering_metrics(fused_embeddings, labels),
        }
        
        # Cross-modal alignment
        results["cross_modal"] = self.cross_modal_metrics.compute_alignment_metrics(
            graph_embeddings, text_embeddings, fused_embeddings
        )
        
        # Hyperbolic geometry
        results["hyperbolic"] = {
            "estimated_curvature": self.hyperbolic_metrics.compute_hyperbolic_curvature(fused_embeddings),
            "hyperbolic_volume": self.hyperbolic_metrics.compute_hyperbolic_volume(fused_embeddings),
        }
        
        # Visualization quality
        results["visualization"] = self.visualization_metrics.compute_visualization_quality(
            fused_embeddings, labels
        )
        
        # Create visualizations if output directory is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create t-SNE visualization
            self.visualization_metrics.create_visualization(
                fused_embeddings, labels, 
                str(output_path / "tsne_visualization.png"), 
                method="tsne"
            )
            
            # Create UMAP visualization
            self.visualization_metrics.create_visualization(
                fused_embeddings, labels, 
                str(output_path / "umap_visualization.png"), 
                method="umap"
            )
        
        return results
