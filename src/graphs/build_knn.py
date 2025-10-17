"""
kNN graph construction for single-cell data.

Builds cell-cell k-nearest neighbor graphs from embeddings.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent
import anndata as ad

logger = logging.getLogger(__name__)


def build_knn_graph(
    adata: ad.AnnData,
    k: int = 30,
    metric: str = "cosine",
    use_embeddings: bool = True,
    embedding_method: str = "scanpy",
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Build k-nearest neighbor graph from single-cell data.
    
    Args:
        adata: AnnData object with expression data
        k: Number of nearest neighbors
        metric: Distance metric ("cosine", "euclidean", "manhattan")
        use_embeddings: Whether to use embeddings instead of raw expression
        embedding_method: Method for computing embeddings ("scanpy", "umap", "pca")
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with kNN edges (source, target, distance)
    """
    logger.info(f"Building kNN graph with k={k}, metric={metric}")
    
    # Get features for kNN computation
    if use_embeddings:
        features = _compute_embeddings(adata, method=embedding_method)
    else:
        features = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    logger.info(f"Using features shape: {features.shape}")
    
    # Build kNN graph
    if metric == "cosine":
        # Use cosine similarity
        knn_model = NearestNeighbors(
            n_neighbors=k + 1,  # +1 to exclude self
            metric="cosine",
            n_jobs=n_jobs
        )
    else:
        knn_model = NearestNeighbors(
            n_neighbors=k + 1,
            metric=metric,
            n_jobs=n_jobs
        )
    
    knn_model.fit(features)
    
    # Find neighbors
    distances, indices = knn_model.kneighbors(features)
    
    # Remove self-connections (first neighbor is always self)
    indices = indices[:, 1:]  # Remove first column (self)
    distances = distances[:, 1:]  # Remove first column (self)
    
    # Create edge list
    edges = []
    cell_ids = adata.obs.index.tolist()
    
    for i, cell_id in enumerate(cell_ids):
        for j, neighbor_idx in enumerate(indices[i]):
            neighbor_id = cell_ids[neighbor_idx]
            distance = distances[i][j]
            
            edges.append({
                'source': cell_id,
                'target': neighbor_id,
                'distance': distance,
                'source_idx': i,
                'target_idx': neighbor_idx
            })
    
    # Create DataFrame
    edges_df = pd.DataFrame(edges)
    
    logger.info(f"Built kNN graph with {len(edges_df)} edges")
    
    return edges_df


def build_knn_graph_fast(
    adata: ad.AnnData,
    k: int = 30,
    metric: str = "cosine",
    use_embeddings: bool = True,
    embedding_method: str = "scanpy",
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Build kNN graph using pynndescent for faster computation.
    
    Args:
        adata: AnnData object with expression data
        k: Number of nearest neighbors
        metric: Distance metric
        use_embeddings: Whether to use embeddings
        embedding_method: Method for computing embeddings
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with kNN edges
    """
    logger.info(f"Building kNN graph (fast) with k={k}, metric={metric}")
    
    # Get features
    if use_embeddings:
        features = _compute_embeddings(adata, method=embedding_method)
    else:
        features = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    logger.info(f"Using features shape: {features.shape}")
    
    # Build kNN graph with pynndescent
    knn_model = NNDescent(
        features,
        n_neighbors=k + 1,  # +1 to exclude self
        metric=metric,
        n_jobs=n_jobs,
        random_state=42
    )
    
    # Find neighbors
    indices, distances = knn_model.query(features, k=k + 1)
    
    # Remove self-connections
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    
    # Create edge list
    edges = []
    cell_ids = adata.obs.index.tolist()
    
    for i, cell_id in enumerate(cell_ids):
        for j, neighbor_idx in enumerate(indices[i]):
            neighbor_id = cell_ids[neighbor_idx]
            distance = distances[i][j]
            
            edges.append({
                'source': cell_id,
                'target': neighbor_id,
                'distance': distance,
                'source_idx': i,
                'target_idx': neighbor_idx
            })
    
    # Create DataFrame
    edges_df = pd.DataFrame(edges)
    
    logger.info(f"Built kNN graph (fast) with {len(edges_df)} edges")
    
    return edges_df


def _compute_embeddings(adata: ad.AnnData, method: str = "scanpy") -> np.ndarray:
    """
    Compute embeddings for kNN graph construction.
    
    Args:
        adata: AnnData object
        method: Embedding method ("scanpy", "umap", "pca")
        
    Returns:
        Embeddings array
    """
    logger.info(f"Computing embeddings using {method}")
    
    if method == "scanpy":
        # Use scanpy's PCA + UMAP pipeline
        adata_copy = adata.copy()
        
        # Normalize and log transform
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_copy.raw = adata_copy
        adata_copy = adata_copy[:, adata_copy.var.highly_variable]
        
        # Scale and PCA
        sc.pp.scale(adata_copy, max_value=10)
        sc.tl.pca(adata_copy, svd_solver='arpack')
        
        # Use PCA components
        embeddings = adata_copy.obsm['X_pca']
        
    elif method == "umap":
        # Use UMAP embeddings
        if 'X_umap' in adata.obsm:
            embeddings = adata.obsm['X_umap']
        else:
            # Compute UMAP if not available
            adata_copy = adata.copy()
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)
            sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata_copy = adata_copy[:, adata_copy.var.highly_variable]
            sc.pp.scale(adata_copy, max_value=10)
            sc.tl.pca(adata_copy, svd_solver='arpack')
            sc.pp.neighbors(adata_copy, n_neighbors=15, n_pcs=40)
            sc.tl.umap(adata_copy)
            embeddings = adata_copy.obsm['X_umap']
    
    elif method == "pca":
        # Use PCA embeddings
        if 'X_pca' in adata.obsm:
            embeddings = adata.obsm['X_pca']
        else:
            # Compute PCA if not available
            adata_copy = adata.copy()
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)
            sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata_copy = adata_copy[:, adata_copy.var.highly_variable]
            sc.pp.scale(adata_copy, max_value=10)
            sc.tl.pca(adata_copy, svd_solver='arpack')
            embeddings = adata_copy.obsm['X_pca']
    
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    logger.info(f"Computed embeddings shape: {embeddings.shape}")
    
    return embeddings


def filter_knn_graph(
    edges_df: pd.DataFrame,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    min_degree: int = 1,
    max_degree: Optional[int] = None
) -> pd.DataFrame:
    """
    Filter kNN graph based on distance and degree criteria.
    
    Args:
        edges_df: DataFrame with kNN edges
        min_distance: Minimum distance threshold
        max_distance: Maximum distance threshold
        min_degree: Minimum degree per node
        max_degree: Maximum degree per node
        
    Returns:
        Filtered edges DataFrame
    """
    logger.info("Filtering kNN graph")
    
    filtered_df = edges_df.copy()
    
    # Filter by distance
    if min_distance is not None:
        filtered_df = filtered_df[filtered_df['distance'] >= min_distance]
        logger.info(f"Filtered by min_distance={min_distance}: {len(filtered_df)} edges")
    
    if max_distance is not None:
        filtered_df = filtered_df[filtered_df['distance'] <= max_distance]
        logger.info(f"Filtered by max_distance={max_distance}: {len(filtered_df)} edges")
    
    # Filter by degree
    if min_degree > 1 or max_degree is not None:
        # Count degrees
        degree_counts = filtered_df['source'].value_counts()
        
        if min_degree > 1:
            valid_sources = degree_counts[degree_counts >= min_degree].index
            filtered_df = filtered_df[filtered_df['source'].isin(valid_sources)]
            logger.info(f"Filtered by min_degree={min_degree}: {len(filtered_df)} edges")
        
        if max_degree is not None:
            valid_sources = degree_counts[degree_counts <= max_degree].index
            filtered_df = filtered_df[filtered_df['source'].isin(valid_sources)]
            logger.info(f"Filtered by max_degree={max_degree}: {len(filtered_df)} edges")
    
    logger.info(f"Final filtered graph has {len(filtered_df)} edges")
    
    return filtered_df


def save_knn_graph(
    edges_df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet"
) -> None:
    """
    Save kNN graph to file.
    
    Args:
        edges_df: DataFrame with kNN edges
        output_path: Output file path
        format: File format ("parquet", "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        edges_df.to_parquet(output_path, index=False)
    elif format == "csv":
        edges_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved kNN graph to {output_path}")


def load_knn_graph(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load kNN graph from file.
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with kNN edges
    """
    input_path = Path(input_path)
    
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        return pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
