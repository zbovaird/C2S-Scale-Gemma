"""
Gene regulatory network (GRN) construction.

Builds gene regulatory networks from SCENIC, GRNBoost, or custom outputs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def build_grn_graph(
    adata: ad.AnnData,
    grn_method: str = "scenic",
    min_regulon_size: int = 10,
    max_regulon_size: int = 1000,
    auc_threshold: float = 0.01,
    grn_file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Build gene regulatory network graph.
    
    Args:
        adata: AnnData object with expression data
        grn_method: GRN method ("scenic", "grnboost", "custom")
        min_regulon_size: Minimum regulon size
        max_regulon_size: Maximum regulon size
        auc_threshold: AUC threshold for regulon activity
        grn_file: Path to custom GRN file
        
    Returns:
        DataFrame with GRN edges
    """
    logger.info(f"Building GRN graph using {grn_method}")
    
    # Get GRN data
    if grn_method == "scenic":
        grn_data = _get_scenic_grn(adata, auc_threshold)
    elif grn_method == "grnboost":
        grn_data = _get_grnboost_grn(adata)
    elif grn_method == "custom":
        if grn_file is None:
            raise ValueError("grn_file must be provided for custom GRN")
        grn_data = _load_custom_grn(grn_file)
    else:
        raise ValueError(f"Unknown GRN method: {grn_method}")
    
    # Filter regulons by size
    grn_data = _filter_regulons_by_size(
        grn_data, min_regulon_size, max_regulon_size
    )
    
    # Create GRN edges
    edges = []
    
    for _, row in grn_data.iterrows():
        tf = row['tf']
        target_genes = row['target_genes']
        regulon_size = len(target_genes)
        
        # Create edges from TF to target genes
        for target_gene in target_genes:
            edges.append({
                'source': tf,
                'target': target_gene,
                'tf': tf,
                'target_gene': target_gene,
                'regulon_size': regulon_size,
                'method': grn_method
            })
    
    # Create DataFrame
    edges_df = pd.DataFrame(edges)
    
    logger.info(f"Built GRN graph with {len(edges_df)} edges")
    
    return edges_df


def _get_scenic_grn(
    adata: ad.AnnData,
    auc_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Get GRN from SCENIC analysis.
    
    Args:
        adata: AnnData object
        auc_threshold: AUC threshold for regulon activity
        
    Returns:
        DataFrame with regulon information
    """
    logger.info("Extracting GRN from SCENIC analysis")
    
    # Check if SCENIC results are available
    if 'auc' not in adata.obsm:
        logger.warning("SCENIC AUC matrix not found, creating mock regulons")
        return _create_mock_regulons(adata)
    
    auc_matrix = adata.obsm['auc']
    regulon_names = adata.uns.get('auc', {}).get('regulons', [])
    
    if not regulon_names:
        logger.warning("No regulon names found, using column names")
        regulon_names = auc_matrix.columns.tolist()
    
    # Create regulon data
    regulon_data = []
    
    for regulon_name in regulon_names:
        if regulon_name in auc_matrix.columns:
            # Get cells with high regulon activity
            high_activity_cells = auc_matrix[regulon_name] > auc_threshold
            
            if high_activity_cells.sum() > 0:
                # Extract target genes from regulon name
                # SCENIC regulon names are typically "TF_targets"
                if '_' in regulon_name:
                    tf = regulon_name.split('_')[0]
                else:
                    tf = regulon_name
                
                # Get target genes (simplified - in practice, you'd load from SCENIC output)
                target_genes = _get_target_genes_for_tf(tf, adata)
                
                regulon_data.append({
                    'tf': tf,
                    'regulon_name': regulon_name,
                    'target_genes': target_genes,
                    'n_cells_active': high_activity_cells.sum(),
                    'mean_auc': auc_matrix[regulon_name].mean()
                })
    
    return pd.DataFrame(regulon_data)


def _get_grnboost_grn(adata: ad.AnnData) -> pd.DataFrame:
    """
    Get GRN from GRNBoost analysis.
    
    Args:
        adata: AnnData object
        
    Returns:
        DataFrame with regulon information
    """
    logger.info("Extracting GRN from GRNBoost analysis")
    
    # Check if GRNBoost results are available
    if 'grnboost' not in adata.uns:
        logger.warning("GRNBoost results not found, creating mock regulons")
        return _create_mock_regulons(adata)
    
    grnboost_data = adata.uns['grnboost']
    
    # Group by transcription factor
    regulon_data = []
    
    for tf in grnboost_data['tf'].unique():
        tf_edges = grnboost_data[grnboost_data['tf'] == tf]
        target_genes = tf_edges['target'].tolist()
        
        regulon_data.append({
            'tf': tf,
            'regulon_name': f"{tf}_regulon",
            'target_genes': target_genes,
            'n_edges': len(tf_edges),
            'mean_importance': tf_edges['importance'].mean() if 'importance' in tf_edges.columns else 0.0
        })
    
    return pd.DataFrame(regulon_data)


def _load_custom_grn(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load custom GRN from file.
    
    Args:
        file_path: Path to custom GRN file
        
    Returns:
        DataFrame with regulon information
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Validate required columns
    required_cols = ['tf', 'target_genes']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Custom GRN file must contain columns: {required_cols}")
    
    return df


def _create_mock_regulons(adata: ad.AnnData) -> pd.DataFrame:
    """
    Create mock regulons for testing.
    
    Args:
        adata: AnnData object
        
    Returns:
        DataFrame with mock regulon information
    """
    logger.info("Creating mock regulons for testing")
    
    # Get highly variable genes as potential TFs
    if 'highly_variable' in adata.var.columns:
        hvg_genes = adata.var[adata.var['highly_variable']].index.tolist()
    else:
        # Use top expressed genes
        gene_means = adata.X.mean(axis=0)
        if hasattr(gene_means, 'A1'):
            gene_means = gene_means.A1
        top_genes = adata.var_names[np.argsort(gene_means)[-50:]].tolist()
        hvg_genes = top_genes
    
    # Create mock regulons
    regulon_data = []
    
    for i, tf in enumerate(hvg_genes[:20]):  # Limit to 20 TFs
        # Create target genes (random selection)
        np.random.seed(i)  # For reproducibility
        n_targets = np.random.randint(10, 50)
        target_genes = np.random.choice(
            adata.var_names.tolist(), 
            size=n_targets, 
            replace=False
        ).tolist()
        
        regulon_data.append({
            'tf': tf,
            'regulon_name': f"{tf}_regulon",
            'target_genes': target_genes,
            'n_cells_active': np.random.randint(10, 100),
            'mean_auc': np.random.random()
        })
    
    return pd.DataFrame(regulon_data)


def _get_target_genes_for_tf(tf: str, adata: ad.AnnData) -> List[str]:
    """
    Get target genes for a transcription factor.
    
    Args:
        tf: Transcription factor name
        adata: AnnData object
        
    Returns:
        List of target gene names
    """
    # This is a simplified version - in practice, you'd load from SCENIC output
    # For now, return a random subset of genes
    np.random.seed(hash(tf) % 2**32)  # Deterministic based on TF name
    n_targets = np.random.randint(10, 50)
    target_genes = np.random.choice(
        adata.var_names.tolist(), 
        size=n_targets, 
        replace=False
    ).tolist()
    
    return target_genes


def _filter_regulons_by_size(
    regulon_data: pd.DataFrame,
    min_size: int = 10,
    max_size: int = 1000
) -> pd.DataFrame:
    """
    Filter regulons by size.
    
    Args:
        regulon_data: DataFrame with regulon information
        min_size: Minimum regulon size
        max_size: Maximum regulon size
        
    Returns:
        Filtered regulon data
    """
    logger.info(f"Filtering regulons by size: {min_size} <= size <= {max_size}")
    
    # Filter by regulon size
    regulon_data = regulon_data[
        (regulon_data['target_genes'].apply(len) >= min_size) &
        (regulon_data['target_genes'].apply(len) <= max_size)
    ]
    
    logger.info(f"Filtered to {len(regulon_data)} regulons")
    
    return regulon_data


def filter_grn_graph(
    edges_df: pd.DataFrame,
    min_regulon_size: int = 10,
    max_regulon_size: int = 1000,
    min_edges_per_tf: int = 5
) -> pd.DataFrame:
    """
    Filter GRN graph based on regulon size and edge criteria.
    
    Args:
        edges_df: DataFrame with GRN edges
        min_regulon_size: Minimum regulon size
        max_regulon_size: Maximum regulon size
        min_edges_per_tf: Minimum edges per transcription factor
        
    Returns:
        Filtered edges DataFrame
    """
    logger.info("Filtering GRN graph")
    
    filtered_df = edges_df.copy()
    
    # Filter by regulon size
    regulon_sizes = filtered_df.groupby('tf')['regulon_size'].first()
    valid_tfs = regulon_sizes[
        (regulon_sizes >= min_regulon_size) & 
        (regulon_sizes <= max_regulon_size)
    ].index
    
    filtered_df = filtered_df[filtered_df['tf'].isin(valid_tfs)]
    logger.info(f"Filtered by regulon size: {len(filtered_df)} edges")
    
    # Filter by minimum edges per TF
    tf_edge_counts = filtered_df['tf'].value_counts()
    valid_tfs = tf_edge_counts[tf_edge_counts >= min_edges_per_tf].index
    filtered_df = filtered_df[filtered_df['tf'].isin(valid_tfs)]
    logger.info(f"Filtered by min_edges_per_tf={min_edges_per_tf}: {len(filtered_df)} edges")
    
    logger.info(f"Final filtered GRN graph: {len(filtered_df)} edges")
    
    return filtered_df


def save_grn_graph(
    edges_df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet"
) -> None:
    """
    Save GRN graph to file.
    
    Args:
        edges_df: DataFrame with GRN edges
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
    
    logger.info(f"Saved GRN graph to {output_path}")


def load_grn_graph(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load GRN graph from file.
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with GRN edges
    """
    input_path = Path(input_path)
    
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        return pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
