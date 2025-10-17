"""
Graph utility functions.

Common utilities for graph construction, validation, and processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def validate_graph_data(edges_df: pd.DataFrame, graph_type: str) -> bool:
    """
    Validate graph data format and content.
    
    Args:
        edges_df: DataFrame with graph edges
        graph_type: Type of graph ("knn", "lr", "grn")
        
    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Validating {graph_type} graph data")
    
    # Check required columns
    if graph_type == "knn":
        required_cols = ['source', 'target', 'distance']
    elif graph_type == "lr":
        required_cols = ['ligand', 'receptor', 'ligand_cell', 'receptor_cell']
    elif graph_type == "grn":
        required_cols = ['source', 'target', 'tf', 'target_gene']
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    missing_cols = [col for col in required_cols if col not in edges_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for empty DataFrame
    if len(edges_df) == 0:
        logger.error("Graph data is empty")
        return False
    
    # Check for self-loops (not allowed in most cases)
    if graph_type == "knn":
        self_loops = edges_df[edges_df['source'] == edges_df['target']]
        if len(self_loops) > 0:
            logger.warning(f"Found {len(self_loops)} self-loops in kNN graph")
    
    # Check for duplicate edges
    if graph_type == "knn":
        duplicates = edges_df.duplicated(subset=['source', 'target'])
    elif graph_type == "lr":
        duplicates = edges_df.duplicated(subset=['ligand_cell', 'receptor_cell'])
    elif graph_type == "grn":
        duplicates = edges_df.duplicated(subset=['source', 'target'])
    
    if duplicates.sum() > 0:
        logger.warning(f"Found {duplicates.sum()} duplicate edges")
    
    logger.info(f"{graph_type} graph validation passed")
    return True


def compute_graph_statistics(edges_df: pd.DataFrame, graph_type: str) -> Dict[str, float]:
    """
    Compute graph statistics.
    
    Args:
        edges_df: DataFrame with graph edges
        graph_type: Type of graph
        
    Returns:
        Dictionary with graph statistics
    """
    logger.info(f"Computing {graph_type} graph statistics")
    
    stats = {
        'n_edges': len(edges_df),
        'n_nodes': 0,
        'density': 0.0,
        'avg_degree': 0.0,
        'max_degree': 0,
        'min_degree': 0
    }
    
    if len(edges_df) == 0:
        return stats
    
    # Get unique nodes
    if graph_type == "knn":
        all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
    elif graph_type == "lr":
        all_nodes = set(edges_df['ligand_cell'].tolist() + edges_df['receptor_cell'].tolist())
    elif graph_type == "grn":
        all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
    
    stats['n_nodes'] = len(all_nodes)
    
    if stats['n_nodes'] > 0:
        # Compute density (for undirected graph)
        max_possible_edges = stats['n_nodes'] * (stats['n_nodes'] - 1) / 2
        stats['density'] = stats['n_edges'] / max_possible_edges if max_possible_edges > 0 else 0.0
        
        # Compute degree statistics
        if graph_type == "knn":
            degree_counts = edges_df['source'].value_counts()
        elif graph_type == "lr":
            degree_counts = edges_df['ligand_cell'].value_counts()
        elif graph_type == "grn":
            degree_counts = edges_df['source'].value_counts()
        
        stats['avg_degree'] = degree_counts.mean()
        stats['max_degree'] = degree_counts.max()
        stats['min_degree'] = degree_counts.min()
    
    logger.info(f"{graph_type} graph statistics: {stats}")
    return stats


def create_networkx_graph(edges_df: pd.DataFrame, graph_type: str) -> nx.Graph:
    """
    Create NetworkX graph from edges DataFrame.
    
    Args:
        edges_df: DataFrame with graph edges
        graph_type: Type of graph
        
    Returns:
        NetworkX graph
    """
    logger.info(f"Creating NetworkX {graph_type} graph")
    
    G = nx.Graph()
    
    if graph_type == "knn":
        for _, row in edges_df.iterrows():
            G.add_edge(
                row['source'], 
                row['target'], 
                weight=row['distance'],
                distance=row['distance']
            )
    elif graph_type == "lr":
        for _, row in edges_df.iterrows():
            G.add_edge(
                row['ligand_cell'], 
                row['receptor_cell'],
                ligand=row['ligand'],
                receptor=row['receptor'],
                score=row['interaction_score']
            )
    elif graph_type == "grn":
        for _, row in edges_df.iterrows():
            G.add_edge(
                row['source'], 
                row['target'],
                tf=row['tf'],
                target_gene=row['target_gene']
            )
    
    logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def sample_graph_nodes(
    edges_df: pd.DataFrame,
    n_samples: int,
    graph_type: str,
    strategy: str = "random"
) -> List[str]:
    """
    Sample nodes from graph.
    
    Args:
        edges_df: DataFrame with graph edges
        n_samples: Number of nodes to sample
        graph_type: Type of graph
        strategy: Sampling strategy ("random", "high_degree", "low_degree")
        
    Returns:
        List of sampled node IDs
    """
    logger.info(f"Sampling {n_samples} nodes from {graph_type} graph using {strategy} strategy")
    
    # Get all nodes
    if graph_type == "knn":
        all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
    elif graph_type == "lr":
        all_nodes = set(edges_df['ligand_cell'].tolist() + edges_df['receptor_cell'].tolist())
    elif graph_type == "grn":
        all_nodes = set(edges_df['source'].tolist() + edges_df['target'].tolist())
    
    all_nodes = list(all_nodes)
    
    if len(all_nodes) <= n_samples:
        return all_nodes
    
    if strategy == "random":
        np.random.seed(42)
        sampled_nodes = np.random.choice(all_nodes, size=n_samples, replace=False).tolist()
    elif strategy == "high_degree":
        # Sample nodes with highest degree
        if graph_type == "knn":
            degree_counts = edges_df['source'].value_counts()
        elif graph_type == "lr":
            degree_counts = edges_df['ligand_cell'].value_counts()
        elif graph_type == "grn":
            degree_counts = edges_df['source'].value_counts()
        
        top_nodes = degree_counts.head(n_samples).index.tolist()
        sampled_nodes = top_nodes
    elif strategy == "low_degree":
        # Sample nodes with lowest degree
        if graph_type == "knn":
            degree_counts = edges_df['source'].value_counts()
        elif graph_type == "lr":
            degree_counts = edges_df['ligand_cell'].value_counts()
        elif graph_type == "grn":
            degree_counts = edges_df['source'].value_counts()
        
        bottom_nodes = degree_counts.tail(n_samples).index.tolist()
        sampled_nodes = bottom_nodes
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    logger.info(f"Sampled {len(sampled_nodes)} nodes")
    return sampled_nodes


def filter_graph_by_nodes(
    edges_df: pd.DataFrame,
    node_list: List[str],
    graph_type: str
) -> pd.DataFrame:
    """
    Filter graph to include only specified nodes.
    
    Args:
        edges_df: DataFrame with graph edges
        node_list: List of nodes to keep
        graph_type: Type of graph
        
    Returns:
        Filtered edges DataFrame
    """
    logger.info(f"Filtering {graph_type} graph to {len(node_list)} nodes")
    
    node_set = set(node_list)
    
    if graph_type == "knn":
        filtered_df = edges_df[
            (edges_df['source'].isin(node_set)) & 
            (edges_df['target'].isin(node_set))
        ]
    elif graph_type == "lr":
        filtered_df = edges_df[
            (edges_df['ligand_cell'].isin(node_set)) & 
            (edges_df['receptor_cell'].isin(node_set))
        ]
    elif graph_type == "grn":
        filtered_df = edges_df[
            (edges_df['source'].isin(node_set)) & 
            (edges_df['target'].isin(node_set))
        ]
    
    logger.info(f"Filtered graph: {len(filtered_df)} edges remaining")
    return filtered_df


def merge_graphs(
    graphs: Dict[str, pd.DataFrame],
    graph_types: List[str]
) -> pd.DataFrame:
    """
    Merge multiple graphs into a single DataFrame.
    
    Args:
        graphs: Dictionary mapping graph type to edges DataFrame
        graph_types: List of graph types to merge
        
    Returns:
        Merged edges DataFrame
    """
    logger.info(f"Merging graphs: {graph_types}")
    
    merged_dfs = []
    
    for graph_type in graph_types:
        if graph_type in graphs and len(graphs[graph_type]) > 0:
            df = graphs[graph_type].copy()
            df['graph_type'] = graph_type
            merged_dfs.append(df)
    
    if merged_dfs:
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        logger.info(f"Merged graph: {len(merged_df)} total edges")
    else:
        merged_df = pd.DataFrame()
        logger.warning("No graphs to merge")
    
    return merged_df


def save_graph_summary(
    graphs: Dict[str, pd.DataFrame],
    output_path: Union[str, Path]
) -> None:
    """
    Save graph summary statistics.
    
    Args:
        graphs: Dictionary mapping graph type to edges DataFrame
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    for graph_type, edges_df in graphs.items():
        if len(edges_df) > 0:
            stats = compute_graph_statistics(edges_df, graph_type)
            stats['graph_type'] = graph_type
            summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved graph summary to {output_path}")


def load_graph_summary(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load graph summary statistics.
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with graph statistics
    """
    input_path = Path(input_path)
    return pd.read_csv(input_path)
