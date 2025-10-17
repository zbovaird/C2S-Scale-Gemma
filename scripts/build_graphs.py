#!/usr/bin/env python3
"""
Graph construction script for C2S-Scale-Gemma Hybrid project.

Builds kNN, ligand-receptor, and gene regulatory network graphs from single-cell data.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import scanpy as sc
from omegaconf import OmegaConf

from src.graphs.build_knn import build_knn_graph, build_knn_graph_fast, save_knn_graph
from src.graphs.build_lr_bipartite import build_lr_bipartite_graph, save_lr_graph
from src.graphs.build_grn import build_grn_graph, save_grn_graph
from src.graphs.utils import validate_graph_data, compute_graph_statistics, save_graph_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_all_graphs(
    adata_path: str,
    output_dir: str,
    config: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Build all graph types from single-cell data.
    
    Args:
        adata_path: Path to AnnData file
        output_dir: Output directory for graphs
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping graph type to edges DataFrame
    """
    logger.info(f"Building graphs from {adata_path}")
    
    # Load data
    adata = sc.read_h5ad(adata_path)
    logger.info(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    graphs = {}
    
    # Build kNN graph
    if config.get('knn_graph', {}).get('enabled', True):
        logger.info("Building kNN graph...")
        try:
            knn_config = config['knn_graph']
            knn_edges = build_knn_graph(
                adata,
                k=knn_config.get('k', 30),
                metric=knn_config.get('metric', 'cosine'),
                use_embeddings=knn_config.get('use_embeddings', True),
                embedding_method=knn_config.get('embedding_method', 'scanpy'),
                n_jobs=knn_config.get('n_jobs', -1)
            )
            
            # Validate and save
            if validate_graph_data(knn_edges, 'knn'):
                graphs['knn'] = knn_edges
                save_knn_graph(knn_edges, output_path / "knn.parquet")
                logger.info(f"Saved kNN graph: {len(knn_edges)} edges")
            else:
                logger.error("kNN graph validation failed")
                
        except Exception as e:
            logger.error(f"Failed to build kNN graph: {e}")
    
    # Build ligand-receptor graph
    if config.get('lr_graph', {}).get('enabled', True):
        logger.info("Building ligand-receptor graph...")
        try:
            lr_config = config['lr_graph']
            lr_edges = build_lr_bipartite_graph(
                adata,
                lr_database=lr_config.get('lr_database', 'cellchat'),
                min_score=lr_config.get('min_score', 0.5),
                confidence_threshold=lr_config.get('confidence_threshold', 0.7),
                lr_pairs_path=lr_config.get('lr_pairs_path')
            )
            
            # Validate and save
            if validate_graph_data(lr_edges, 'lr'):
                graphs['lr'] = lr_edges
                save_lr_graph(lr_edges, output_path / "lr.parquet")
                logger.info(f"Saved ligand-receptor graph: {len(lr_edges)} edges")
            else:
                logger.error("Ligand-receptor graph validation failed")
                
        except Exception as e:
            logger.error(f"Failed to build ligand-receptor graph: {e}")
    
    # Build GRN graph
    if config.get('grn_graph', {}).get('enabled', True):
        logger.info("Building GRN graph...")
        try:
            grn_config = config['grn_graph']
            grn_edges = build_grn_graph(
                adata,
                grn_method=grn_config.get('grn_method', 'scenic'),
                min_regulon_size=grn_config.get('min_regulon_size', 10),
                max_regulon_size=grn_config.get('max_regulon_size', 1000),
                auc_threshold=grn_config.get('auc_threshold', 0.01),
                grn_file=grn_config.get('grn_file')
            )
            
            # Validate and save
            if validate_graph_data(grn_edges, 'grn'):
                graphs['grn'] = grn_edges
                save_grn_graph(grn_edges, output_path / "grn.parquet")
                logger.info(f"Saved GRN graph: {len(grn_edges)} edges")
            else:
                logger.error("GRN graph validation failed")
                
        except Exception as e:
            logger.error(f"Failed to build GRN graph: {e}")
    
    # Save graph summary
    if graphs:
        save_graph_summary(graphs, output_path / "graph_summary.csv")
        logger.info("Saved graph summary")
    
    return graphs


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build graphs from single-cell data")
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="configs/datasets.toml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to AnnData file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed/graphs",
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--knn-only",
        action="store_true",
        help="Build only kNN graph"
    )
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="Build only ligand-receptor graph"
    )
    parser.add_argument(
        "--grn-only",
        action="store_true",
        help="Build only GRN graph"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.cfg):
        cfg = OmegaConf.load(args.cfg)
        logger.info(f"Loaded configuration from {args.cfg}")
    else:
        logger.warning(f"Configuration file {args.cfg} not found, using defaults")
        cfg = OmegaConf.create({})
    
    # Override configuration based on command line arguments
    if args.knn_only:
        cfg['lr_graph'] = {'enabled': False}
        cfg['grn_graph'] = {'enabled': False}
    elif args.lr_only:
        cfg['knn_graph'] = {'enabled': False}
        cfg['grn_graph'] = {'enabled': False}
    elif args.grn_only:
        cfg['knn_graph'] = {'enabled': False}
        cfg['lr_graph'] = {'enabled': False}
    
    # Set default configurations if not present
    if 'knn_graph' not in cfg:
        cfg['knn_graph'] = {
            'enabled': True,
            'k': 30,
            'metric': 'cosine',
            'use_embeddings': True,
            'embedding_method': 'scanpy',
            'n_jobs': -1
        }
    
    if 'lr_graph' not in cfg:
        cfg['lr_graph'] = {
            'enabled': True,
            'lr_database': 'cellchat',
            'min_score': 0.5,
            'confidence_threshold': 0.7
        }
    
    if 'grn_graph' not in cfg:
        cfg['grn_graph'] = {
            'enabled': True,
            'grn_method': 'scenic',
            'min_regulon_size': 10,
            'max_regulon_size': 1000,
            'auc_threshold': 0.01
        }
    
    try:
        # Build graphs
        graphs = build_all_graphs(args.data_path, args.output_dir, cfg)
        
        # Print summary
        logger.info("Graph construction completed!")
        for graph_type, edges_df in graphs.items():
            stats = compute_graph_statistics(edges_df, graph_type)
            logger.info(f"{graph_type.upper()} graph: {stats['n_edges']} edges, {stats['n_nodes']} nodes")
        
    except Exception as e:
        logger.error(f"Graph construction failed: {e}")
        raise


if __name__ == "__main__":
    main()
