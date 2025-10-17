#!/usr/bin/env python3
"""
Data download script for C2S-Scale-Gemma Hybrid project.

Downloads Cell2Sentence dataset from HuggingFace and prepares it for processing.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import scanpy as sc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_cell2sentence_dataset(
    dataset_name: str = "vandijklab/cell2sentence",
    cache_dir: str = "./cache",
    output_dir: str = "./data/raw"
) -> None:
    """
    Download Cell2Sentence dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        cache_dir: Directory to cache downloaded data
        output_dir: Directory to save processed data
    """
    logger.info(f"Downloading dataset: {dataset_name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to load the dataset
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        logger.info(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
        
        # Save each split
        for split_name, split_data in dataset.items():
            output_file = output_path / f"{split_name}.parquet"
            split_data.to_parquet(output_file)
            logger.info(f"Saved {split_name} split to {output_file}")
            
    except Exception as e:
        logger.warning(f"Failed to load dataset {dataset_name}: {e}")
        logger.info("Attempting to download individual files...")
        
        # Try to download individual files
        try:
            # Download specific files that might be available
            files_to_download = [
                "cells.h5ad",
                "metadata.csv", 
                "gene_names.txt",
                "cell_types.csv"
            ]
            
            for filename in files_to_download:
                try:
                    file_path = hf_hub_download(
                        repo_id=dataset_name,
                        filename=filename,
                        cache_dir=cache_dir
                    )
                    # Copy to output directory
                    import shutil
                    shutil.copy2(file_path, output_path / filename)
                    logger.info(f"Downloaded {filename}")
                except Exception as file_e:
                    logger.debug(f"Could not download {filename}: {file_e}")
                    
        except Exception as download_e:
            logger.error(f"Failed to download individual files: {download_e}")
            raise


def download_sample_data(output_dir: str = "./data/raw") -> None:
    """
    Download sample single-cell data for testing if main dataset is not available.
    
    Args:
        output_dir: Directory to save sample data
    """
    logger.info("Downloading sample single-cell data...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use scanpy's built-in datasets
    try:
        # Download a small dataset for testing
        adata = sc.datasets.pbmc3k()
        
        # Save as h5ad
        adata.write(output_path / "pbmc3k.h5ad")
        logger.info(f"Saved sample data to {output_path / 'pbmc3k.h5ad'}")
        
        # Create metadata
        metadata = pd.DataFrame({
            'cell_id': adata.obs.index,
            'cell_type': adata.obs.get('cell_type', 'unknown'),
            'n_genes': adata.obs.get('n_genes', 0),
            'total_counts': adata.obs.get('total_counts', 0)
        })
        metadata.to_csv(output_path / "metadata.csv", index=False)
        logger.info(f"Saved metadata to {output_path / 'metadata.csv'}")
        
    except Exception as e:
        logger.error(f"Failed to download sample data: {e}")
        raise


def create_cell_sentences(adata_path: str, output_dir: str) -> None:
    """
    Convert single-cell data to cell sentences format.
    
    Args:
        adata_path: Path to AnnData file
        output_dir: Directory to save cell sentences
    """
    logger.info(f"Converting {adata_path} to cell sentences...")
    
    # Load data
    adata = sc.read_h5ad(adata_path)
    
    # Get gene names
    if hasattr(adata, 'var_names'):
        gene_names = adata.var_names.tolist()
    else:
        gene_names = [f"Gene_{i}" for i in range(adata.n_vars)]
    
    # Convert to cell sentences
    cell_sentences = []
    for i in range(adata.n_obs):
        # Get expression values for this cell
        expr = adata.X[i].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[i]
        
        # Rank genes by expression (descending)
        ranked_indices = expr.argsort()[::-1]
        
        # Create sentence: gene names ordered by expression
        sentence = " ".join([gene_names[idx] for idx in ranked_indices if expr[idx] > 0])
        cell_sentences.append(sentence)
    
    # Save cell sentences
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sentences_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'cell_sentence': cell_sentences,
        'cell_type': adata.obs.get('cell_type', 'unknown'),
        'tissue': adata.obs.get('tissue', 'unknown')
    })
    
    sentences_df.to_csv(output_path / "cell_sentences.csv", index=False)
    logger.info(f"Saved cell sentences to {output_path / 'cell_sentences.csv'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download Cell2Sentence dataset")
    parser.add_argument(
        "--cfg", 
        type=str, 
        default="configs/datasets.toml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="vandijklab/cell2sentence",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for HuggingFace downloads"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Download only sample data for testing"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.cfg):
        if args.cfg.endswith('.toml'):
            import toml
            with open(args.cfg, 'r') as f:
                cfg_dict = toml.load(f)
            cfg = OmegaConf.create(cfg_dict)
        else:
            cfg = OmegaConf.load(args.cfg)
        logger.info(f"Loaded configuration from {args.cfg}")
    else:
        logger.warning(f"Configuration file {args.cfg} not found, using defaults")
        cfg = OmegaConf.create({})
    
    # Override with command line arguments
    dataset_name = args.dataset_name or cfg.get("dataset_name", "vandijklab/cell2sentence")
    output_dir = args.output_dir or cfg.get("raw_data_dir", "./data/raw")
    cache_dir = args.cache_dir or cfg.get("cache_dir", "./cache")
    
    try:
        if args.sample_only:
            download_sample_data(output_dir)
        else:
            download_cell2sentence_dataset(dataset_name, cache_dir, output_dir)
            
            # Convert to cell sentences if we have h5ad files
            output_path = Path(output_dir)
            h5ad_files = list(output_path.glob("*.h5ad"))
            
            if h5ad_files:
                for h5ad_file in h5ad_files:
                    create_cell_sentences(str(h5ad_file), output_dir)
            else:
                logger.info("No h5ad files found, skipping cell sentence conversion")
                
        logger.info("Data download completed successfully!")
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        raise


if __name__ == "__main__":
    main()
