"""
Ligand-receptor bipartite graph construction.

Builds ligand-receptor interaction graphs from curated databases.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def build_lr_bipartite_graph(
    adata: ad.AnnData,
    lr_database: str = "cellchat",
    min_score: float = 0.5,
    confidence_threshold: float = 0.7,
    lr_pairs_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Build ligand-receptor bipartite graph.
    
    Args:
        adata: AnnData object with expression data
        lr_database: Ligand-receptor database ("cellchat", "cellphonedb", "custom")
        min_score: Minimum interaction score
        confidence_threshold: Confidence threshold for interactions
        lr_pairs_path: Path to custom ligand-receptor pairs file
        
    Returns:
        DataFrame with ligand-receptor edges
    """
    logger.info(f"Building ligand-receptor graph using {lr_database}")
    
    # Get ligand-receptor pairs
    lr_pairs = _get_lr_pairs(lr_database, lr_pairs_path)
    
    # Get gene names
    gene_names = adata.var_names.tolist()
    
    # Filter L-R pairs to genes present in data
    available_genes = set(gene_names)
    lr_pairs = lr_pairs[
        (lr_pairs['ligand'].isin(available_genes)) & 
        (lr_pairs['receptor'].isin(available_genes))
    ]
    
    logger.info(f"Found {len(lr_pairs)} ligand-receptor pairs in data")
    
    # Compute interaction scores
    interactions = []
    
    for _, lr_pair in lr_pairs.iterrows():
        ligand = lr_pair['ligand']
        receptor = lr_pair['receptor']
        
        # Get expression of ligand and receptor
        ligand_expr = adata[:, ligand].X.toarray().flatten() if hasattr(adata[:, ligand].X, 'toarray') else adata[:, ligand].X.flatten()
        receptor_expr = adata[:, receptor].X.toarray().flatten() if hasattr(adata[:, receptor].X, 'toarray') else adata[:, receptor].X.flatten()
        
        # Find cells expressing ligand and receptor
        ligand_cells = np.where(ligand_expr > 0)[0]
        receptor_cells = np.where(receptor_expr > 0)[0]
        
        # Create interactions between ligand and receptor expressing cells
        for lig_cell in ligand_cells:
            for rec_cell in receptor_cells:
                if lig_cell != rec_cell:  # No self-interactions
                    # Compute interaction score
                    score = ligand_expr[lig_cell] * receptor_expr[rec_cell]
                    
                    if score >= min_score:
                        interactions.append({
                            'ligand': ligand,
                            'receptor': receptor,
                            'ligand_cell': adata.obs.index[lig_cell],
                            'receptor_cell': adata.obs.index[rec_cell],
                            'ligand_expr': ligand_expr[lig_cell],
                            'receptor_expr': receptor_expr[rec_cell],
                            'interaction_score': score,
                            'ligand_cell_idx': lig_cell,
                            'receptor_cell_idx': rec_cell
                        })
    
    # Create DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    if len(interactions_df) > 0:
        # Filter by confidence threshold
        interactions_df = interactions_df[interactions_df['interaction_score'] >= confidence_threshold]
        
        logger.info(f"Built ligand-receptor graph with {len(interactions_df)} interactions")
    else:
        logger.warning("No ligand-receptor interactions found")
    
    return interactions_df


def _get_lr_pairs(
    database: str,
    custom_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Get ligand-receptor pairs from database.
    
    Args:
        database: Database name
        custom_path: Path to custom pairs file
        
    Returns:
        DataFrame with ligand-receptor pairs
    """
    if database == "cellchat":
        return _get_cellchat_lr_pairs()
    elif database == "cellphonedb":
        return _get_cellphonedb_lr_pairs()
    elif database == "custom":
        if custom_path is None:
            raise ValueError("custom_path must be provided for custom database")
        return _load_custom_lr_pairs(custom_path)
    else:
        raise ValueError(f"Unknown database: {database}")


def _get_cellchat_lr_pairs() -> pd.DataFrame:
    """
    Get ligand-receptor pairs from CellChat database.
    
    Returns:
        DataFrame with ligand-receptor pairs
    """
    # CellChat ligand-receptor pairs (simplified version)
    # In practice, you would load from the actual CellChat database
    lr_pairs = [
        # Growth factors
        ("VEGFA", "FLT1"), ("VEGFA", "KDR"), ("VEGFA", "FLT4"),
        ("EGF", "EGFR"), ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
        ("FGF2", "FGFR1"), ("FGF2", "FGFR2"), ("FGF2", "FGFR3"),
        
        # Chemokines
        ("CXCL12", "CXCR4"), ("CXCL12", "CXCR7"),
        ("CCL2", "CCR2"), ("CCL5", "CCR5"), ("CCL19", "CCR7"),
        ("CXCL10", "CXCR3"), ("CXCL9", "CXCR3"),
        
        # Interleukins
        ("IL1B", "IL1R1"), ("IL6", "IL6R"), ("IL10", "IL10RA"),
        ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"),
        
        # Notch signaling
        ("DLL1", "NOTCH1"), ("DLL4", "NOTCH1"), ("JAG1", "NOTCH1"),
        ("DLL1", "NOTCH2"), ("DLL4", "NOTCH2"), ("JAG1", "NOTCH2"),
        
        # Wnt signaling
        ("WNT1", "FZD1"), ("WNT3A", "FZD1"), ("WNT5A", "FZD2"),
        
        # Hedgehog signaling
        ("SHH", "PTCH1"), ("SHH", "PTCH2"),
        
        # BMP signaling
        ("BMP2", "BMPR1A"), ("BMP4", "BMPR1A"), ("BMP7", "BMPR1A"),
        
        # Other important pairs
        ("IFNG", "IFNGR1"), ("IFNG", "IFNGR2"),
        ("PDGFA", "PDGFRA"), ("PDGFB", "PDGFRB"),
        ("KITLG", "KIT"), ("FLT3LG", "FLT3"),
    ]
    
    df = pd.DataFrame(lr_pairs, columns=['ligand', 'receptor'])
    df['database'] = 'cellchat'
    df['confidence'] = 1.0  # Default confidence
    
    return df


def _get_cellphonedb_lr_pairs() -> pd.DataFrame:
    """
    Get ligand-receptor pairs from CellPhoneDB database.
    
    Returns:
        DataFrame with ligand-receptor pairs
    """
    # CellPhoneDB ligand-receptor pairs (simplified version)
    # In practice, you would load from the actual CellPhoneDB database
    lr_pairs = [
        # Immune interactions
        ("CD40LG", "CD40"), ("CD80", "CD28"), ("CD86", "CD28"),
        ("CD80", "CTLA4"), ("CD86", "CTLA4"),
        ("ICOSLG", "ICOS"), ("TNFSF4", "TNFRSF4"),
        
        # Adhesion molecules
        ("ITGAL", "ICAM1"), ("ITGAL", "ICAM2"), ("ITGAL", "ICAM3"),
        ("ITGAM", "ICAM1"), ("ITGAM", "ICAM2"),
        ("ITGB1", "VCAM1"), ("ITGB2", "ICAM1"),
        
        # Chemokine receptors
        ("CCL3", "CCR1"), ("CCL4", "CCR1"), ("CCL5", "CCR1"),
        ("CCL2", "CCR2"), ("CCL7", "CCR2"), ("CCL8", "CCR2"),
        ("CCL3", "CCR5"), ("CCL4", "CCR5"), ("CCL5", "CCR5"),
        
        # Growth factor receptors
        ("VEGFA", "FLT1"), ("VEGFA", "KDR"), ("VEGFA", "FLT4"),
        ("EGF", "EGFR"), ("TGFA", "EGFR"), ("AREG", "EGFR"),
        ("TGFB1", "TGFBR1"), ("TGFB2", "TGFBR1"), ("TGFB3", "TGFBR1"),
        
        # Interleukin receptors
        ("IL1A", "IL1R1"), ("IL1B", "IL1R1"), ("IL1RN", "IL1R1"),
        ("IL6", "IL6R"), ("IL11", "IL6R"),
        ("IL10", "IL10RA"), ("IL22", "IL10RB"),
        
        # TNF family
        ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"),
        ("TNFSF10", "TNFRSF10A"), ("TNFSF10", "TNFRSF10B"),
        ("TNFSF11", "TNFRSF11A"), ("TNFSF11", "TNFRSF11B"),
    ]
    
    df = pd.DataFrame(lr_pairs, columns=['ligand', 'receptor'])
    df['database'] = 'cellphonedb'
    df['confidence'] = 1.0  # Default confidence
    
    return df


def _load_custom_lr_pairs(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load custom ligand-receptor pairs from file.
    
    Args:
        file_path: Path to custom pairs file
        
    Returns:
        DataFrame with ligand-receptor pairs
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
    required_cols = ['ligand', 'receptor']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Custom file must contain columns: {required_cols}")
    
    # Add default confidence if not present
    if 'confidence' not in df.columns:
        df['confidence'] = 1.0
    
    if 'database' not in df.columns:
        df['database'] = 'custom'
    
    return df


def filter_lr_interactions(
    interactions_df: pd.DataFrame,
    min_score: float = 0.5,
    max_interactions_per_cell: Optional[int] = None,
    min_interactions_per_pair: int = 1
) -> pd.DataFrame:
    """
    Filter ligand-receptor interactions.
    
    Args:
        interactions_df: DataFrame with interactions
        min_score: Minimum interaction score
        max_interactions_per_cell: Maximum interactions per cell
        min_interactions_per_pair: Minimum interactions per ligand-receptor pair
        
    Returns:
        Filtered interactions DataFrame
    """
    logger.info("Filtering ligand-receptor interactions")
    
    filtered_df = interactions_df.copy()
    
    # Filter by score
    filtered_df = filtered_df[filtered_df['interaction_score'] >= min_score]
    logger.info(f"Filtered by min_score={min_score}: {len(filtered_df)} interactions")
    
    # Filter by interactions per cell
    if max_interactions_per_cell is not None:
        # Count interactions per ligand cell
        lig_counts = filtered_df['ligand_cell'].value_counts()
        valid_lig_cells = lig_counts[lig_counts <= max_interactions_per_cell].index
        filtered_df = filtered_df[filtered_df['ligand_cell'].isin(valid_lig_cells)]
        
        # Count interactions per receptor cell
        rec_counts = filtered_df['receptor_cell'].value_counts()
        valid_rec_cells = rec_counts[rec_counts <= max_interactions_per_cell].index
        filtered_df = filtered_df[filtered_df['receptor_cell'].isin(valid_rec_cells)]
        
        logger.info(f"Filtered by max_interactions_per_cell={max_interactions_per_cell}: {len(filtered_df)} interactions")
    
    # Filter by interactions per pair
    if min_interactions_per_pair > 1:
        pair_counts = filtered_df.groupby(['ligand', 'receptor']).size()
        valid_pairs = pair_counts[pair_counts >= min_interactions_per_pair].index
        filtered_df = filtered_df[
            filtered_df.set_index(['ligand', 'receptor']).index.isin(valid_pairs)
        ]
        logger.info(f"Filtered by min_interactions_per_pair={min_interactions_per_pair}: {len(filtered_df)} interactions")
    
    logger.info(f"Final filtered interactions: {len(filtered_df)}")
    
    return filtered_df


def save_lr_graph(
    interactions_df: pd.DataFrame,
    output_path: Union[str, Path],
    format: str = "parquet"
) -> None:
    """
    Save ligand-receptor graph to file.
    
    Args:
        interactions_df: DataFrame with interactions
        output_path: Output file path
        format: File format ("parquet", "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        interactions_df.to_parquet(output_path, index=False)
    elif format == "csv":
        interactions_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved ligand-receptor graph to {output_path}")


def load_lr_graph(input_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load ligand-receptor graph from file.
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with ligand-receptor interactions
    """
    input_path = Path(input_path)
    
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        return pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
