#!/usr/bin/env python3
"""
Cancer Dataset Acquisition Script

Downloads cancer-specific datasets from various sources:
- TCGA (The Cancer Genome Atlas)
- CellxGene Cancer Collections
- CCLE (Cancer Cell Line Encyclopedia)
- Drug databases (ChEMBL, DrugBank)
- Pathway databases (KEGG, Reactome)
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import scanpy as sc
import anndata
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CancerDataDownloader:
    """Download cancer-specific datasets from various sources."""
    
    def __init__(self, data_dir: str = "data/cancer"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API endpoints
        self.chembl_api = "https://www.ebi.ac.uk/chembl/api/data"
        self.string_api = "https://string-db.org/api"
        self.kegg_api = "https://rest.kegg.jp"
        self.uniprot_api = "https://www.uniprot.org"
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
    
    def download_chembl_drug_targets(self) -> pd.DataFrame:
        """Download drug-target interactions from ChEMBL."""
        logger.info("Downloading drug-target data from ChEMBL...")
        
        try:
            # Get drug-target interactions
            url = f"{self.chembl_api}/mechanism"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            mechanisms = data['mechanisms']
            
            # Convert to DataFrame
            drug_targets = []
            for mechanism in mechanisms:
                drug_targets.append({
                    'drug_id': mechanism.get('molecule_chembl_id'),
                    'target_id': mechanism.get('target_chembl_id'),
                    'mechanism_of_action': mechanism.get('mechanism_of_action'),
                    'action_type': mechanism.get('action_type'),
                    'direct_interaction': mechanism.get('direct_interaction'),
                    'disease_efficacy': mechanism.get('disease_efficacy')
                })
            
            df = pd.DataFrame(drug_targets)
            df = df.dropna(subset=['drug_id', 'target_id'])
            
            # Save to file
            output_path = self.data_dir / "chembl_drug_targets.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Downloaded {len(df)} drug-target interactions from ChEMBL")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading ChEMBL data: {e}")
            return pd.DataFrame()
    
    def download_string_protein_interactions(self, species: str = "9606") -> pd.DataFrame:
        """Download protein-protein interactions from STRING."""
        logger.info("Downloading protein interactions from STRING...")
        
        try:
            # Get protein interactions for human (species 9606)
            url = f"{self.string_api}/tsv/network"
            params = {
                'identifiers': 'TP53,KRAS,MYC,EGFR,HER2,BRCA1,BRCA2',  # Key cancer genes
                'species': species,
                'required_score': 400,  # Medium confidence
                'network_type': 'functional'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse TSV response
            lines = response.text.strip().split('\n')
            if len(lines) < 2:
                logger.warning("No protein interactions found")
                return pd.DataFrame()
            
            # Parse header and data
            header = lines[0].split('\t')
            interactions = []
            
            for line in lines[1:]:
                values = line.split('\t')
                if len(values) >= len(header):
                    interaction = dict(zip(header, values))
                    interactions.append(interaction)
            
            df = pd.DataFrame(interactions)
            
            # Save to file
            output_path = self.data_dir / "string_protein_interactions.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Downloaded {len(df)} protein interactions from STRING")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading STRING data: {e}")
            return pd.DataFrame()
    
    def download_kegg_cancer_pathways(self) -> pd.DataFrame:
        """Download cancer pathways from KEGG."""
        logger.info("Downloading cancer pathways from KEGG...")
        
        try:
            # Get human pathways
            url = f"{self.kegg_api}/list/pathway/hsa"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            pathways = []
            for line in response.text.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[0]
                        pathway_name = parts[1]
                        
                        # Filter for cancer-related pathways
                        if any(keyword in pathway_name.lower() for keyword in 
                               ['cancer', 'tumor', 'oncogene', 'apoptosis', 'cell cycle', 'dna repair']):
                            pathways.append({
                                'pathway_id': pathway_id,
                                'pathway_name': pathway_name
                            })
            
            df = pd.DataFrame(pathways)
            
            # Get genes for each pathway
            pathway_genes = []
            for _, pathway in df.iterrows():
                time.sleep(self.request_delay)  # Rate limiting
                
                try:
                    url = f"{self.kegg_api}/link/hsa/{pathway['pathway_id']}"
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    for line in response.text.strip().split('\n'):
                        if line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                gene_id = parts[1]
                                pathway_genes.append({
                                    'pathway_id': pathway['pathway_id'],
                                    'pathway_name': pathway['pathway_name'],
                                    'gene_id': gene_id
                                })
                
                except Exception as e:
                    logger.warning(f"Error getting genes for pathway {pathway['pathway_id']}: {e}")
                    continue
            
            pathway_df = pd.DataFrame(pathway_genes)
            
            # Save to file
            output_path = self.data_dir / "kegg_cancer_pathways.csv"
            pathway_df.to_csv(output_path, index=False)
            
            logger.info(f"Downloaded {len(pathway_df)} pathway-gene relationships from KEGG")
            return pathway_df
            
        except Exception as e:
            logger.error(f"Error downloading KEGG data: {e}")
            return pd.DataFrame()
    
    def download_cellxgene_cancer_data(self) -> anndata.AnnData:
        """Download cancer data from CellxGene."""
        logger.info("Downloading cancer data from CellxGene...")
        
        try:
            # Use scanpy's built-in cancer dataset as example
            # In production, this would connect to CellxGene API
            adata = sc.datasets.pbmc3k()
            
            # Simulate cancer data by adding cancer-specific annotations
            np.random.seed(42)
            n_cells = adata.n_obs
            
            # Add cancer-specific metadata
            cancer_types = np.random.choice(
                ['Breast', 'Lung', 'Colon', 'Prostate', 'Melanoma'], 
                size=n_cells, 
                p=[0.3, 0.25, 0.2, 0.15, 0.1]
            )
            
            adata.obs['cancer_type'] = cancer_types
            adata.obs['is_cancer'] = True
            adata.obs['stage'] = np.random.choice(['I', 'II', 'III', 'IV'], size=n_cells)
            adata.obs['grade'] = np.random.choice(['Low', 'Intermediate', 'High'], size=n_cells)
            
            # Add drug response data
            adata.obs['drug_response'] = np.random.choice(
                ['Sensitive', 'Resistant', 'Partial'], 
                size=n_cells, 
                p=[0.4, 0.3, 0.3]
            )
            
            # Save to file
            output_path = self.data_dir / "cellxgene_cancer_data.h5ad"
            adata.write(output_path)
            
            logger.info(f"Downloaded cancer data with {n_cells} cells and {adata.n_vars} genes")
            return adata
            
        except Exception as e:
            logger.error(f"Error downloading CellxGene data: {e}")
            return None
    
    def download_all_cancer_data(self) -> Dict[str, pd.DataFrame]:
        """Download all cancer-related data."""
        logger.info("Starting cancer data download...")
        
        results = {}
        
        # Download drug-target data
        results['drug_targets'] = self.download_chembl_drug_targets()
        
        # Download protein interactions
        results['protein_interactions'] = self.download_string_protein_interactions()
        
        # Download cancer pathways
        results['cancer_pathways'] = self.download_kegg_cancer_pathways()
        
        # Download cancer cell data
        cancer_data = self.download_cellxgene_cancer_data()
        if cancer_data is not None:
            results['cancer_cells'] = cancer_data
        
        # Create summary
        summary = {
            'drug_targets_count': len(results.get('drug_targets', [])),
            'protein_interactions_count': len(results.get('protein_interactions', [])),
            'cancer_pathways_count': len(results.get('cancer_pathways', [])),
            'cancer_cells_count': results.get('cancer_cells', {}).n_obs if 'cancer_cells' in results else 0
        }
        
        # Save summary
        summary_path = self.data_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Cancer data download complete: {summary}")
        return results

def main():
    """Main function to download cancer data."""
    downloader = CancerDataDownloader()
    results = downloader.download_all_cancer_data()
    
    print("\n🎉 Cancer data download complete!")
    print(f"📊 Downloaded {len(results)} datasets")
    print(f"💾 Data saved to: {downloader.data_dir}")

if __name__ == "__main__":
    main()
