#!/usr/bin/env python3
"""
Cancer Research Pipeline

Complete pipeline for cancer research using C2S-Scale-Gemma hybrid model:
1. Download cancer data
2. Build cancer-specific graphs
3. Train cancer-specific models
4. Evaluate on cancer research tasks
5. Deploy to Vertex AI
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import toml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules
from data.dataset import CancerDataset
from graphs.build_cancer_graphs import CancerGraphBuilder
from hgnn.cancer_encoder import CancerUHGEncoder
from text.cancer_gemma_loader import CancerGemmaLoader
from fusion.cancer_trainer import CancerHybridTrainer
from eval.cancer_tasks import CancerTaskEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CancerResearchPipeline:
    """Complete cancer research pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_downloader = None
        self.graph_builder = None
        self.hgnn_encoder = None
        self.text_encoder = None
        self.trainer = None
        self.evaluator = None
        
        logger.info(f"Initialized cancer research pipeline with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from TOML file."""
        with open(config_path, 'r') as f:
            config = toml.load(f)
        return config
    
    def download_cancer_data(self) -> Dict:
        """Download cancer-specific datasets."""
        logger.info("Step 1: Downloading cancer data...")
        
        try:
            from scripts.download_cancer_data import CancerDataDownloader
            
            self.data_downloader = CancerDataDownloader(
                data_dir=self.config['data']['cancer_data_dir']
            )
            
            results = self.data_downloader.download_all_cancer_data()
            
            logger.info(f"Downloaded {len(results)} cancer datasets")
            return results
            
        except Exception as e:
            logger.error(f"Error downloading cancer data: {e}")
            raise
    
    def build_cancer_graphs(self) -> Dict:
        """Build cancer-specific graphs."""
        logger.info("Step 2: Building cancer-specific graphs...")
        
        try:
            from scripts.build_cancer_graphs import CancerGraphBuilder
            
            self.graph_builder = CancerGraphBuilder(
                data_dir=self.config['data']['cancer_data_dir'],
                output_dir=self.config['data']['cancer_graphs_dir']
            )
            
            graphs = self.graph_builder.build_all_cancer_graphs()
            
            logger.info(f"Built {len(graphs)} cancer graphs")
            return graphs
            
        except Exception as e:
            logger.error(f"Error building cancer graphs: {e}")
            raise
    
    def initialize_models(self):
        """Initialize cancer-specific models."""
        logger.info("Step 3: Initializing cancer-specific models...")
        
        try:
            # Initialize cancer-specific HGNN encoder
            self.hgnn_encoder = CancerUHGEncoder(
                input_dim=self.config['hgnn']['input_dim'],
                hidden_dim=self.config['hgnn']['hidden_dim'],
                output_dim=self.config['hgnn']['output_dim'],
                num_layers=self.config['hgnn']['num_layers'],
                dropout=self.config['hgnn']['dropout'],
                curvature=self.config['hgnn']['curvature'],
                cancer_specific=True,
                hierarchical_taxonomy=self.config['hgnn']['cancer']['hierarchical_taxonomy'],
                temporal_evolution=self.config['hgnn']['cancer']['temporal_evolution'],
                metastasis_prediction=self.config['hgnn']['cancer']['metastasis_prediction']
            ).to(self.device)
            
            # Initialize cancer-specific text encoder
            self.text_encoder = CancerGemmaLoader(
                model_name=self.config['text']['model_name'],
                device=self.device,
                cancer_specific=True,
                clinical_context=self.config['text']['cancer']['clinical_context'],
                drug_gene_relationships=self.config['text']['cancer']['drug_gene_relationships'],
                pathway_information=self.config['text']['cancer']['pathway_information']
            )
            
            # Initialize cancer hybrid trainer
            self.trainer = CancerHybridTrainer(
                hgnn_encoder=self.hgnn_encoder,
                text_encoder=self.text_encoder,
                device=self.device,
                cancer_specific=True,
                drug_response_weight=self.config['training']['cancer']['drug_response_weight'],
                prognosis_weight=self.config['training']['cancer']['prognosis_weight'],
                biomarker_weight=self.config['training']['cancer']['biomarker_weight'],
                classification_weight=self.config['training']['cancer']['classification_weight']
            )
            
            logger.info("Initialized cancer-specific models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def train_cancer_models(self):
        """Train cancer-specific models."""
        logger.info("Step 4: Training cancer-specific models...")
        
        try:
            # Load cancer dataset
            cancer_dataset = CancerDataset(
                data_dir=self.config['data']['cancer_data_dir'],
                graphs_dir=self.config['data']['cancer_graphs_dir'],
                tokenizer=self.text_encoder.tokenizer,
                max_length=self.config['text']['max_length']
            )
            
            # Training configuration
            training_config = {
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['learning_rate'],
                'num_epochs': self.config['training']['num_epochs'],
                'gradient_accumulation_steps': self.config['training']['gradient_accumulation_steps'],
                'warmup_steps': self.config['training']['warmup_steps']
            }
            
            # Train the hybrid model
            training_results = self.trainer.train(
                dataset=cancer_dataset,
                config=training_config
            )
            
            logger.info(f"Training completed: {training_results}")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def evaluate_cancer_tasks(self):
        """Evaluate on cancer research tasks."""
        logger.info("Step 5: Evaluating cancer research tasks...")
        
        try:
            # Initialize evaluator
            self.evaluator = CancerTaskEvaluator(
                hgnn_encoder=self.hgnn_encoder,
                text_encoder=self.text_encoder,
                device=self.device
            )
            
            # Evaluation tasks
            evaluation_tasks = {
                'drug_response_prediction': self.config['evaluation']['cancer_tasks']['drug_response_prediction'],
                'cancer_type_classification': self.config['evaluation']['cancer_tasks']['cancer_type_classification'],
                'prognosis_prediction': self.config['evaluation']['cancer_tasks']['prognosis_prediction'],
                'biomarker_discovery': self.config['evaluation']['cancer_tasks']['biomarker_discovery'],
                'drug_discovery': self.config['evaluation']['cancer_tasks']['drug_discovery']
            }
            
            # Run evaluations
            evaluation_results = {}
            for task_name, enabled in evaluation_tasks.items():
                if enabled:
                    logger.info(f"Evaluating {task_name}...")
                    results = self.evaluator.evaluate_task(task_name)
                    evaluation_results[task_name] = results
            
            # Save evaluation results
            results_path = Path("results") / "cancer_evaluation_results.json"
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Evaluation completed: {evaluation_results}")
            
        except Exception as e:
            logger.error(f"Error evaluating tasks: {e}")
            raise
    
    def deploy_to_vertex_ai(self):
        """Deploy to Vertex AI for production."""
        logger.info("Step 6: Deploying to Vertex AI...")
        
        try:
            # Create Vertex AI deployment configuration
            vertex_config = {
                'project_id': self.config['vertex_ai']['project_id'],
                'region': self.config['vertex_ai']['region'],
                'bucket_name': self.config['vertex_ai']['bucket_name'],
                'machine_type': self.config['vertex_ai']['compute']['machine_type'],
                'accelerator_type': self.config['vertex_ai']['compute']['accelerator_type'],
                'accelerator_count': self.config['vertex_ai']['compute']['accelerator_count']
            }
            
            # Export models for deployment
            self.trainer.export_for_deployment(
                output_dir="deployment_artifacts",
                vertex_config=vertex_config
            )
            
            logger.info("Deployment artifacts created for Vertex AI")
            
        except Exception as e:
            logger.error(f"Error deploying to Vertex AI: {e}")
            raise
    
    def run_complete_pipeline(self):
        """Run the complete cancer research pipeline."""
        logger.info("Starting complete cancer research pipeline...")
        
        try:
            # Step 1: Download cancer data
            cancer_data = self.download_cancer_data()
            
            # Step 2: Build cancer graphs
            cancer_graphs = self.build_cancer_graphs()
            
            # Step 3: Initialize models
            self.initialize_models()
            
            # Step 4: Train models
            self.train_cancer_models()
            
            # Step 5: Evaluate tasks
            self.evaluate_cancer_tasks()
            
            # Step 6: Deploy to Vertex AI
            self.deploy_to_vertex_ai()
            
            logger.info("🎉 Cancer research pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cancer Research Pipeline")
    parser.add_argument("--cfg", required=True, help="Configuration file path")
    parser.add_argument("--step", choices=["download", "graphs", "train", "evaluate", "deploy", "all"], 
                       default="all", help="Pipeline step to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CancerResearchPipeline(args.cfg)
    
    # Run specified step
    if args.step == "download":
        pipeline.download_cancer_data()
    elif args.step == "graphs":
        pipeline.build_cancer_graphs()
    elif args.step == "train":
        pipeline.initialize_models()
        pipeline.train_cancer_models()
    elif args.step == "evaluate":
        pipeline.evaluate_cancer_tasks()
    elif args.step == "deploy":
        pipeline.deploy_to_vertex_ai()
    elif args.step == "all":
        pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()
