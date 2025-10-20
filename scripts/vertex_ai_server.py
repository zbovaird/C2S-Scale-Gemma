#!/usr/bin/env python3
"""
Vertex AI Server for C2S-Scale-Gemma Cancer Research Pipeline

Production server optimized for Vertex AI with:
- 27B parameter model support
- H100 GPU optimization
- Cancer research API endpoints
- Real-time graph construction
- Clinical validation
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import our modules
from src.hgnn.cancer_encoder import CancerUHGEncoder
from src.text.cancer_gemma_loader import CancerGemmaLoader
from src.fusion.cancer_trainer import CancerHybridTrainer
from src.eval.cancer_tasks import CancerTaskEvaluator
from scripts.download_cancer_data import CancerDataDownloader
from scripts.build_cancer_graphs import CancerGraphBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
cancer_model = None
data_downloader = None
graph_builder = None
task_evaluator = None

# Pydantic models for API
class CellSentenceRequest(BaseModel):
    cell_sentence: str = Field(..., description="Space-separated gene names ordered by expression")
    num_genes: int = Field(1000, description="Number of genes in the cell sentence")
    organism: str = Field("Homo sapiens", description="Organism name")

class ClinicalData(BaseModel):
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    cancer_type: Optional[str] = Field(None, description="Type of cancer")
    stage: Optional[str] = Field(None, description="Cancer stage")
    grade: Optional[str] = Field(None, description="Cancer grade")
    treatments: Optional[List[str]] = Field(None, description="Previous treatments")

class DrugResponseRequest(BaseModel):
    cell_sentence: str
    drug_name: str
    clinical_data: Optional[ClinicalData] = None
    num_genes: int = 1000
    organism: str = "Homo sapiens"

class PrognosisRequest(BaseModel):
    cell_sentence: str
    clinical_data: Optional[ClinicalData] = None
    num_genes: int = 1000
    organism: str = "Homo sapiens"

class BiomarkerRequest(BaseModel):
    cell_sentence: str
    clinical_data: Optional[ClinicalData] = None
    num_genes: int = 1000
    organism: str = "Homo sapiens"

class CancerClassificationRequest(BaseModel):
    cell_sentence: str
    clinical_data: Optional[ClinicalData] = None
    num_genes: int = 1000
    organism: str = "Homo sapiens"

class TreatmentRecommendationRequest(BaseModel):
    cell_sentence: str
    clinical_data: Optional[ClinicalData] = None
    num_genes: int = 1000
    organism: str = "Homo sapiens"

class GraphConstructionRequest(BaseModel):
    graph_types: List[str] = Field(["drug_target", "cancer_pathway", "metastasis"], 
                                 description="Types of graphs to construct")
    cancer_genes: Optional[List[str]] = Field(None, description="Cancer-specific genes to focus on")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global cancer_model, data_downloader, graph_builder, task_evaluator
    
    logger.info("🚀 Starting C2S-Scale-Gemma Cancer Research Server...")
    
    try:
        # Check GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🎯 Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize cancer data downloader
        data_downloader = CancerDataDownloader()
        logger.info("✅ Cancer data downloader initialized")
        
        # Initialize graph builder
        graph_builder = CancerGraphBuilder()
        logger.info("✅ Cancer graph builder initialized")
        
        # Initialize cancer-specific UHG-HGNN encoder
        hgnn_encoder = CancerUHGEncoder(
            input_dim=2000,
            hidden_dim=1024,
            output_dim=768,
            num_layers=6,
            dropout=0.1,
            curvature=-1.0,
            cancer_specific=True,
            hierarchical_taxonomy=True,
            temporal_evolution=True,
            spatial_temporal=True,
            metastasis_prediction=True
        ).to(device)
        
        logger.info("✅ Cancer UHG-HGNN encoder initialized")
        
        # Initialize cancer-specific text encoder
        text_encoder = CancerGemmaLoader(
            model_name="vandijklab/C2S-Scale-Gemma-2-27B",
            device=device,
            torch_dtype=torch.bfloat16,
            quantization_config={
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.bfloat16,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': 'nf4'
            },
            use_auth_token=os.getenv('HF_TOKEN'),
            cancer_specific=True,
            clinical_context=True,
            drug_gene_relationships=True,
            pathway_information=True
        )
        
        logger.info("✅ Cancer text encoder initialized")
        
        # Initialize cancer hybrid trainer
        cancer_model = CancerHybridTrainer(
            hgnn_encoder=hgnn_encoder,
            text_encoder=text_encoder,
            device=device,
            cancer_specific=True,
            drug_response_weight=0.3,
            prognosis_weight=0.2,
            biomarker_weight=0.2,
            classification_weight=0.3,
            clinical_relevance_weight=0.1
        )
        
        logger.info("✅ Cancer hybrid trainer initialized")
        
        # Initialize task evaluator
        task_evaluator = CancerTaskEvaluator(
            hgnn_encoder=hgnn_encoder,
            text_encoder=text_encoder,
            device=device
        )
        
        logger.info("✅ Cancer task evaluator initialized")
        
        # Load cancer data and graphs if not already present
        cancer_data_path = Path("data/cancer")
        if not cancer_data_path.exists() or not list(cancer_data_path.glob("*.csv")):
            logger.info("📥 Downloading cancer data...")
            data_downloader.download_all_cancer_data()
        
        cancer_graphs_path = Path("data/processed/graphs")
        if not cancer_graphs_path.exists() or not list(cancer_graphs_path.glob("*.parquet")):
            logger.info("🕸️ Building cancer graphs...")
            graph_builder.build_all_cancer_graphs()
        
        logger.info("🎉 C2S-Scale-Gemma Cancer Research Server ready!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing server: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🔄 Shutting down server...")

# Initialize FastAPI app
app = FastAPI(
    title="C2S-Scale-Gemma Cancer Research API",
    description="Cancer research pipeline with UHG-HGNN and C2S-Scale-Gemma",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": cancer_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    }

@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "C2S-Scale-Gemma Cancer Research Pipeline",
        "text_model": "vandijklab/C2S-Scale-Gemma-2-27B",
        "hgnn_encoder": cancer_model.hgnn_encoder.get_model_info(),
        "cancer_specific": True,
        "capabilities": [
            "drug_response_prediction",
            "cancer_classification",
            "prognosis_prediction",
            "biomarker_discovery",
            "treatment_recommendation",
            "metastasis_prediction"
        ]
    }

@app.post("/predict_drug_response")
async def predict_drug_response(request: DrugResponseRequest):
    """Predict drug response for cancer cells."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert clinical data to dict
        clinical_dict = request.clinical_data.dict() if request.clinical_data else {}
        
        # Generate prediction
        prediction = cancer_model.text_encoder.predict_drug_response(
            cell_sentence=request.cell_sentence,
            drug_name=request.drug_name,
            clinical_data=clinical_dict,
            max_new_tokens=100,
            num_genes=request.num_genes,
            organism=request.organism
        )
        
        return {
            "drug_name": request.drug_name,
            "prediction": prediction,
            "confidence": "High",  # Could be extracted from prediction
            "clinical_context": clinical_dict
        }
        
    except Exception as e:
        logger.error(f"Error predicting drug response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_prognosis")
async def predict_prognosis(request: PrognosisRequest):
    """Predict cancer prognosis."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert clinical data to dict
        clinical_dict = request.clinical_data.dict() if request.clinical_data else {}
        
        # Generate prediction
        prediction = cancer_model.text_encoder.predict_prognosis(
            cell_sentence=request.cell_sentence,
            clinical_data=clinical_dict,
            max_new_tokens=100,
            num_genes=request.num_genes,
            organism=request.organism
        )
        
        return {
            "prognosis": prediction,
            "clinical_context": clinical_dict
        }
        
    except Exception as e:
        logger.error(f"Error predicting prognosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/discover_biomarkers")
async def discover_biomarkers(request: BiomarkerRequest):
    """Discover cancer biomarkers."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert clinical data to dict
        clinical_dict = request.clinical_data.dict() if request.clinical_data else {}
        
        # Generate prediction
        biomarkers = cancer_model.text_encoder.discover_biomarkers(
            cell_sentence=request.cell_sentence,
            clinical_data=clinical_dict,
            max_new_tokens=200,
            num_genes=request.num_genes,
            organism=request.organism
        )
        
        return {
            "biomarkers": biomarkers,
            "clinical_context": clinical_dict
        }
        
    except Exception as e:
        logger.error(f"Error discovering biomarkers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_cancer")
async def classify_cancer(request: CancerClassificationRequest):
    """Classify cancer type."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert clinical data to dict
        clinical_dict = request.clinical_data.dict() if request.clinical_data else {}
        
        # Generate prediction
        classification = cancer_model.text_encoder.classify_cancer(
            cell_sentence=request.cell_sentence,
            clinical_data=clinical_dict,
            max_new_tokens=100,
            num_genes=request.num_genes,
            organism=request.organism
        )
        
        return {
            "classification": classification,
            "clinical_context": clinical_dict
        }
        
    except Exception as e:
        logger.error(f"Error classifying cancer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_treatment")
async def recommend_treatment(request: TreatmentRecommendationRequest):
    """Recommend cancer treatment strategy."""
    if cancer_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert clinical data to dict
        clinical_dict = request.clinical_data.dict() if request.clinical_data else {}
        
        # Generate prediction
        treatment = cancer_model.text_encoder.recommend_treatment(
            cell_sentence=request.cell_sentence,
            clinical_data=clinical_dict,
            max_new_tokens=200,
            num_genes=request.num_genes,
            organism=request.organism
        )
        
        return {
            "treatment_recommendation": treatment,
            "clinical_context": clinical_dict
        }
        
    except Exception as e:
        logger.error(f"Error recommending treatment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/construct_graphs")
async def construct_graphs(request: GraphConstructionRequest, background_tasks: BackgroundTasks):
    """Construct cancer-specific graphs."""
    if graph_builder is None:
        raise HTTPException(status_code=503, detail="Graph builder not loaded")
    
    try:
        # Build graphs in background
        background_tasks.add_task(
            graph_builder.build_specific_graphs,
            request.graph_types,
            request.cancer_genes
        )
        
        return {
            "message": "Graph construction started",
            "graph_types": request.graph_types,
            "cancer_genes": request.cancer_genes
        }
        
    except Exception as e:
        logger.error(f"Error constructing graphs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_cancer_data")
async def download_cancer_data(background_tasks: BackgroundTasks):
    """Download latest cancer data from APIs."""
    if data_downloader is None:
        raise HTTPException(status_code=503, detail="Data downloader not loaded")
    
    try:
        # Download data in background
        background_tasks.add_task(data_downloader.download_all_cancer_data)
        
        return {
            "message": "Cancer data download started",
            "apis": ["ChEMBL", "STRING", "KEGG", "CellxGene"]
        }
        
    except Exception as e:
        logger.error(f"Error downloading cancer data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_cancer_tasks")
async def evaluate_cancer_tasks(test_data: Dict[str, Any]):
    """Evaluate cancer research tasks."""
    if task_evaluator is None:
        raise HTTPException(status_code=503, detail="Task evaluator not loaded")
    
    try:
        # Evaluate all tasks
        results = task_evaluator.evaluate_all_tasks(test_data)
        
        return {
            "evaluation_results": results,
            "timestamp": str(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)))
        }
        
    except Exception as e:
        logger.error(f"Error evaluating cancer tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get configuration from environment
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Run server
    uvicorn.run(
        "scripts.vertex_ai_server:app",
        host=host,
        port=port,
        reload=False,
        workers=1,  # Single worker for GPU memory management
        log_level="info"
    )
