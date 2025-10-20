#!/usr/bin/env python3
"""
Vertex AI Deployment Configuration

Deploy C2S-Scale-Gemma Cancer Research Pipeline to Vertex AI with:
- H100 GPU optimization
- 27B parameter model support
- Cancer research API endpoints
- Auto-scaling configuration
- Monitoring and logging
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class VertexAIDeployment:
    """Vertex AI deployment configuration."""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.deployment_dir = Path("deployment/vertex_ai")
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dockerfile(self):
        """Create optimized Dockerfile for Vertex AI."""
        dockerfile_content = """# Dockerfile for C2S-Scale-Gemma Cancer Research Pipeline
# Optimized for Vertex AI with H100 GPUs

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3.10-dev \\
    python3-pip \\
    git \\
    curl \\
    wget \\
    build-essential \\
    cmake \\
    pkg-config \\
    libhdf5-dev \\
    libssl-dev \\
    libffi-dev \\
    libxml2-dev \\
    libxslt1-dev \\
    zlib1g-dev \\
    libjpeg-dev \\
    libpng-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN uv sync --frozen

# Create directories for data and models
RUN mkdir -p /app/data/cancer /app/data/processed/graphs /app/models /app/logs

# Set up HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Expose port for Vertex AI
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "scripts/vertex_ai_server.py"]
"""
        
        dockerfile_path = self.deployment_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Created Dockerfile at {dockerfile_path}")
    
    def create_vertex_ai_config(self):
        """Create Vertex AI configuration."""
        config = {
            "project_id": self.project_id,
            "region": self.region,
            "model": {
                "display_name": "c2s-scale-gemma-cancer-research",
                "description": "C2S-Scale-Gemma Cancer Research Pipeline with UHG-HGNN",
                "container_spec": {
                    "image_uri": f"gcr.io/{self.project_id}/c2s-scale-gemma-cancer:latest",
                    "command": ["python", "scripts/vertex_ai_server.py"],
                    "args": [],
                    "env": [
                        {"name": "PORT", "value": "8080"},
                        {"name": "HOST", "value": "0.0.0.0"},
                        {"name": "HF_TOKEN", "value": "${HF_TOKEN}"}
                    ],
                    "ports": [{"container_port": 8080}],
                    "resources": {
                        "cpu": "8",
                        "memory": "64Gi",
                        "gpu": {
                            "type": "NVIDIA_H100_80GB",
                            "count": "1"
                        }
                    }
                }
            },
            "endpoint": {
                "display_name": "c2s-scale-gemma-cancer-endpoint",
                "description": "Cancer research API endpoint",
                "traffic_split": {
                    "c2s-scale-gemma-cancer-research": 100
                }
            },
            "deployment": {
                "machine_type": "a2-ultragpu-8g",
                "accelerator_type": "NVIDIA_H100_80GB",
                "accelerator_count": 1,
                "min_replica_count": 1,
                "max_replica_count": 10,
                "autoscaling": {
                    "min_replicas": 1,
                    "max_replicas": 10,
                    "target_cpu_utilization": 70,
                    "target_memory_utilization": 80
                }
            },
            "monitoring": {
                "enable_request_logging": True,
                "enable_response_logging": True,
                "sampling_rate": 1.0
            }
        }
        
        config_path = self.deployment_dir / "vertex_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created Vertex AI config at {config_path}")
        return config
    
    def create_deployment_script(self):
        """Create deployment script."""
        script_content = f"""#!/bin/bash
# Vertex AI Deployment Script for C2S-Scale-Gemma Cancer Research Pipeline

set -e

PROJECT_ID="{self.project_id}"
REGION="{self.region}"
IMAGE_NAME="c2s-scale-gemma-cancer"
IMAGE_TAG="latest"
IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"

echo "🚀 Deploying C2S-Scale-Gemma Cancer Research Pipeline to Vertex AI"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Image URI: $IMAGE_URI"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable compute.googleapis.com

# Build and push Docker image
echo "🐳 Building and pushing Docker image..."
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# Create Vertex AI model
echo "🤖 Creating Vertex AI model..."
gcloud ai models upload \\
    --region=$REGION \\
    --display-name="c2s-scale-gemma-cancer-research" \\
    --description="C2S-Scale-Gemma Cancer Research Pipeline with UHG-HGNN" \\
    --container-image-uri=$IMAGE_URI \\
    --container-ports=8080 \\
    --container-env-vars="PORT=8080,HOST=0.0.0.0" \\
    --machine-type="a2-ultragpu-8g" \\
    --accelerator-type="NVIDIA_H100_80GB" \\
    --accelerator-count=1

# Create endpoint
echo "🌐 Creating endpoint..."
gcloud ai endpoints create \\
    --region=$REGION \\
    --display-name="c2s-scale-gemma-cancer-endpoint" \\
    --description="Cancer research API endpoint"

# Deploy model to endpoint
echo "🚀 Deploying model to endpoint..."
ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="displayName=c2s-scale-gemma-cancer-endpoint" --format="value(name)" | head -1)
MODEL_ID=$(gcloud ai models list --region=$REGION --filter="displayName=c2s-scale-gemma-cancer-research" --format="value(name)" | head -1)

gcloud ai endpoints deploy-model $ENDPOINT_ID \\
    --region=$REGION \\
    --model=$MODEL_ID \\
    --display-name="c2s-scale-gemma-cancer-deployment" \\
    --machine-type="a2-ultragpu-8g" \\
    --accelerator-type="NVIDIA_H100_80GB" \\
    --accelerator-count=1 \\
    --min-replica-count=1 \\
    --max-replica-count=10 \\
    --traffic-allocation=100

echo "✅ Deployment complete!"
echo "Endpoint ID: $ENDPOINT_ID"
echo "Model ID: $MODEL_ID"

# Get endpoint URL
ENDPOINT_URL=$(gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION --format="value(publicEndpoint)")
echo "🌐 Endpoint URL: $ENDPOINT_URL"

# Test endpoint
echo "🧪 Testing endpoint..."
curl -X GET "$ENDPOINT_URL/health" || echo "⚠️ Health check failed - endpoint may still be starting"

echo "🎉 C2S-Scale-Gemma Cancer Research Pipeline deployed successfully!"
echo "📊 Monitor deployment: https://console.cloud.google.com/vertex-ai/endpoints?project=$PROJECT_ID"
"""
        
        script_path = self.deployment_dir / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created deployment script at {script_path}")
    
    def create_kubernetes_config(self):
        """Create Kubernetes configuration for Vertex AI."""
        k8s_config = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "c2s-scale-gemma-cancer",
                "labels": {
                    "app": "c2s-scale-gemma-cancer"
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "c2s-scale-gemma-cancer"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "c2s-scale-gemma-cancer"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "c2s-scale-gemma-cancer",
                                "image": f"gcr.io/{self.project_id}/c2s-scale-gemma-cancer:latest",
                                "ports": [
                                    {
                                        "containerPort": 8080
                                    }
                                ],
                                "env": [
                                    {
                                        "name": "PORT",
                                        "value": "8080"
                                    },
                                    {
                                        "name": "HOST",
                                        "value": "0.0.0.0"
                                    },
                                    {
                                        "name": "HF_TOKEN",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "huggingface-token",
                                                "key": "token"
                                            }
                                        }
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "8",
                                        "memory": "64Gi",
                                        "nvidia.com/gpu": "1"
                                    },
                                    "limits": {
                                        "cpu": "8",
                                        "memory": "64Gi",
                                        "nvidia.com/gpu": "1"
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                }
                            }
                        ],
                        "nodeSelector": {
                            "cloud.google.com/gke-accelerator": "nvidia-tesla-h100"
                        }
                    }
                }
            }
        }
        
        k8s_path = self.deployment_dir / "k8s-deployment.yaml"
        with open(k8s_path, 'w') as f:
            yaml.dump(k8s_config, f, default_flow_style=False)
        
        logger.info(f"Created Kubernetes config at {k8s_path}")
    
    def create_monitoring_config(self):
        """Create monitoring configuration."""
        monitoring_config = {
            "alerts": [
                {
                    "name": "high_cpu_usage",
                    "condition": "cpu_utilization > 80",
                    "duration": "5m",
                    "severity": "warning"
                },
                {
                    "name": "high_memory_usage",
                    "condition": "memory_utilization > 85",
                    "duration": "5m",
                    "severity": "warning"
                },
                {
                    "name": "gpu_utilization_low",
                    "condition": "gpu_utilization < 20",
                    "duration": "10m",
                    "severity": "info"
                },
                {
                    "name": "endpoint_errors",
                    "condition": "error_rate > 5",
                    "duration": "2m",
                    "severity": "critical"
                }
            ],
            "dashboards": [
                {
                    "name": "c2s-scale-gemma-cancer-dashboard",
                    "metrics": [
                        "request_count",
                        "response_time",
                        "error_rate",
                        "cpu_utilization",
                        "memory_utilization",
                        "gpu_utilization",
                        "model_inference_time"
                    ]
                }
            ],
            "logs": [
                {
                    "name": "cancer_research_logs",
                    "filter": "resource.type=\"aiplatform.googleapis.com/Endpoint\"",
                    "fields": [
                        "timestamp",
                        "severity",
                        "message",
                        "request_id",
                        "model_name",
                        "inference_time"
                    ]
                }
            ]
        }
        
        monitoring_path = self.deployment_dir / "monitoring_config.json"
        with open(monitoring_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        logger.info(f"Created monitoring config at {monitoring_path}")
    
    def create_requirements_txt(self):
        """Create requirements.txt for deployment."""
        requirements = """# Core dependencies
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
bitsandbytes>=0.41.0
peft>=0.4.0

# FastAPI and server
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Single-cell analysis
scanpy>=1.9.0
anndata>=0.9.0

# Graph processing
torch-geometric>=2.3.0
networkx>=3.1

# Cancer research
requests>=2.31.0
tqdm>=4.65.0

# Monitoring and logging
mlflow>=2.5.0
wandb>=0.15.0

# Google Cloud
google-cloud-aiplatform>=1.35.0
google-cloud-storage>=2.10.0
google-cloud-logging>=3.5.0

# HuggingFace
huggingface-hub>=0.16.0
datasets>=2.14.0
tokenizers>=0.13.0
sentencepiece>=0.1.99
tiktoken>=0.5.0
"""
        
        requirements_path = self.deployment_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        
        logger.info(f"Created requirements.txt at {requirements_path}")
    
    def create_all_configs(self):
        """Create all deployment configurations."""
        logger.info("Creating Vertex AI deployment configurations...")
        
        self.create_dockerfile()
        self.create_vertex_ai_config()
        self.create_deployment_script()
        self.create_kubernetes_config()
        self.create_monitoring_config()
        self.create_requirements_txt()
        
        logger.info("✅ All deployment configurations created!")

def main():
    """Main function to create deployment configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Vertex AI deployment configurations")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    
    args = parser.parse_args()
    
    # Create deployment configurations
    deployment = VertexAIDeployment(
        project_id=args.project_id,
        region=args.region
    )
    
    deployment.create_all_configs()
    
    print(f"🎉 Vertex AI deployment configurations created!")
    print(f"📁 Deployment files in: {deployment.deployment_dir}")
    print(f"🚀 Run deployment with: {deployment.deployment_dir}/deploy.sh")

if __name__ == "__main__":
    main()
