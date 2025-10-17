#!/usr/bin/env python3
"""
Quick start script for C2S-Scale-Gemma Colab prototype.

This script helps you get started quickly in Google Colab with A100 GPU.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies for Colab."""
    packages = [
        "uhg",
        "torch",
        "transformers>=4.43.0",
        "accelerate>=1.1.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.11.0",
        "datasets>=2.20.0",
        "scikit-learn>=1.5.0",
        "scanpy>=1.9.0",
        "anndata>=0.10.0",
        "umap-learn>=0.5.0",
        "pynndescent>=0.5.0",
        "mlflow>=2.14.0",
        "omegaconf>=2.3.0",
        "networkx>=3.2.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "wandb>=0.17.0",
        "python-dotenv>=1.0.0",
        "flash-attn",
        "xformers"
    ]
    
    print("🚀 Installing dependencies for A100 GPU...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\n🎯 Dependencies installed successfully!")

def check_gpu():
    """Check GPU availability and optimize for A100."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"🚀 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            if "A100" in gpu_name:
                print("✅ A100 GPU detected! Enabling optimizations...")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_per_process_memory_fraction(0.9)
                print("✅ A100 optimizations enabled")
            else:
                print("⚠️  Non-A100 GPU detected. Consider using A100 for best performance.")
        else:
            print("❌ No GPU detected. Please enable GPU in Colab runtime.")
    except ImportError:
        print("❌ PyTorch not installed. Please run install_dependencies() first.")

def download_notebook():
    """Download the Colab notebook."""
    notebook_url = "https://raw.githubusercontent.com/zbovaird/C2S-Scale-Gemma/main/notebooks/colab_prototype.ipynb"
    
    print("📥 Downloading Colab notebook...")
    try:
        import urllib.request
        urllib.request.urlretrieve(notebook_url, "colab_prototype.ipynb")
        print("✅ Notebook downloaded successfully!")
        print("📝 Open colab_prototype.ipynb in Colab to get started.")
    except Exception as e:
        print(f"❌ Failed to download notebook: {e}")
        print("📝 Please manually download the notebook from the GitHub repository.")

def main():
    """Main function to set up Colab environment."""
    print("🎯 C2S-Scale-Gemma Colab Setup")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    check_gpu()
    
    # Download notebook
    download_notebook()
    
    print("\n🚀 Setup complete!")
    print("Next steps:")
    print("1. Open colab_prototype.ipynb in Colab")
    print("2. Select A100 GPU runtime")
    print("3. Run all cells to train the hybrid model")
    print("4. Monitor performance with built-in metrics")

if __name__ == "__main__":
    main()
