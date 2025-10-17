# C2S-Scale-Gemma A100 Colab Setup
# Run this cell first in Colab with A100 GPU

# Install dependencies
!pip install uhg torch transformers accelerate peft datasets scikit-learn scanpy anndata umap-learn pynndescent mlflow omegaconf networkx pandas numpy tqdm pyyaml wandb python-dotenv bitsandbytes flash-attn xformers

# Clone repository
!git clone https://github.com/zbovaird/C2S-Scale-Gemma.git
%cd C2S-Scale-Gemma

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# A100 optimizations
if "A100" in torch.cuda.get_device_name(0):
    print("🚀 A100 GPU detected! Enabling optimizations...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.9)
    print("✅ A100 optimizations enabled")

print("🎯 Setup complete! Now run the notebook cells.")
