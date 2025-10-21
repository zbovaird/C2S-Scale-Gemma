FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv for Python package management and update PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv --version

# Set PATH environment variable for subsequent commands
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN uv sync --frozen --verbose

# Explicitly install PyTorch with CUDA support using pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --verbose --no-cache-dir

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Create directories
RUN mkdir -p /app/data/cancer /app/data/processed/graphs /app/models /app/logs

# Set up HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "scripts/vertex_ai_server.py"]
