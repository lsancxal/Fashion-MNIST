# Fashion-MNIST CNN Training (GPU-enabled)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install CUDA-enabled PyTorch first, then the rest
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir matplotlib>=3.7.0 "Pillow>=9.0.0" scikit-learn>=1.3.0 seaborn>=0.12.0

# Copy application
COPY src/ ./src/
COPY main.py .

# Use non-interactive matplotlib backend (no display in container)
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

# Run training
CMD ["python3", "-u", "main.py"]
