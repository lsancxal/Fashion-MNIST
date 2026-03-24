# Fashion-MNIST

This project classifies the Fashion-MNIST dataset using a convolutional neural network.

## Run With Docker

### Prerequisites

- Docker installed and running
- NVIDIA GPU + drivers (for GPU training)
- NVIDIA Container Toolkit / Docker GPU support enabled

### 1) Build the image

From the project root:

```bash
docker compose build fashion-mnist
```

This builds the `fashion-mnist:latest` image from the included `Dockerfile`.

### 2) Run training (GPU)

If your Compose version supports GPU flags on `run`, use:

```bash
docker compose run --rm --gpus all fashion-mnist
```

If your Compose version does not support `--gpus` on `run`, use the equivalent `docker run` command:

```bash
docker run --rm --gpus all \
  -e MPLBACKEND=Agg \
  -e PYTHONUNBUFFERED=1 \
  -v fashion-mnist-data:/app/.fashion \
  -v "$(pwd)/output:/app/output" \
  fashion-mnist:latest
```

### 3) Verify CUDA is available in the container

```bash
docker run --rm --gpus all fashion-mnist:latest \
  python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output should include `cuda: True` when GPU is correctly configured.

### Notes

- Training outputs are written to `./output`.
- Dataset cache is stored in Docker volume `fashion-mnist-data`.
- If GPU is not detected, first validate host setup with:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```
