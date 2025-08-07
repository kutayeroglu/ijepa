# Local Usage:
# ```
# docker build -t ghcr.io/bouncmpe/cuda-python3 containers/cuda-python3/
# docker run -it --rm --gpus=all ghcr.io/bouncmpe/cuda-python3
# ```

# Base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="Kutay Eroglu"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    wget \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip

# NEW: Install PyTorch, TorchVision, and TorchAudio for CUDA 12.1
# This is the most reliable method for getting GPU-enabled versions.
RUN python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt
