# Local Usage:
# ```
# docker build -t ghcr.io/bouncmpe/cuda-python3 containers/cuda-python3/
# docker run -it --rm --gpus=all ghcr.io/bouncmpe/cuda-python3
# ```

# Base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

LABEL maintainer="Kutay Eroglu"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y install --no-install-recommends \
    git \
    wget \
    cmake \
    ninja-build \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 && \
    \
    # Install Python packages
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    \
    # Clean up APT caches to reduce image size
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt
