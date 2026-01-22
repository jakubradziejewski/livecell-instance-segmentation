FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies (cached unless this layer changes)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Setup python alternatives (cached)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip (cached)
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# KEY OPTIMIZATION: Copy requirements.txt FIRST (before other code)
# This layer is cached unless requirements.txt changes
COPY requirements.txt .

# Install Python packages (cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (this changes often but doesn't trigger pip reinstall!)
COPY . .

# Create directories
RUN mkdir -p /workspace/data /workspace/models /workspace/experiments

EXPOSE 8888 6006

CMD ["bash"]