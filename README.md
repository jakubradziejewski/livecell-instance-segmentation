# LIVECell Instance Segmentation

Instance segmentation of live cells using Mask R-CNN.

## Dataset

LIVECell: https://sartorius-research.github.io/LIVECell/

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with drivers  
- NVIDIA Container Toolkit

Note: This guide uses `docker compose` (V2). If you have the older 
standalone version, use `docker-compose` (with hyphen) instead.

### Installing NVIDIA Container Toolkit
```bash
# 1. Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Update package list
sudo apt-get update

# 3. Install the toolkit
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 5. Restart Docker
sudo systemctl restart docker

# 6. Verify nvidia runtime is available
docker info | grep -i runtime
# After step 6, you should see: Runtimes: io.containerd.runc.v2 nvidia runc

# Check GPU is available
docker compose run --rm training nvidia-smi

# Test PyTorch + CUDA
docker compose run --rm training python -c "import torch; print(torch.cuda.is_available())"
```


### Setup & Run
```bash
# 1. Clone repository
git clone https://github.com/jakubradziejewski/livecell-instance-segmentation.git
cd livecell-instance-segmentation

# 2. Build Docker image (first time: ~15-20 minutes)
docker compose build

# 3. Download dataset
# Download from: https://sartorius-research.github.io/LIVECell/
# Place in: data/raw/

# Start a container
docker compose up -d

# Attach to running container
docker compose exec training bash

# 4. Run training
docker compose run --rm training python train.py
```

### Development
```bash
# Start interactive session
docker compose run --rm training bash

# Inside container, you can:
python scripts/preprocess.py
python train.py
python evaluate.py
```

### Useful Commands
```bash
docker compose build                    # Build image
docker compose build --no-cache        # Rebuild from scratch (if issues)


# Adding Libraries
# 1. Edit requirements.txt
# 2. docker-compose build (only ~30 sec!)

```

## Project Structure
```
livecell-instance-segmentation/
├── data/              # Dataset
├── models/            # Saved models
├── src/               # Source code
├── scripts/           # Utility scripts
├── configs/           # Configuration files
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker services
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.10
- PyTorch 2.7.1
- CUDA 11.8
- See `requirements.txt` for full list

