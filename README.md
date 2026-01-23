# LIVECell Instance Segmentation

Instance segmentation of live cells using Mask R-CNN.

## Dataset

LIVECell: https://sartorius-research.github.io/LIVECell/

## Quick Start

### Prerequisites

- Docker Desktop installed and running
- NVIDIA GPU with drivers

### Setup & Run
```bash
# 1. Clone repository
git clone https://github.com/yourusername/livecell-instance-segmentation.git
cd livecell-instance-segmentation

# 2. Build Docker image (first time: ~15-20 minutes)
docker-compose build

# 3. Download dataset
# Download from: https://sartorius-research.github.io/LIVECell/
# Place in: data/raw/

# Start a container
docker-compose up -d

# Attach to running container
docker-compose exec training bash

# 4. Run training
docker-compose run --rm training python train.py
```

### Development
```bash
# Start interactive session
docker-compose run --rm training bash

# Inside container, you can:
python scripts/preprocess.py
python train.py
python evaluate.py
```

### Useful Commands
```bash
# Check GPU is available
docker-compose run --rm training nvidia-smi

# Test PyTorch + CUDA
docker-compose run --rm training python -c "import torch; print(torch.cuda.is_available())"

docker-compose build                    # Build image
docker-compose build --no-cache        # Rebuild from scratch (if issues)


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

