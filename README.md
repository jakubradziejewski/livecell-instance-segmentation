# LIVECell Instance Segmentation

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/W&B-Experiment_Tracking-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF6F00?style=for-the-badge&logo=gradio&logoColor=white)

**Instance segmentation of live cells using custom Mask R-CNN with ResNet-18 backbone**

[Dataset](https://sartorius-research.github.io/LIVECell/) • [Report Bug](https://github.com/jakubradziejewski/livecell-instance-segmentation/issues) 

</div>

---

## 🔬 About

This project implements instance segmentation for live cell microscopy images using a custom Mask R-CNN architecture. The model features:

- **Custom Mask R-CNN** with ResNet-18 backbone
- **CBAM attention mechanisms** for enhanced feature extraction
- **Feature Pyramid Network (FPN)** for multi-scale detection
- **Custom RPN** (Region Proposal Network) with anchor generation
- Training on the **LIVECell dataset** with 8 different cell types

---

## 🛠 Technologies

| Technology | Purpose |
|------------|---------|
| **PyTorch 2.7.1** | Deep learning framework |
| **CUDA 11.8** | GPU acceleration |
| **Docker & Docker Compose** | Containerization and environment management |
| **DVC (Data Version Control)** | Dataset versioning with Google Drive storage |
| **Weights & Biases** | Experiment tracking and metrics logging |
| **Gradio** | Interactive web interface for model inference |
| **OpenCV & PIL** | Image processing |
| **pycocotools** | COCO format annotation handling |
| **Matplotlib & Seaborn** | Visualization |

---

## 🚀 Quick Start

### 1. Setup Docker Environment

#### Install NVIDIA Container Toolkit

```bash
# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install the toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

#### Verify GPU Access

```bash
# Check nvidia runtime is available
docker info | grep -i runtime
# Expected output: Runtimes: io.containerd.runc.v2 nvidia runc

# Test GPU accessibility
docker compose run --rm training nvidia-smi

# Test PyTorch CUDA support
docker compose run --rm training python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Build and Start Container

```bash
# Clone the repository
git clone https://github.com/jakubradziejewski/livecell-instance-segmentation.git
cd livecell-instance-segmentation

# Build Docker image (first time: ~15-20 minutes)
docker compose build

# Start container in detached mode
docker compose up -d

# Attach to running container
docker compose exec training bash
```

**Note:** Use `docker-compose` (with hyphen) if you have the older standalone version.

---

### 2. Download Dataset

#### Download Directly from Source
```bash
# Inside the container
python scripts/download_data.py
```

This will automatically:
- Download annotation files for train/val/test splits
- Download and extract all microscopy images (~8GB)
- Organize images into proper directory structure

#### Optional: Setup Your Own DVC Storage

If you want to set up your own data versioning with DVC and Google Drive for tracking dataset changes, see **[DVC.md](DVC.md)** for detailed instructions on:
- Creating your own Google OAuth credentials
- Setting up your own Google Drive storage
- Configuring DVC for version control

---

### 3. Setup Experiment Tracking (Weights & Biases)

Create a `.env` file in the project root to store your Weights & Biases credentials:

```bash
# Inside the container or on your host machine
cat > .env << EOF
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=livecell-segmentation
WANDB_ENTITY=your_username_or_team
EOF
```

**Getting your W&B API key:**
1. Sign up at [wandb.ai](https://wandb.ai)
2. Go to [Settings → API keys](https://wandb.ai/settings)
3. Copy your API key
4. Paste it in the `.env` file

The training scripts will automatically use these credentials for logging.

---

### 4. Preprocess Dataset

The preprocessing step tiles large microscopy images into smaller patches for training.

```bash
# Inside the container
python preprocess_dataset.py \
    --source_dir data \
    --output_dir data_split \
    --num_images_per_split 100
```

**Parameters:**
- `--source_dir`: Directory containing the raw LIVECell dataset
- `--output_dir`: Where to save the preprocessed tiles
- `--num_images_per_split`: Total number of source images to process (automatically split 70/15/15 for train/val/test)

**Output:** Creates tiled images with corresponding COCO annotations in `data_split/`

---

### 5. Train Models

You can train either a custom Mask R-CNN or use transfer learning:

#### Option A: Train Custom Mask R-CNN

```bash
python train_custom.py
```

This trains the custom architecture from scratch with:
- ResNet-18 backbone
- CBAM attention modules
- Custom FPN, RPN, box head, and mask head

#### Option B: Transfer Learning (Pretrained Mask R-CNN)

```bash
python train_transfer.py
```

Fine-tunes a pretrained Mask R-CNN (ResNet-50 backbone) on the LIVECell dataset.

**Training Configuration:**
- Modify hyperparameters in the training scripts
- Models are saved to `models/` directory
- Training metrics are logged to Weights & Biases

---

### 6. Visualize Predictions

Generate visualizations of model predictions on test images:

```bash
python visualize.py \
    --model_path models/custom_model.pth \
    --model_type custom \
    --data_dir data_split \
    --output_dir visualizations \
    --num_images 10
```

**Parameters:**
- `--model_path`: Path to saved model checkpoint
- `--model_type`: Either `custom` or `transfer`
- `--data_dir`: Directory with preprocessed data
- `--output_dir`: Where to save visualization images
- `--num_images`: Number of test images to visualize

**Output:** Saves images with overlaid masks and bounding boxes to `visualizations/`

---

### 7. Interactive Demo (Gradio)

Launch a web-based interface for real-time inference:

```bash
python app_gradio.py
```

**Access the interface:**
- Open your browser and navigate to: `http://localhost:7860`
- Upload a microscopy image
- Adjust the confidence threshold
- Click "Run Detection" to see results

**Features:**
- Real-time instance segmentation
- Adjustable confidence threshold
- Visual overlay of detected cell masks
- Detection count and scores

---

### 8. Explain Predictions

Generate interpretability visualizations using GradCAM:

```bash
python explain_predictions.py \
    --model_path models/custom_model.pth \
    --model_type custom \
    --data_dir data_split \
    --output_dir explanations \
    --num_images 5
```

**Output:** Creates heatmaps showing which regions of the image the model focuses on for making predictions.

---

## 📁 Project Structure

```
livecell-instance-segmentation/
├── data/                          # Raw LIVECell dataset
│   ├── annotations/               # COCO format annotations
│   └── images/                    # Raw microscopy images
│
├── data_split/                    # Preprocessed tiled dataset
│   ├── train/
│   ├── val/
│   ├── test/
│   └── annotations/
│
├── models/                        # Saved model checkpoints
│   ├── custom_model.pth
│   └── transfer_model.pth
│
├── src/                           # Source code
│   ├── components/                # Model components
│   │   ├── cbam.py               # CBAM attention module
│   │   ├── fpn.py                # Feature Pyramid Network
│   │   ├── rpn.py                # Region Proposal Network
│   │   ├── anchor_generator.py   # Anchor generation
│   │   ├── box_head.py           # Bounding box prediction head
│   │   └── mask_head.py          # Segmentation mask head
│   └── utils/                     # Utility functions
│
├── visualizations/                # Output visualizations
├── explanations/                  # GradCAM explanations
│
├── train_custom.py                # Train custom Mask R-CNN
├── train_transfer.py              # Transfer learning script
├── preprocess_dataset.py          # Dataset preprocessing
├── visualize.py                   # Generate prediction visualizations
├── explain_predictions.py         # Generate GradCAM explanations
├── app_gradio.py                  # Gradio web interface
├── dataset.py                     # PyTorch dataset class
├── custom_maskrcnn.py             # Custom model architecture
│
├── Dockerfile                     # Docker image definition
├── docker-compose.yml             # Docker services configuration
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (create this)
├── .dvc/                          # DVC configuration
├── DVC.md                         # DVC setup guide
└── README.md                      # This file
```

---

## 📊 Dataset

**LIVECell Dataset**: A large-scale dataset for label-free live cell segmentation

- **Size**: 5,239 high-resolution microscopy images
- **Cell Types**: 8 different cell morphologies
- **Annotations**: Over 1.6M individual cell instances
- **Format**: COCO instance segmentation format

**Download**: [LIVECell Official Website](https://sartorius-research.github.io/LIVECell/)

**Citation:**
```
@article{edlund2021livecell,
  title={LIVECell—A large-scale dataset for label-free live cell segmentation},
  author={Edlund, Christopher and Jackson, Timothy R and Khalid, Nabeel and 
          Bevan, Nicola and Dale, Timothy and Dengel, Andreas and Ahmed, Sheraz and 
          Trygg, Johan and Sj{\"o}gren, Rickard},
  journal={Nature Methods},
  year={2021}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
