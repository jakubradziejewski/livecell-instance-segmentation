"""
Simple Mask R-CNN Training and Testing Script
Uses ResNet-18 backbone with pretrained weights from COCO
"""

import os
import sys
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Import our dataset
sys.path.append('src')
from dataset import get_dataloaders


def get_model_maskrcnn(num_classes, pretrained=True):
    """
    Load Mask R-CNN model with ResNet-50-FPN backbone.
    (ResNet-18 is not available in torchvision's Mask R-CNN, so using ResNet-50)
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use COCO pretrained weights
        
    Returns:
        model: Mask R-CNN model
    """
    # Load pretrained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: Mask R-CNN model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track loss
        total_loss += losses.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': losses.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
        
        # Free memory
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def predict(model, dataloader, device, num_samples=5):
    """
    Make predictions on samples from dataloader.
    
    Args:
        model: Mask R-CNN model
        dataloader: Dataloader to get samples from
        device: Device to run inference on
        num_samples: Number of samples to predict
        
    Returns:
        results: List of (image, prediction, target) tuples
    """
    model.eval()
    results = []
    
    for images, targets in dataloader:
        if len(results) >= num_samples:
            break
        
        # Move to device
        images_device = [img.to(device) for img in images]
        
        # Predict
        predictions = model(images_device)
        
        # Collect results
        for img, pred, target in zip(images, predictions, targets):
            if len(results) >= num_samples:
                break
            
            results.append({
                'image': img.cpu(),
                'prediction': {k: v.cpu() for k, v in pred.items()},
                'target': {k: v.cpu() for k, v in target.items()}
            })
        
        # Free memory
        del images_device, predictions
        torch.cuda.empty_cache()
    
    return results


def visualize_predictions(results, save_dir='outputs'):
    """
    Visualize predictions and save to disk.
    
    Args:
        results: List of prediction results
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, result in enumerate(results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        img = result['image'].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Ground truth
        target = result['target']
        pred = result['prediction']
        
        # Plot 1: Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot 2: Ground truth
        axes[1].imshow(img)
        axes[1].set_title(f'Ground Truth ({len(target["boxes"])} instances)')
        
        # Draw ground truth boxes
        for box in target['boxes']:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            axes[1].add_patch(rect)
        axes[1].axis('off')
        
        # Plot 3: Predictions (filter by confidence > 0.5)
        axes[2].imshow(img)
        
        # Filter predictions by score
        scores = pred['scores']
        keep = scores > 0.5
        
        num_pred = keep.sum().item()
        axes[2].set_title(f'Predictions ({num_pred} instances, score > 0.5)')
        
        pred_boxes = pred['boxes'][keep]
        pred_scores = pred['scores'][keep]
        
        # Draw prediction boxes
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
            axes[2].text(
                x1, y1 - 5, f'{score:.2f}',
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved prediction {idx+1} to {save_dir}/prediction_{idx+1}.png")
        print(f"  Ground truth instances: {len(target['boxes'])}")
        print(f"  Predicted instances (score > 0.5): {num_pred}")
        if num_pred > 0:
            print(f"  Average confidence: {pred_scores.mean():.3f}")
        print()


def main():
    """
    Main training and testing script.
    """
    print("=" * 80)
    print("Simple Mask R-CNN Training & Testing")
    print("=" * 80)
    
    # Configuration
    data_dir = 'data_split'
    batch_size = 2  # Small batch size for 4GB GPU
    num_workers = 2
    lr = 0.001
    num_epochs = 1
    num_classes = 2  # Background + cell
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Num classes: {num_classes}")
    print()
    
    # Load data
    print("Loading datasets...")
    print("-" * 80)
    dataloaders = get_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print()
    
    # Create model
    print("Creating model...")
    print("-" * 80)
    model = get_model_maskrcnn(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print()
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Training
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
        print()
    
    print("=" * 80)
    print("Training completed!")
    print()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/maskrcnn_simple_1epoch.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    print()
    
    # Testing / Predictions
    print("=" * 80)
    print("Making predictions on validation set...")
    print("-" * 80)
    
    results = predict(model, val_loader, device, num_samples=5)
    
    print(f"✓ Generated {len(results)} predictions")
    print()
    
    # Visualize
    print("Visualizing predictions...")
    print("-" * 80)
    visualize_predictions(results, save_dir='outputs')
    
    print("=" * 80)
    print("✓ All done!")
    print("=" * 80)
    print(f"Check 'outputs/' directory for visualization results")


if __name__ == "__main__":
    main()