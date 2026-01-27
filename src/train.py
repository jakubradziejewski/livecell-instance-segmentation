"""
Training Script for Custom Mask R-CNN Architecture
With proper evaluation metrics matching simple_training.py
With command-line model selection
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou
import argparse

# Import custom model and dataset
sys.path.append('src')
from custom_maskrcnn import get_custom_model
from transfer_learning_model import get_model_maskrcnn
from dataset import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Train custom model for one epoch.
    
    Args:
        model: Custom Mask R-CNN model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        metrics: Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0
    loss_rpn_cls = 0
    loss_rpn_reg = 0
    loss_box_cls = 0
    loss_box_reg = 0
    loss_mask = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Sum losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track losses
        total_loss += losses.item()
        loss_rpn_cls += loss_dict.get('loss_rpn_cls', torch.tensor(0.0)).item()
        loss_rpn_reg += loss_dict.get('loss_rpn_reg', torch.tensor(0.0)).item()
        loss_box_cls += loss_dict.get('loss_box_cls', torch.tensor(0.0)).item()
        loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_mask += loss_dict.get('loss_mask', torch.tensor(0.0)).item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': losses.item(),
            'avg': total_loss / (batch_idx + 1)
        })
        
        # Free memory
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    
    n = len(dataloader)
    metrics = {
        'total_loss': total_loss / n,
        'loss_rpn_cls': loss_rpn_cls / n,
        'loss_rpn_reg': loss_rpn_reg / n,
        'loss_box_cls': loss_box_cls / n,
        'loss_box_reg': loss_box_reg / n,
        'loss_mask': loss_mask / n,
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, iou_threshold=0.5):
    """
    Evaluate model with IoU, precision, recall metrics (like simple_training.py).
    
    Args:
        model: Custom Mask R-CNN model
        dataloader: Validation dataloader
        device: Device
        iou_threshold: IoU threshold for matching predictions to ground truth
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    all_ious = []
    all_precisions = []
    all_recalls = []
    total_gt_instances = 0
    total_pred_instances = 0
    total_true_positives = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for images, targets in progress_bar:
        # Move to device
        images_device = [img.to(device) for img in images]
        
        # Predict
        predictions = model(images_device)
        
        # Calculate metrics for each image
        for pred, target in zip(predictions, targets):
            gt_boxes = target['boxes']
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            
            # Filter predictions by score threshold
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            
            total_gt_instances += len(gt_boxes)
            total_pred_instances += len(pred_boxes)
            
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            # Calculate IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            
            # For each prediction, find best matching ground truth
            if len(iou_matrix) > 0:
                max_ious, max_indices = iou_matrix.max(dim=1)
                all_ious.extend(max_ious.tolist())
                
                # Count true positives (IoU > threshold)
                true_positives = (max_ious > iou_threshold).sum().item()
                total_true_positives += true_positives
                
                # Precision and recall for this image
                precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
                
                all_precisions.append(precision)
                all_recalls.append(recall)
        
        # Free memory
        del images_device, predictions
        torch.cuda.empty_cache()
    
    # Calculate overall metrics
    metrics = {
        'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'total_gt_instances': total_gt_instances,
        'total_pred_instances': total_pred_instances,
        'total_true_positives': total_true_positives,
    }
    
    # Calculate F1 score
    if metrics['mean_precision'] + metrics['mean_recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['mean_precision'] * metrics['mean_recall']) / \
                              (metrics['mean_precision'] + metrics['mean_recall'])
    else:
        metrics['f1_score'] = 0.0
    
    return metrics


def save_training_plot(train_losses, val_metrics_history, save_path='outputs/custom_training_plot.png'):
    """
    Plot training curves with multiple metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: IoU
    if val_metrics_history:
        val_ious = [m['mean_iou'] for m in val_metrics_history]
        axes[0, 1].plot(epochs, val_ious, 'g-', label='Mean IoU', linewidth=2, marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation Mean IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    if val_metrics_history:
        val_precisions = [m['mean_precision'] for m in val_metrics_history]
        val_recalls = [m['mean_recall'] for m in val_metrics_history]
        axes[1, 0].plot(epochs, val_precisions, 'r-', label='Precision', linewidth=2, marker='^')
        axes[1, 0].plot(epochs, val_recalls, 'orange', label='Recall', linewidth=2, marker='v')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: F1 Score
    if val_metrics_history:
        val_f1s = [m['f1_score'] for m in val_metrics_history]
        axes[1, 1].plot(epochs, val_f1s, 'purple', label='F1 Score', linewidth=2, marker='D')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training plot saved to {save_path}")


def main():
    """
    Main training script for custom architecture.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN model (custom or transfer learning)')
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'maskrcnn'],
                        help='Model type to train: "custom" for custom architecture or "maskrcnn" for transfer learning')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    args = parser.parse_args()
    
    print(f"Training with {args.model.upper()} model")
    print("=" * 80)
    if args.model == 'custom':
        print("Architecture: ResNet-18 Backbone + Custom FPN/RPN/Heads (>50% own layers)")
    else:
        print("Architecture: Pretrained Mask R-CNN with Transfer Learning")
    
    # Configuration
    data_dir = 'data_split'
    batch_size = args.batch_size
    num_workers = 2
    lr = args.lr
    num_epochs = args.num_epochs
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Model type: {args.model}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Num classes: {num_classes}")
    
    # Load data
    print("\nLoading datasets...")
    print("-" * 80)
    dataloaders = get_dataloaders(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} images, {len(test_loader)} batches")
    
    # Create model based on argument
    print(f"\nCreating {args.model} model...")
    print("-" * 80)
    
    if args.model == 'custom':
        model = get_custom_model(num_classes=num_classes)
        model.to(device)
        
        # Count parameters for custom model
        param_info = model.count_parameters()
        
        print(f"Custom model created")
        print(f"  Total parameters:      {param_info['total']:,}")
        print(f"  Backbone (borrowed):   {param_info['backbone']:,} ({100-param_info['custom_percentage']:.1f}%)")
        print(f"  Custom layers (yours): {param_info['custom']:,} ({param_info['custom_percentage']:.1f}%)")
        print(f"  Custom >50%: {param_info['custom_percentage'] > 50}")
        print(f"  Model size: ~{param_info['total'] * 4 / (1024**2):.1f} MB")
    else:
        model = get_model_maskrcnn(num_classes=num_classes)
        model.to(device)
        
        # Count total parameters for maskrcnn model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Mask R-CNN model created")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB")
        param_info = None  # No custom param info for maskrcnn
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"  Total Loss:    {train_metrics['total_loss']:.4f}")
        print(f"  RPN Cls Loss:  {train_metrics['loss_rpn_cls']:.4f}")
        print(f"  RPN Reg Loss:  {train_metrics['loss_rpn_reg']:.4f}")
        print(f"  Box Cls Loss:  {train_metrics['loss_box_cls']:.4f}")
        print(f"  Box Reg Loss:  {train_metrics['loss_box_reg']:.4f}")
        print(f"  Mask Loss:     {train_metrics['loss_mask']:.4f}")
        
        train_losses.append(train_metrics['total_loss'])
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, iou_threshold=0.5)
        val_metrics_history.append(val_metrics)
        
        print(f"Validation Metrics:")
        print(f"  Mean IoU:        {val_metrics['mean_iou']:.4f}")
        print(f"  Mean Precision:  {val_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:     {val_metrics['mean_recall']:.4f}")
        print(f"  F1 Score:        {val_metrics['f1_score']:.4f}")
        print(f"  GT Instances:    {val_metrics['total_gt_instances']}")
        print(f"  Pred Instances:  {val_metrics['total_pred_instances']}")
        print(f"  True Positives:  {val_metrics['total_true_positives']}")
        
        # Step scheduler
        scheduler.step()
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nTraining completed!")
    print("=" * 80)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{args.model}_maskrcnn_{num_epochs}epochs.pth'
    
    save_dict = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'model_type': args.model,
    }
    
    if param_info is not None:
        save_dict['param_info'] = param_info
    
    torch.save(save_dict, model_path)
    
    print(f"✓ Model saved to {model_path}")
    print()
    
    # Save training plot
    print("Creating training plots...")
    plot_path = f'outputs/{args.model}_training_plot.png'
    save_training_plot(train_losses, val_metrics_history, save_path=plot_path)
    
    # Test on test set
    print("\nEvaluating on TEST set...")
    print("-" * 80)
    
    test_metrics = evaluate(model, test_loader, device, iou_threshold=0.5)
    
    print(f"\nTest Metrics:")
    print(f"  Mean IoU:        {test_metrics['mean_iou']:.4f}")
    print(f"  Mean Precision:  {test_metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:     {test_metrics['mean_recall']:.4f}")
    print(f"  F1 Score:        {test_metrics['f1_score']:.4f}")
    print(f"  GT Instances:    {test_metrics['total_gt_instances']}")
    print(f"  Pred Instances:  {test_metrics['total_pred_instances']}")
    print(f"  True Positives:  {test_metrics['total_true_positives']}")
    
    print("\n" + "=" * 80)
    print(f"{args.model.upper()} Model Training Complete!")
    print(f"\nFiles saved:")
    print(f"  Model: {model_path}")
    print(f"  Training plot: {plot_path}")


if __name__ == "__main__":
    main()