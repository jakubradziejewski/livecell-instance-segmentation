"""
Simple Mask R-CNN Training and Testing Script
Uses ResNet-50 backbone with pretrained weights from COCO
Train on train set, validate on val set, test on test set
"""

import os
import sys
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou
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
        metrics: Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_mask = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Track individual losses
        total_loss += losses.item()
        loss_classifier += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
        loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_mask += loss_dict.get('loss_mask', torch.tensor(0.0)).item()
        loss_objectness += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
        loss_rpn_box_reg += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': losses.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
        
        # Free memory
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    
    n = len(dataloader)
    metrics = {
        'loss': total_loss / n,
        'loss_classifier': loss_classifier / n,
        'loss_box_reg': loss_box_reg / n,
        'loss_mask': loss_mask / n,
        'loss_objectness': loss_objectness / n,
        'loss_rpn_box_reg': loss_rpn_box_reg / n,
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, iou_threshold=0.5):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Mask R-CNN model
        dataloader: Dataloader
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


def visualize_predictions(results, save_dir='outputs', dataset_name='test'):
    """
    Visualize predictions and save to disk.
    
    Args:
        results: List of prediction results
        save_dir: Directory to save visualizations
        dataset_name: Name of dataset (for filenames)
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
        
        # Calculate IoU for matched predictions
        if len(target['boxes']) > 0 and len(pred_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, target['boxes'])
            max_ious, _ = iou_matrix.max(dim=1)
        else:
            max_ious = torch.zeros(len(pred_boxes))
        
        # Draw prediction boxes
        for box, score, iou in zip(pred_boxes, pred_scores, max_ious):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[2].add_patch(rect)
            axes[2].text(
                x1, y1 - 5, f'{score:.2f} (IoU:{iou:.2f})',
                color='red', fontsize=8, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        axes[2].axis('off')
        
        plt.tight_layout()
        filename = f'{dataset_name}_prediction_{idx+1}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
        print(f"  Ground truth instances: {len(target['boxes'])}")
        print(f"  Predicted instances (score > 0.5): {num_pred}")
        if num_pred > 0:
            print(f"  Average confidence: {pred_scores.mean():.3f}")
            print(f"  Average IoU: {max_ious.mean():.3f}")
        print()


def main():
    """
    Main training and testing script.
    """
    print("=" * 80)
    print("Mask R-CNN Instance Segmentation")
    print("Train on TRAIN, validate on VAL, test on TEST")
    print("=" * 80)
    
    # Configuration
    data_dir = 'data_split'
    batch_size = 2  # Small batch size for 4GB GPU
    num_workers = 2
    lr = 0.001
    num_epochs = 1
    num_classes = 2  # Background + cell (instance segmentation with 1 class!)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Num classes: {num_classes} (background + cell)")
    print(f"  Task: Instance Segmentation (distinguishing individual cells)")
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
    test_loader = dataloaders['test']
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} images, {len(test_loader)} batches")
    print()
    
    # Create model
    print("Creating model...")
    print("-" * 80)
    model = get_model_maskrcnn(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created (Transfer Learning from COCO)")
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
    print("Starting training on TRAIN set...")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"  Total Loss:      {train_metrics['loss']:.4f}")
        print(f"  Classifier Loss: {train_metrics['loss_classifier']:.4f}")
        print(f"  Box Reg Loss:    {train_metrics['loss_box_reg']:.4f}")
        print(f"  Mask Loss:       {train_metrics['loss_mask']:.4f}")
        print(f"  Objectness Loss: {train_metrics['loss_objectness']:.4f}")
        print(f"  RPN Box Loss:    {train_metrics['loss_rpn_box_reg']:.4f}")
        print()
    
    print("=" * 80)
    print("Training completed!")
    print()
    
    # Validation
    print("=" * 80)
    print("Evaluating on VALIDATION set...")
    print("-" * 80)
    
    val_metrics = evaluate(model, val_loader, device, iou_threshold=0.5)
    
    print(f"\nValidation Metrics (IoU threshold: 0.5):")
    print(f"  Mean IoU:        {val_metrics['mean_iou']:.4f}")
    print(f"  Mean Precision:  {val_metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:     {val_metrics['mean_recall']:.4f}")
    print(f"  F1 Score:        {val_metrics['f1_score']:.4f}")
    print(f"  GT Instances:    {val_metrics['total_gt_instances']}")
    print(f"  Pred Instances:  {val_metrics['total_pred_instances']}")
    print(f"  True Positives:  {val_metrics['total_true_positives']}")
    print()
    
    # Testing
    print("=" * 80)
    print("Evaluating on TEST set...")
    print("-" * 80)
    
    test_metrics = evaluate(model, test_loader, device, iou_threshold=0.5)
    
    print(f"\nTest Metrics (IoU threshold: 0.5):")
    print(f"  Mean IoU:        {test_metrics['mean_iou']:.4f}")
    print(f"  Mean Precision:  {test_metrics['mean_precision']:.4f}")
    print(f"  Mean Recall:     {test_metrics['mean_recall']:.4f}")
    print(f"  F1 Score:        {test_metrics['f1_score']:.4f}")
    print(f"  GT Instances:    {test_metrics['total_gt_instances']}")
    print(f"  Pred Instances:  {test_metrics['total_pred_instances']}")
    print(f"  True Positives:  {test_metrics['total_true_positives']}")
    print()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/maskrcnn_resnet50_1epoch.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    print()
    
    # Predictions on TEST set
    print("=" * 80)
    print("Generating predictions on TEST set...")
    print("-" * 80)
    
    test_results = predict(model, test_loader, device, num_samples=5)
    
    print(f"✓ Generated {len(test_results)} predictions")
    print()
    
    # Visualize
    print("Visualizing test predictions...")
    print("-" * 80)
    visualize_predictions(test_results, save_dir='outputs', dataset_name='test')
    
    print("=" * 80)
    print("✓ All done!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Model:              models/maskrcnn_resnet50_1epoch.pth")
    print(f"  Visualizations:     outputs/test_prediction_*.png")
    print(f"  Val Mean IoU:       {val_metrics['mean_iou']:.4f}")
    print(f"  Test Mean IoU:      {test_metrics['mean_iou']:.4f}")
    print(f"  Test F1 Score:      {test_metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()