"""
Universal Prediction Visualizer
Works with any trained model (.pth file) - both transfer learning and custom architecture
Visualizes predictions on test set and creates a tile montage
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou
from tqdm import tqdm

# Import dataset
sys.path.append('src')
from dataset import get_dataloader


def load_model(model_path, model_type='transfer', num_classes=2, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to .pth file
        model_type: 'transfer' or 'custom'
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
    """
    print(f"Loading {model_type} model from {model_path}...")
    
    if model_type == 'transfer':
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        # Create model architecture
        model = maskrcnn_resnet50_fpn(pretrained=False)
        
        # Replace heads
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        
    elif model_type == 'custom':
        from custom_maskrcnn import get_custom_model
        
        # Create model
        model = get_custom_model(num_classes=num_classes, pretrained_backbone=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


@torch.no_grad()
def predict_samples(model, dataloader, device, num_samples=5):
    """
    Make predictions on samples from dataloader.
    
    Args:
        model: Trained model
        dataloader: Dataloader to get samples from
        device: Device to run inference on
        num_samples: Number of samples to predict
        
    Returns:
        results: List of (image, prediction, target) dictionaries
    """
    model.eval()
    results = []
    
    print(f"\nGenerating predictions on {num_samples} samples...")
    
    for images, targets in tqdm(dataloader, desc="Predicting"):
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


def visualize_predictions(results, save_dir='outputs', prefix='test', score_threshold=0.5):
    """
    Visualize predictions and save to disk.
    
    Args:
        results: List of prediction results
        save_dir: Directory to save visualizations
        prefix: Prefix for filenames
        score_threshold: Minimum confidence score to display
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nVisualizing predictions...")
    
    for idx, result in enumerate(tqdm(results, desc="Creating visualizations")):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        img = result['image'].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Ground truth and predictions
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
        
        # Plot 3: Predictions
        axes[2].imshow(img)
        
        # Filter predictions by score
        scores = pred['scores']
        keep = scores > score_threshold
        
        num_pred = keep.sum().item()
        axes[2].set_title(f'Predictions ({num_pred} instances, score > {score_threshold})')
        
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
        filename = f'{prefix}_prediction_{idx+1}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(results)} visualizations to {save_dir}/")


def create_tile_montage(dataloader, save_dir='outputs', num_tiles=20, filename='tile_montage.png'):
    """
    Create a montage showing multiple tiles from the test dataset.
    Useful for understanding the tiling strategy used in preprocessing.
    
    Args:
        dataloader: Test dataloader
        save_dir: Directory to save montage
        num_tiles: Number of tiles to include
        filename: Output filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nCreating tile montage with {num_tiles} tiles...")
    
    # Collect tiles
    tiles = []
    targets_list = []
    
    for images, targets in dataloader:
        for img, target in zip(images, targets):
            if len(tiles) >= num_tiles:
                break
            tiles.append(img)
            targets_list.append(target)
        if len(tiles) >= num_tiles:
            break
    
    # Create montage
    n_cols = 5
    n_rows = int(np.ceil(num_tiles / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, (tile, target) in enumerate(zip(tiles, targets_list)):
        if idx >= len(axes):
            break
        
        # Convert to numpy
        img = tile.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[idx].imshow(img)
        
        # Draw ground truth boxes
        num_instances = len(target['boxes'])
        for box in target['boxes']:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='green', facecolor='none', alpha=0.7
            )
            axes[idx].add_patch(rect)
        
        axes[idx].set_title(f'Tile {idx+1}: {num_instances} cells', fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(tiles), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Tile montage saved to {save_path}")


def print_statistics(results):
    """
    Print statistics about predictions.
    
    Args:
        results: List of prediction results
    """
    print("\n" + "=" * 80)
    print("PREDICTION STATISTICS")
    print("=" * 80)
    
    total_gt = 0
    total_pred = 0
    total_ious = []
    
    for result in results:
        target = result['target']
        pred = result['prediction']
        
        # Filter by score
        keep = pred['scores'] > 0.5
        pred_boxes = pred['boxes'][keep]
        
        total_gt += len(target['boxes'])
        total_pred += len(pred_boxes)
        
        # Calculate IoU
        if len(target['boxes']) > 0 and len(pred_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, target['boxes'])
            max_ious, _ = iou_matrix.max(dim=1)
            total_ious.extend(max_ious.tolist())
    
    print(f"\nAcross {len(results)} test images:")
    print(f"  Total GT instances:     {total_gt}")
    print(f"  Total predictions:      {total_pred}")
    print(f"  Average GT per image:   {total_gt / len(results):.1f}")
    print(f"  Average pred per image: {total_pred / len(results):.1f}")
    
    if total_ious:
        print(f"\nIoU Statistics (for matched predictions):")
        print(f"  Mean IoU:    {np.mean(total_ious):.4f}")
        print(f"  Median IoU:  {np.median(total_ious):.4f}")
        print(f"  Min IoU:     {np.min(total_ious):.4f}")
        print(f"  Max IoU:     {np.max(total_ious):.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Visualize predictions from trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--model_type', type=str, default='transfer',
                        choices=['transfer', 'custom'],
                        help='Model type: transfer or custom')
    parser.add_argument('--data_dir', type=str, default='data_split',
                        help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--num_tiles', type=int, default=20,
                        help='Number of tiles to show in montage')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum confidence score to display')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("UNIVERSAL PREDICTION VISUALIZER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model path:       {args.model_path}")
    print(f"  Model type:       {args.model_type}")
    print(f"  Data directory:   {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device:           {device}")
    print(f"  Num samples:      {args.num_samples}")
    print(f"  Score threshold:  {args.score_threshold}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"\n❌ Error: Model file not found: {args.model_path}")
        return
    
    # Load test dataloader
    print("\n" + "-" * 80)
    print("Loading test dataset...")
    try:
        test_loader = get_dataloader(
            root_dir=args.data_dir,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
        )
        print(f"✓ Test dataset loaded: {len(test_loader.dataset)} images")
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        return
    
    # Load model
    print("-" * 80)
    try:
        model = load_model(
            args.model_path,
            model_type=args.model_type,
            num_classes=2,
            device=device
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate predictions
    print("-" * 80)
    try:
        results = predict_samples(
            model,
            test_loader,
            device,
            num_samples=args.num_samples
        )
        print(f"✓ Generated {len(results)} predictions")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Visualize predictions
    print("-" * 80)
    try:
        visualize_predictions(
            results,
            save_dir=args.output_dir,
            prefix=f'{args.model_type}_test',
            score_threshold=args.score_threshold
        )
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create tile montage
    print("-" * 80)
    try:
        create_tile_montage(
            test_loader,
            save_dir=args.output_dir,
            num_tiles=args.num_tiles,
            filename='test_tile_montage.png'
        )
    except Exception as e:
        print(f"\n❌ Error creating montage: {e}")
        import traceback
        traceback.print_exc()
    
    # Print statistics
    try:
        print_statistics(results)
    except Exception as e:
        print(f"\n⚠️ Warning: Could not compute statistics: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in '{args.output_dir}/':")
    print(f"  - {args.model_type}_test_prediction_1.png to {args.num_samples}.png")
    print(f"  - test_tile_montage.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()