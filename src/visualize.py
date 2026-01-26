"""
Enhanced Visualizer for Instance Segmentation
Shows actual pixel-level masks for each detected cell instance
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.append('src')
from dataset import get_dataloader


def load_model(model_path, model_type='custom', num_classes=2, device='cuda'):
    """Load trained model."""
    print(f"Loading {model_type} model from {model_path}...")
    
    if model_type == 'custom':
        from custom_maskrcnn import get_custom_model
        
        model = get_custom_model(num_classes=num_classes, pretrained_backbone=False)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
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


def create_instance_overlay(image, masks, boxes, scores, alpha=0.5):
    """
    Create a visualization with instance masks overlaid on the image.
    Each instance gets a different color.
    
    Args:
        image: [H, W, 3] numpy array
        masks: [N, H, W] tensor of binary masks
        boxes: [N, 4] tensor of bounding boxes
        scores: [N] tensor of confidence scores
        alpha: transparency of mask overlay
    
    Returns:
        overlaid_image: numpy array with masks overlaid
    """
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Generate distinct colors for each instance
    np.random.seed(42)
    colors = np.random.rand(len(masks), 3)
    
    # Create colored mask overlay
    for idx, (mask, color, score) in enumerate(zip(masks, colors, scores)):
        mask_np = mask.cpu().numpy().astype(bool)
        
        # Create colored mask
        colored_mask = np.zeros((h, w, 3))
        colored_mask[mask_np] = color
        
        # Blend with original image
        overlay = np.where(
            mask_np[..., None],
            overlay * (1 - alpha) + colored_mask * 255 * alpha,
            overlay
        )
    
    return overlay.astype(np.uint8)


def visualize_instance_segmentation(result, save_path, score_threshold=0.5):
    """
    Create comprehensive instance segmentation visualization.
    Shows: original image, ground truth, predictions with masks, and individual instances.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Original image
    img = result['image'].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    
    target = result['target']
    pred = result['prediction']
    
    # Filter predictions by score
    keep = pred['scores'] > score_threshold
    pred_boxes = pred['boxes'][keep]
    pred_scores = pred['scores'][keep]
    pred_masks = pred['masks'][keep]
    
    # ===== Plot 1: Original Image =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # ===== Plot 2: Ground Truth Masks =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create GT mask overlay
    if len(target['masks']) > 0:
        gt_masks_np = target['masks'].cpu().numpy()
        gt_colored = np.zeros((*img_uint8.shape[:2], 3), dtype=np.uint8)
        
        # Assign each GT instance a unique color
        colors = np.random.RandomState(42).rand(len(gt_masks_np), 3)
        for idx, (mask, color) in enumerate(zip(gt_masks_np, colors)):
            gt_colored[mask > 0] = (color * 255).astype(np.uint8)
        
        ax2.imshow(img_uint8)
        ax2.imshow(gt_colored, alpha=0.5)
    else:
        ax2.imshow(img)
    
    ax2.set_title(f'Ground Truth\n({len(target["boxes"])} instances)', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # ===== Plot 3: Predicted Masks Overlay =====
    ax3 = fig.add_subplot(gs[0, 2])
    
    if len(pred_masks) > 0:
        overlay = create_instance_overlay(
            img_uint8, pred_masks, pred_boxes, pred_scores, alpha=0.5
        )
        ax3.imshow(overlay)
    else:
        ax3.imshow(img)
    
    ax3.set_title(f'Predicted Masks\n({len(pred_boxes)} instances, score>{score_threshold})', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # ===== Plot 4: Predictions with Bounding Boxes =====
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img)
    
    # Calculate IoU for color coding
    if len(target['boxes']) > 0 and len(pred_boxes) > 0:
        iou_matrix = box_iou(pred_boxes, target['boxes'])
        max_ious, _ = iou_matrix.max(dim=1)
    else:
        max_ious = torch.zeros(len(pred_boxes))
    
    # Draw boxes with color based on IoU
    for box, score, iou in zip(pred_boxes, pred_scores, max_ious):
        x1, y1, x2, y2 = box.cpu()
        
        # Color code: green for good match, red for poor match
        color = 'green' if iou > 0.5 else 'orange' if iou > 0.3 else 'red'
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax4.add_patch(rect)
        
        ax4.text(
            x1, y1 - 5, f'{score:.2f}',
            color=color, fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
    
    ax4.set_title('Bounding Boxes\n(Green=Good, Orange=OK, Red=Poor)', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # ===== Plot 5-8: Individual Instance Masks (Top 4) =====
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        
        if i < len(pred_masks):
            mask = pred_masks[i].cpu().numpy()
            score = pred_scores[i].item()
            box = pred_boxes[i].cpu().numpy()
            
            # Show masked region
            masked_img = img_uint8.copy()
            masked_img[mask == 0] = masked_img[mask == 0] * 0.3  # Dim background
            
            ax.imshow(masked_img)
            
            # Draw box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='cyan', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Calculate mask stats
            mask_area = mask.sum()
            box_area = (x2 - x1) * (y2 - y1)
            fill_ratio = mask_area / box_area if box_area > 0 else 0
            
            ax.set_title(f'Instance {i+1}\nScore: {score:.2f}, Fill: {fill_ratio:.1%}', 
                        fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No instance', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Instance {i+1}', fontsize=10)
        
        ax.axis('off')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def predict_and_visualize(model, dataloader, device, save_dir, num_samples=5, score_threshold=0.5):
    """Generate predictions and create visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    print(f"\nGenerating predictions on {num_samples} samples...")
    
    for images, targets in tqdm(dataloader, desc="Predicting"):
        if len(results) >= num_samples:
            break
        
        images_device = [img.to(device) for img in images]
        predictions = model(images_device)
        
        for img, pred, target in zip(images, predictions, targets):
            if len(results) >= num_samples:
                break
            
            results.append({
                'image': img.cpu(),
                'prediction': {k: v.cpu() for k, v in pred.items()},
                'target': {k: v.cpu() for k, v in target.items()}
            })
        
        del images_device, predictions
        torch.cuda.empty_cache()
    
    print(f"\nCreating instance segmentation visualizations...")
    for idx, result in enumerate(tqdm(results, desc="Visualizing")):
        save_path = os.path.join(save_dir, f'instance_seg_{idx+1}.png')
        visualize_instance_segmentation(result, save_path, score_threshold)
    
    return results


def print_mask_statistics(results):
    """Print statistics about the predicted masks."""
    print("\n" + "=" * 80)
    print("INSTANCE SEGMENTATION STATISTICS")
    print("=" * 80)
    
    total_instances = 0
    total_mask_pixels = 0
    total_box_pixels = 0
    mask_qualities = []
    
    for result in results:
        pred = result['prediction']
        keep = pred['scores'] > 0.5
        
        pred_masks = pred['masks'][keep]
        pred_boxes = pred['boxes'][keep]
        
        total_instances += len(pred_masks)
        
        for mask, box in zip(pred_masks, pred_boxes):
            mask_np = mask.cpu().numpy()
            mask_area = (mask_np > 0).sum()
            
            x1, y1, x2, y2 = box.cpu()
            box_area = (x2 - x1) * (y2 - y1)
            
            total_mask_pixels += mask_area
            total_box_pixels += box_area.item()
            
            # Mask quality: ratio of mask area to box area
            quality = mask_area / box_area.item() if box_area > 0 else 0
            mask_qualities.append(quality)
    
    print(f"\nAcross {len(results)} test images:")
    print(f"  Total instances detected: {total_instances}")
    print(f"  Average instances per image: {total_instances / len(results):.1f}")
    
    if mask_qualities:
        print(f"\nMask Quality Metrics:")
        print(f"  Mean mask fill ratio: {np.mean(mask_qualities):.2%}")
        print(f"  Median mask fill ratio: {np.median(mask_qualities):.2%}")
        print(f"  Min fill ratio: {np.min(mask_qualities):.2%}")
        print(f"  Max fill ratio: {np.max(mask_qualities):.2%}")
        print(f"\n  (Fill ratio = mask pixels / bounding box pixels)")
        print(f"  (Higher is better, ~50-80% is typical for cells)")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Visualize instance segmentation predictions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--data_dir', type=str, default='data_split',
                        help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/instance_seg',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum confidence score to display')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("INSTANCE SEGMENTATION VISUALIZER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {device}")
    print(f"  Samples: {args.num_samples}")
    
    # Load data
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False
    )
    print(f"\n✓ Loaded test dataset: {len(test_loader.dataset)} images")
    
    # Load model
    model = load_model(args.model_path, model_type='custom', device=device)
    
    # Generate visualizations
    results = predict_and_visualize(
        model, test_loader, device, 
        args.output_dir, args.num_samples, 
        args.score_threshold
    )
    
    # Print statistics
    print_mask_statistics(results)
    
    print(f"\n✓ Visualizations saved to: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()