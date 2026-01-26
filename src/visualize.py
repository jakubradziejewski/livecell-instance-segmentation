"""
Visualizer that reconstructs full images from tiles and shows best/median/worst results
"""

import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from tqdm import tqdm
from collections import defaultdict
import json

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


def parse_tile_filename(filename):
    """
    Extract original image name and tile index from tile filename.
    Format: originalname_tile_XX.png
    Returns: (original_name, tile_idx)
    """
    parts = filename.rsplit('_tile_', 1)
    if len(parts) == 2:
        original_name = parts[0]
        tile_idx = int(parts[1].replace('.png', ''))
        return original_name, tile_idx
    return None, None


@torch.no_grad()
def collect_all_predictions(model, dataloader, device):
    """
    Collect all predictions organized by original image.
    
    Returns:
        dict: {original_image_name: {
            'tiles': [(tile_idx, image, target, prediction), ...],
            'performance': float (mean IoU across tiles)
        }}
    """
    print("\nCollecting predictions from all tiles...")
    
    image_groups = defaultdict(lambda: {'tiles': [], 'performance': []})
    
    for images, targets in tqdm(dataloader, desc="Processing tiles"):
        images_device = [img.to(device) for img in images]
        predictions = model(images_device)
        
        for img, pred, target in zip(images, predictions, targets):
            # Get image filename from target
            img_id = target['image_id'].item()
            
            # Extract original image name from annotation file
            # We'll use the dataset's COCO object for this
            img_info = dataloader.dataset.coco.imgs[img_id]
            filename = img_info['file_name']
            
            original_name, tile_idx = parse_tile_filename(filename)
            
            if original_name is None:
                continue
            
            # Calculate IoU for this tile
            gt_boxes = target['boxes'].cpu()
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            
            # Filter predictions
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            
            # Calculate mean IoU
            tile_iou = 0.0
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                if len(iou_matrix) > 0:
                    max_ious, _ = iou_matrix.max(dim=1)
                    tile_iou = max_ious.mean().item()
            
            image_groups[original_name]['tiles'].append({
                'tile_idx': tile_idx,
                'image': img.cpu(),
                'target': {k: v.cpu() for k, v in target.items()},
                'prediction': {k: v.cpu() for k, v in pred.items()}
            })
            image_groups[original_name]['performance'].append(tile_iou)
        
        del images_device, predictions
        torch.cuda.empty_cache()
    
    # Calculate average performance for each image
    for img_name in image_groups:
        perfs = image_groups[img_name]['performance']
        image_groups[img_name]['avg_performance'] = np.mean(perfs) if perfs else 0.0
        
        # Sort tiles by index for proper reconstruction
        image_groups[img_name]['tiles'].sort(key=lambda x: x['tile_idx'])
    
    print(f"✓ Collected predictions for {len(image_groups)} original images")
    
    return dict(image_groups)


def reconstruct_full_image(tiles_data, tile_overlap=50):
    """
    Reconstruct full image and INSTANCE masks from tiles.
    Each cell instance gets a unique ID.
    
    Args:
        tiles_data: List of tile dictionaries with image, target, prediction
        tile_overlap: Overlap used during tiling
        
    Returns:
        Tuple of (full_image, gt_instances, pred_instances)
        where instances are [H, W] arrays with unique integer IDs per cell
    """
    if not tiles_data:
        return None, None, None
    
    # Get dimensions from first tile
    first_tile = tiles_data[0]['image']
    tile_h, tile_w = first_tile.shape[1], first_tile.shape[2]
    
    # Estimate grid size (assuming roughly square grid)
    n_tiles = len(tiles_data)
    n_cols = int(np.sqrt(n_tiles))
    n_rows = int(np.ceil(n_tiles / n_cols))
    
    # Calculate full image size
    full_w = n_cols * tile_w - (n_cols - 1) * tile_overlap
    full_h = n_rows * tile_h - (n_rows - 1) * tile_overlap
    
    # Initialize full arrays
    full_image = torch.zeros(3, full_h, full_w)
    gt_instances = torch.zeros(full_h, full_w, dtype=torch.int32)
    pred_instances = torch.zeros(full_h, full_w, dtype=torch.int32)
    
    # Track coverage for proper blending
    coverage = torch.zeros(full_h, full_w)
    
    # Instance ID counters (each cell gets unique ID)
    gt_instance_id = 1
    pred_instance_id = 1
    
    # Place each tile
    for tile_data in tiles_data:
        tile_idx = tile_data['tile_idx']
        
        # Calculate tile position
        row = tile_idx // n_cols
        col = tile_idx % n_cols
        
        x_start = col * (tile_w - tile_overlap)
        y_start = row * (tile_h - tile_overlap)
        
        x_end = min(x_start + tile_w, full_w)
        y_end = min(y_start + tile_h, full_h)
        
        tile_x_end = x_end - x_start
        tile_y_end = y_end - y_start
        
        # Place image
        full_image[:, y_start:y_end, x_start:x_end] += tile_data['image'][:, :tile_y_end, :tile_x_end]
        coverage[y_start:y_end, x_start:x_end] += 1
        
        # Place GT masks - EACH INSTANCE GETS UNIQUE ID
        target_masks = tile_data['target']['masks']
        for mask in target_masks:
            mask_region = mask[:tile_y_end, :tile_x_end] > 0
            if mask_region.any():
                # Assign unique ID to this instance
                full_region = gt_instances[y_start:y_end, x_start:x_end]
                full_region[mask_region] = gt_instance_id
                gt_instance_id += 1
        
        # Place prediction masks - EACH INSTANCE GETS UNIQUE ID
        pred_masks = tile_data['prediction']['masks']
        for mask in pred_masks:
            mask_region = mask[:tile_y_end, :tile_x_end] > 0
            if mask_region.any():
                # Assign unique ID to this instance
                full_region = pred_instances[y_start:y_end, x_start:x_end]
                full_region[mask_region] = pred_instance_id
                pred_instance_id += 1
    
    # Average overlapping regions for image
    coverage = coverage.clamp(min=1)
    full_image = full_image / coverage.unsqueeze(0)
    
    return full_image, gt_instances, pred_instances


def create_colored_instance_map(instance_map, num_colors=1000):
    """
    Convert instance map to RGB image where each instance has a unique color.
    
    Args:
        instance_map: [H, W] array with integer instance IDs (0 = background)
        num_colors: Number of distinct colors to generate
        
    Returns:
        [H, W, 3] RGB array with colored instances
    """
    h, w = instance_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Get unique instance IDs (excluding background = 0)
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    if len(unique_ids) == 0:
        return colored
    
    # Generate distinct colors using HSV color space
    np.random.seed(42)  # For reproducibility
    colors = []
    for i in range(num_colors):
        hue = (i * 137.508) % 360  # Golden angle for better distribution
        saturation = 0.6 + (i % 3) * 0.15  # Vary saturation
        value = 0.8 + (i % 2) * 0.2  # Vary brightness
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    # Assign colors to instances
    for idx, instance_id in enumerate(unique_ids):
        mask = instance_map == instance_id
        color_idx = idx % len(colors)
        colored[mask] = colors[color_idx]
    
    return colored


def visualize_reconstruction(img_name, full_image, gt_instances, pred_instances, performance, save_path):
    """
    Create 3-panel visualization with COLORED INSTANCE SEGMENTATION.
    Each cell gets a unique color.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Convert image to numpy
    img_np = full_image.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Convert instance maps to numpy
    gt_instances_np = gt_instances.numpy()
    pred_instances_np = pred_instances.numpy()
    
    # Create colored instance maps
    gt_colored = create_colored_instance_map(gt_instances_np)
    pred_colored = create_colored_instance_map(pred_instances_np)
    
    # Count instances
    n_gt = len(np.unique(gt_instances_np)) - 1  # -1 to exclude background
    n_pred = len(np.unique(pred_instances_np)) - 1
    
    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # Panel 2: Ground Truth Instances (colored)
    axes[1].imshow(img_np)
    axes[1].imshow(gt_colored, alpha=0.6)
    axes[1].set_title(f'Ground Truth\n({n_gt} cells)', 
                     fontsize=16, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # Panel 3: Predicted Instances (colored)
    axes[2].imshow(img_np)
    axes[2].imshow(pred_colored, alpha=0.6)
    axes[2].set_title(f'Predictions\n({n_pred} cells, IoU: {performance:.3f})', 
                     fontsize=16, fontweight='bold', pad=15)
    axes[2].axis('off')
    
    # Overall title
    fig.suptitle(f'Instance Segmentation: {img_name}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add legend explaining colors
    fig.text(0.5, 0.02, 'Each cell instance has a unique color', 
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path} (GT: {n_gt} cells, Pred: {n_pred} cells)")


def select_representative_images(image_groups, n_best=2, n_median=1, n_worst=2):
    """
    Select best, median, and worst performing images.
    
    Returns:
        List of (img_name, performance, category) tuples
    """
    # Sort by performance
    sorted_images = sorted(
        image_groups.items(),
        key=lambda x: x[1]['avg_performance'],
        reverse=True
    )
    
    if len(sorted_images) < (n_best + n_median + n_worst):
        # If not enough images, just return all
        return [(name, data['avg_performance'], 'available') 
                for name, data in sorted_images]
    
    selected = []
    
    # Best performers
    for i in range(n_best):
        name, data = sorted_images[i]
        selected.append((name, data['avg_performance'], 'best'))
    
    # Median performer
    median_idx = len(sorted_images) // 2
    for i in range(median_idx, median_idx + n_median):
        if i < len(sorted_images):
            name, data = sorted_images[i]
            selected.append((name, data['avg_performance'], 'median'))
    
    # Worst performers
    for i in range(n_worst):
        idx = -(i + 1)
        if abs(idx) <= len(sorted_images):
            name, data = sorted_images[idx]
            selected.append((name, data['avg_performance'], 'worst'))
    
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Visualize reconstructed full images from tiles'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--data_dir', type=str, default='data_split_a172',
                        help='Path to preprocessed data directory')
    parser.add_argument('--output_dir', type=str, default='outputs/reconstructed',
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--tile_overlap', type=int, default=50,
                        help='Tile overlap used during preprocessing')
    parser.add_argument('--n_best', type=int, default=2,
                        help='Number of best performing images to show')
    parser.add_argument('--n_median', type=int, default=1,
                        help='Number of median performing images to show')
    parser.add_argument('--n_worst', type=int, default=2,
                        help='Number of worst performing images to show')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("TILE RECONSTRUCTION VISUALIZER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {device}")
    print(f"  Tile overlap: {args.tile_overlap}px")
    print(f"  Selection: {args.n_best} best, {args.n_median} median, {args.n_worst} worst")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_loader = get_dataloader(
        root_dir=args.data_dir,
        split='test',
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False
    )
    print(f"\n✓ Loaded test dataset: {len(test_loader.dataset)} tile images")
    
    # Load model
    model = load_model(args.model_path, model_type='custom', device=device)
    
    # Collect all predictions
    image_groups = collect_all_predictions(model, test_loader, device)
    
    # Select representative images
    print("\nSelecting representative images...")
    selected = select_representative_images(
        image_groups,
        n_best=args.n_best,
        n_median=args.n_median,
        n_worst=args.n_worst
    )
    
    print(f"✓ Selected {len(selected)} images for visualization")
    
    # Generate visualizations
    print("\nGenerating reconstructed visualizations...")
    print("-" * 80)
    
    results_summary = []
    
    for img_name, performance, category in selected:
        print(f"\nProcessing: {img_name} ({category}, IoU: {performance:.3f})")
        
        # Reconstruct full image with instance-level masks
        tiles_data = image_groups[img_name]['tiles']
        full_image, gt_instances, pred_instances = reconstruct_full_image(
            tiles_data,
            tile_overlap=args.tile_overlap
        )
        
        if full_image is None:
            print(f"  ⚠ Failed to reconstruct {img_name}")
            continue
        
        # Save visualization
        save_path = os.path.join(
            args.output_dir,
            f'{category}_{img_name}_reconstructed.png'
        )
        
        # Create colored instance visualization
        visualize_reconstruction(
            img_name, full_image, gt_instances, pred_instances, performance, save_path
        )
        
        results_summary.append({
            'image_name': img_name,
            'category': category,
            'performance': performance,
            'num_tiles': len(tiles_data)
        })
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGenerated {len(results_summary)} reconstructed visualizations:")
    
    for result in results_summary:
        print(f"  [{result['category']:6s}] {result['image_name']:40s} "
              f"IoU: {result['performance']:.3f} ({result['num_tiles']} tiles)")
    
    print(f"\n✓ All visualizations saved to: {args.output_dir}/")
    print(f"✓ Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()