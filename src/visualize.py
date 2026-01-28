import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image, ImageDraw
from torchvision import transforms
from collections import defaultdict
import re
import json
from pycocotools import mask as maskUtils

IMG_WIDTH = 704
IMG_HEIGHT = 520

N_MINI_COLS = 7  # Number of mini-tile columns in the grid (7 for 5x5 tiles)
N_MINI_ROWS = 7  # Number of mini-tile rows in the grid (7 for 5x5 tiles)
TILE_SIZE = 3    # Each tile is TILE_SIZE x TILE_SIZE mini-tiles (3x3)
N_TILES_COL = N_MINI_COLS - TILE_SIZE + 1  # Number of tile positions horizontally (7-3+1 = 5)
N_TILES_ROW = N_MINI_ROWS - TILE_SIZE + 1  # Number of tile positions vertically (7-3+1 = 5)
TOTAL_TILES = N_TILES_COL * N_TILES_ROW


def load_model(model_path, model_type='custom', num_classes=2, device='cuda'):
    print(f"Loading {model_type} model from {model_path}...")
    
    if model_type == 'custom':
        from custom_maskrcnn import get_custom_model
        
        model = get_custom_model(num_classes=num_classes)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    elif model_type == 'transfer':
        model = maskrcnn_resnet50_fpn(pretrained=False)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.to(device)
    model.eval()
    
    return model


def group_tiles_by_image(test_dir):
    tiles_by_image = defaultdict(list)
    pattern = re.compile(r'^(.+)_tile_(\d{2})\.png$')
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist.")
        return {}

    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith('.png'):
            match = pattern.match(filename)
            if match:
                base_name = match.group(1)
                tile_num = int(match.group(2))
                tiles_by_image[base_name].append({
                    'path': os.path.join(test_dir, filename),
                    'tile_num': tile_num,
                    'filename': filename
                })
    
    for base_name in tiles_by_image:
        tiles_by_image[base_name] = sorted(
            tiles_by_image[base_name], 
            key=lambda x: x['tile_num']
        )
    return dict(tiles_by_image)


def get_tile_position_in_grid(tile_num):
    row_start = tile_num // N_TILES_COL
    col_start = tile_num % N_TILES_COL
    return col_start, row_start


def calculate_mask_area_in_region(mask, region, tile_offset):
    tile_x_offset, tile_y_offset = tile_offset
    
    # Convert region from global to tile-local coordinates
    reg_x_min = region[0] - tile_x_offset
    reg_y_min = region[1] - tile_y_offset
    reg_x_max = region[2] - tile_x_offset
    reg_y_max = region[3] - tile_y_offset
    
    # Clip to mask boundaries
    mask_h, mask_w = mask.shape
    reg_x_min = max(0, reg_x_min)
    reg_y_min = max(0, reg_y_min)
    reg_x_max = min(mask_w, reg_x_max)
    reg_y_max = min(mask_h, reg_y_max)
    
    if reg_x_min >= reg_x_max or reg_y_min >= reg_y_max:
        return 0.0
    
    # Count pixels in region
    region_mask = mask[int(reg_y_min):int(reg_y_max), int(reg_x_min):int(reg_x_max)]
    pixels_in_region = region_mask.sum()
    total_pixels = mask.sum()
    
    return float(pixels_in_region / total_pixels) if total_pixels > 0 else 0.0


@torch.no_grad()
def predict_on_tiles(model, tiles_info, device, transform):
    model.eval()
    results = []
    for tile_info in tiles_info:
        image = Image.open(tile_info['path']).convert('RGB')
        image_tensor = transform(image).to(device)
        prediction = model([image_tensor])[0]
        results.append({
            'tile_num': tile_info['tile_num'],
            'image': image_tensor.cpu(),
            'prediction': {k: v.cpu() for k, v in prediction.items()}
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def get_valid_mini_tiles_for_tile(tile_num):
    col_start, row_start = get_tile_position_in_grid(tile_num)
    
    valid_mini_tiles = []
    
    for local_row in range(TILE_SIZE):
        for local_col in range(TILE_SIZE):
            mini_col = col_start + local_col
            mini_row = row_start + local_row
            
            # Check if this is the center mini-tile of the 3x3 tile
            is_center = (local_col == 1 and local_row == 1)
            
            # Check if this mini-tile is on the border of the grid
            is_on_border = (mini_col == 0 or mini_col == N_MINI_COLS - 1 or 
                           mini_row == 0 or mini_row == N_MINI_ROWS - 1)
            
            if is_center or is_on_border:
                valid_mini_tiles.append((mini_col, mini_row))
    
    return valid_mini_tiles


def filter_detections_by_border_mini_tiles(results, score_threshold=0.5, mask_threshold=0.4):
    mini_tile_width = IMG_WIDTH // N_MINI_COLS
    mini_tile_height = IMG_HEIGHT // N_MINI_ROWS
    
    filtered_detections = []
    processed_mini_tiles = set()
    
    # Sort results by tile_num to ensure consistent processing order
    sorted_results = sorted(results, key=lambda x: x['tile_num'])
    
    for result in sorted_results:
        tile_num = result['tile_num']
        pred = result['prediction']
        
        col_start, row_start = get_tile_position_in_grid(tile_num)
        
        # Calculate tile offset in global coordinates
        tile_x_offset = col_start * mini_tile_width
        tile_y_offset = row_start * mini_tile_height
        
        # Get valid mini-tiles for this tile (border OR center mini-tiles not yet processed)
        valid_mini_tiles = get_valid_mini_tiles_for_tile(tile_num)
        
        new_mini_tiles = [mt for mt in valid_mini_tiles if mt not in processed_mini_tiles]
        
        if not new_mini_tiles:
            continue
        
        keep = pred['scores'] > score_threshold
        boxes = pred['boxes'][keep]
        masks = pred['masks'][keep]
        scores = pred['scores'][keep]
        
        # Create combined region from all valid mini-tiles for this tile
        # This allows cells that span multiple valid mini-tiles to be captured
        valid_mini_tile_regions = []
        
        for mini_col, mini_row in new_mini_tiles:
            
            # Calculate mini-tile region in global coordinates
            mini_tile_x_min = mini_col * mini_tile_width
            mini_tile_y_min = mini_row * mini_tile_height
            mini_tile_x_max = mini_tile_x_min + mini_tile_width
            mini_tile_y_max = mini_tile_y_min + mini_tile_height
            
            valid_mini_tile_regions.append((mini_tile_x_min, mini_tile_y_min, mini_tile_x_max, mini_tile_y_max))
        
        # For each detection, check if it belongs to the combined valid region
        for box, mask, score in zip(boxes, masks, scores):
            box_np = box.numpy()
            mask_np = (mask.squeeze().numpy() > 0.5)
            
            # Calculate total area of mask in all valid mini-tiles combined
            total_area_in_valid_region = 0.0
            
            for region in valid_mini_tile_regions:
                area_fraction = calculate_mask_area_in_region(
                    mask_np, region, (tile_x_offset, tile_y_offset)
                )
                total_area_in_valid_region += area_fraction
            
            # Only keep if mask area in valid region exceeds threshold
            if total_area_in_valid_region > mask_threshold:
                # Convert coordinates to global image space
                global_box = [
                    box_np[0] + tile_x_offset,
                    box_np[1] + tile_y_offset,
                    box_np[2] + tile_x_offset,
                    box_np[3] + tile_y_offset
                ]
                
                filtered_detections.append({
                    'box': global_box,
                    'mask': mask_np,
                    'score': float(score),
                    'tile_num': tile_num,
                    'offset': (tile_x_offset, tile_y_offset),
                    'area_fraction': total_area_in_valid_region,
                    'mini_tile': new_mini_tiles  # Store all valid mini-tiles
                })
        
        processed_mini_tiles.update(new_mini_tiles)
    
    return filtered_detections


def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    images_dict = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in images_dict:
            filename = images_dict[image_id]
            annotations_by_image[filename].append(ann)
    return dict(annotations_by_image), images_dict


def find_original_image(base_name, images_dict):
    for filename in images_dict.values():
        name_without_ext = os.path.splitext(filename)[0]
        if base_name == name_without_ext or base_name == filename:
            return filename
    return None


def reconstruct_full_image(results):
    mini_tile_width = IMG_WIDTH // N_MINI_COLS
    mini_tile_height = IMG_HEIGHT // N_MINI_ROWS
    
    full_canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    tile_coverage = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)
    
    for result in results:
        tile_num = result['tile_num']
        col_start, row_start = get_tile_position_in_grid(tile_num)
        
        tile_x_offset = col_start * mini_tile_width
        tile_y_offset = row_start * mini_tile_height
        
        img_np = result['image'].permute(1, 2, 0).numpy()
        h, w, _ = img_np.shape
        
        # Place tile pixels that haven't been covered yet
        y_end = min(tile_y_offset + h, IMG_HEIGHT)
        x_end = min(tile_x_offset + w, IMG_WIDTH)
        
        uncovered = ~tile_coverage[tile_y_offset:y_end, tile_x_offset:x_end]
        
        full_canvas[tile_y_offset:y_end, tile_x_offset:x_end][uncovered] = \
            np.clip(img_np[:y_end-tile_y_offset, :x_end-tile_x_offset], 0, 1)[uncovered]
        
        tile_coverage[tile_y_offset:y_end, tile_x_offset:x_end] = True
    
    return full_canvas


def create_mask_overlay(filtered_detections):
    mask_canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 4))
    
    for idx, det in enumerate(filtered_detections):
        offset_x, offset_y = det['offset']
        mask = det['mask']
        
        # Create colored mask
        color = plt.cm.tab20(idx % 20)
        
        mask_h, mask_w = mask.shape
        
        # Calculate where the mask should be placed on the canvas
        y_start = offset_y
        x_start = offset_x
        y_end = offset_y + mask_h
        x_end = offset_x + mask_w
        
        # Calculate overlap region with canvas
        canvas_y_start = max(0, y_start)
        canvas_x_start = max(0, x_start)
        canvas_y_end = min(IMG_HEIGHT, y_end)
        canvas_x_end = min(IMG_WIDTH, x_end)
        
        # Calculate corresponding region in mask
        mask_y_start = canvas_y_start - y_start
        mask_x_start = canvas_x_start - x_start
        mask_y_end = mask_y_start + (canvas_y_end - canvas_y_start)
        mask_x_end = mask_x_start + (canvas_x_end - canvas_x_start)
        
        # Only place the part of mask that overlaps with canvas
        if canvas_y_start < canvas_y_end and canvas_x_start < canvas_x_end:
            mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            mask_canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end][mask_region] = (*color[:3], 0.5)
    
    return mask_canvas


def create_gt_mask_overlay(annotations):
    mask_overlay = np.zeros((IMG_HEIGHT, IMG_WIDTH, 4))
    
    instance_counter = 0
    for ann in annotations:
        if 'segmentation' in ann:
            # Decode COCO RLE or polygon segmentation
            if isinstance(ann['segmentation'], dict):
                # RLE format
                rle = ann['segmentation']
                binary_mask = maskUtils.decode(rle)
            elif isinstance(ann['segmentation'], list):
                # Polygon format - create mask using PIL
                mask_pil = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)
                draw = ImageDraw.Draw(mask_pil)
                for polygon in ann['segmentation']:
                    # Convert polygon to pairs of (x, y)
                    poly_points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(poly_points, outline=1, fill=1)
                binary_mask = np.array(mask_pil)
            else:
                continue
            
            # Create colored mask
            color = plt.cm.tab20(instance_counter % 20)
            colored_mask = np.zeros((*binary_mask.shape, 4))
            colored_mask[binary_mask > 0] = color
            colored_mask[binary_mask > 0, 3] = 0.5  # Set alpha
            
            mask_overlay = np.maximum(mask_overlay, colored_mask)
            instance_counter += 1
    
    return mask_overlay, instance_counter


def visualize_with_ground_truth(base_name, original_img_path, annotations, 
                                results_list, model_names,
                                save_dir='outputs', score_threshold=0.5, mask_threshold=0.4):

    os.makedirs(save_dir, exist_ok=True)
    
    num_models = len(results_list)
    num_plots = 1 + num_models
    
    try:
        original_img = Image.open(original_img_path).convert('RGB')
        original_img_np = np.array(original_img)
    except Exception as e:
        print(f"  Warning: Could not load original image for GT: {e}")
        original_img_np = None
    
    full_canvas = reconstruct_full_image(results_list[0])
    
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 10))
    if num_plots == 1:
        axes = [axes]
    
    # Plot 1: Ground Truth
    if original_img_np is not None:
        axes[0].imshow(original_img_np)
        gt_overlay, gt_count = create_gt_mask_overlay(annotations)
        axes[0].imshow(gt_overlay)
        axes[0].set_title(f"Ground Truth: {base_name}\nInstances: {gt_count}", fontsize=12)
    else:
        axes[0].text(0.5, 0.5, "GT Not Available", ha='center', va='center', fontsize=14)
        axes[0].set_title(f"Ground Truth: {base_name}", fontsize=12)
    axes[0].axis('off')
    
    # Plot 2: Model Predictions
    for idx, (results, model_name) in enumerate(zip(results_list, model_names)):
        ax = axes[idx + 1]
        
        filtered_det = filter_detections_by_border_mini_tiles(
            results, score_threshold, mask_threshold
        )
        
        ax.imshow(full_canvas)
        
        if filtered_det:
            mask_overlay = create_mask_overlay(filtered_det)
            ax.imshow(mask_overlay)
            
            # add confidence scores
            for det in filtered_det:
                mask = det['mask']
                offset_x, offset_y = det['offset']
                score = det['score']
                
                if mask.any():
                    y_coords, x_coords = np.where(mask)
                    center_y, center_x = y_coords.mean() + offset_y, x_coords.mean() + offset_x
                    ax.text(
                        center_x, center_y, 
                        f'{score:.2f}',
                        color='white', fontsize=6, weight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1)
                    )
        
        ax.set_title(f"{model_name}: {base_name}\nInstances: {len(filtered_det)}", fontsize=12)
        ax.axis('off')
    
    plt.suptitle(f"Ground Truth vs Predictions | Score>{score_threshold} | Mask>{mask_threshold*100:.0f}%", 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{base_name}_GT_VS_PREDICTIONS.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved GT vs Predictions: {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Mask R-CNN dual model inference with tiled images')
    parser.add_argument('--model1_path', type=str, default='models/custom_maskrcnn_5epochs.pth',
                        help='Path to first model .pth file')
    parser.add_argument('--model1_type', type=str, default='custom', choices=['custom', 'transfer'],
                        help='Type of first model (custom or transfer)')
    parser.add_argument('--model1_name', type=str, default='Custom Model',
                        help='Display name for first model')
    
    parser.add_argument('--model2_path', type=str, default=None,
                        help='Path to second model .pth file (optional)')
    parser.add_argument('--model2_type', type=str, default='transfer', choices=['custom', 'transfer'],
                        help='Type of second model (custom or transfer)')
    parser.add_argument('--model2_name', type=str, default='Transfer Learning Model',
                        help='Display name for second model')
    
    parser.add_argument('--test_dir', type=str, default='data_split/test/images',
                        help='Directory containing test tiles')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save output visualizations')
    
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum confidence score for detections')
    parser.add_argument('--mask_threshold', type=float, default=0.4,
                        help='Minimum fraction of mask that must be in valid mini-tiles (default: 0.4 = 40%%)')
    
    args = parser.parse_args()
    
    print("\nVisualization\n")
    print(f"Model 1 ({args.model1_name}): {args.model1_path} (type: {args.model1_type})")
    if args.model2_path:
        print(f"Model 2 ({args.model2_name}): {args.model2_path} (type: {args.model2_type})")
    print(f"Test directory: {args.test_dir}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Mask threshold: {args.mask_threshold} ({args.mask_threshold*100:.0f}%)")
    

    test_dir = args.test_dir
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_splits = [
        {'name': 'Test', 'images': 'data/test/images', 'ann': 'data/annotations/livecell_coco_test.json'},
        {'name': 'Train', 'images': 'data/train/images', 'ann': 'data/annotations/livecell_coco_train.json'},
        {'name': 'Val', 'images': 'data/val/images', 'ann': 'data/annotations/livecell_coco_val.json'}
    ]
    

    global_ann_by_image = {}
    global_images_dict = {}
    filename_to_dir = {}


    for split in data_splits:
        if os.path.exists(split['ann']):
            anns, imgs = load_coco_annotations(split['ann'])
            global_ann_by_image.update(anns)
            global_images_dict.update(imgs)
            for fname in imgs.values():
                filename_to_dir[fname] = split['images']


    tiles_by_image = group_tiles_by_image(test_dir)
    print(f"\nProcessing {len(tiles_by_image)} image sets found in {test_dir}")


    model1 = load_model(args.model1_path, args.model1_type, num_classes, device)
    
    model2 = None
    if args.model2_path:
        model2 = load_model(args.model2_path, args.model2_type, num_classes, device)
        print()

    transform = transforms.Compose([transforms.ToTensor()])
    
    for base_name, tiles_info in tiles_by_image.items():
        print(f"\nWorking on: {base_name}")
        
        original_filename = find_original_image(base_name, global_images_dict)
        original_img_path = None
        annotations = []
        
        if original_filename and original_filename in filename_to_dir:
            orig_path = os.path.join(filename_to_dir[original_filename], original_filename)
            if os.path.exists(orig_path):
                original_img_path = orig_path
                annotations = global_ann_by_image.get(original_filename, [])
        
        results_model1 = predict_on_tiles(model1, tiles_info, device, transform)
        
        results_list = [results_model1]
        model_names = [args.model1_name]
        
        if model2:
            results_model2 = predict_on_tiles(model2, tiles_info, device, transform)
            results_list.append(results_model2)
            model_names.append(args.model2_name)
        
        if results_list[0] and original_img_path:
            visualize_with_ground_truth(
                base_name, 
                original_img_path,
                annotations,
                results_list,
                model_names,
                save_dir=args.output_dir,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold
            )


    print("\nFinished. Visualizations saved to", args.output_dir)


if __name__ == "__main__":
    main()