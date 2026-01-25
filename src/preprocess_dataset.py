"""
LIVECell Dataset Preprocessor with Image Tiling
Splits large images into manageable tiles for training on constrained GPU
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import random


class LIVECellPreprocessor:
    """
    Preprocesses LIVECell dataset by:
    1. Selecting subset of images
    2. Splitting into train/val/test
    3. Tiling large images into smaller patches
    4. Remapping annotations to tiles
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        num_images: int = 100,
        tiles_per_image: int = 20,
        tile_overlap: int = 50,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42
    ):
        """
        Args:
            source_dir: Original LIVECell data directory (contains 'images' folder and 'annotations' folder)
            output_dir: Where to save preprocessed data
            num_images: Number of images to select from dataset
            tiles_per_image: How many tiles to create per image (approx)
            tile_overlap: Overlap between adjacent tiles in pixels
            splits: (train, val, test) split ratios
            random_seed: For reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.tiles_per_image = tiles_per_image
        self.tile_overlap = tile_overlap
        self.splits = splits
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Validate splits
        assert abs(sum(splits) - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # Detect dataset structure
        self._detect_structure()
        
    def _detect_structure(self):
        """
        Auto-detect LIVECell dataset structure.
        Can be either:
        - images/ and annotations/ at root
        - train/images/, val/images/, etc.
        """
        print(f"Detecting dataset structure in {self.source_dir}...")
        
        # Check for direct structure (images/ and annotations/)
        if (self.source_dir / 'images').exists():
            self.images_dir = self.source_dir / 'images'
            self.annotations_dir = self.source_dir / 'annotations'
            self.structure_type = 'flat'
            print(f"  ✓ Detected flat structure: images/ and annotations/")
        # Check for split structure (train/images/, etc.)
        elif (self.source_dir / 'train' / 'images').exists():
            self.images_dir = self.source_dir / 'train' / 'images'
            self.annotations_dir = self.source_dir / 'annotations'
            self.structure_type = 'split'
            print(f"  ✓ Detected split structure: train/images/, etc.")
        else:
            raise ValueError(
                f"Cannot detect valid LIVECell structure in {self.source_dir}\n"
                f"Expected either:\n"
                f"  - {self.source_dir}/images/ and {self.source_dir}/annotations/\n"
                f"  - {self.source_dir}/train/images/ and {self.source_dir}/annotations/"
            )
        
        # Find annotation file
        possible_ann_files = [
            'livecell_coco_train.json',
            'LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json',
            'livecell_train.json'
        ]
        
        self.ann_file = None
        for ann_file in possible_ann_files:
            full_path = self.annotations_dir / ann_file
            if full_path.exists():
                self.ann_file = full_path
                print(f"  ✓ Found annotation file: {ann_file}")
                break
        
        # If not found in standard locations, search recursively
        if self.ann_file is None:
            print(f"  Searching for annotation files in {self.annotations_dir}...")
            for root, dirs, files in os.walk(self.annotations_dir):
                for file in files:
                    if 'train' in file.lower() and file.endswith('.json'):
                        self.ann_file = Path(root) / file
                        print(f"  ✓ Found annotation file: {self.ann_file}")
                        break
                if self.ann_file:
                    break
        
        if self.ann_file is None:
            raise ValueError(f"Cannot find annotation file in {self.annotations_dir}")
        
        print(f"  Images directory: {self.images_dir}")
        print(f"  Annotation file: {self.ann_file}")
    
    def calculate_tile_grid(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate optimal tile size and grid dimensions.
        
        Args:
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            (tile_width, tile_height, n_cols, n_rows)
        """
        # Calculate grid dimensions (approximate square root)
        n_cols = int(np.sqrt(self.tiles_per_image))
        n_rows = int(np.ceil(self.tiles_per_image / n_cols))
        
        # Calculate tile size with overlap
        tile_width = (img_width + (n_cols - 1) * self.tile_overlap) // n_cols
        tile_height = (img_height + (n_rows - 1) * self.tile_overlap) // n_rows
        
        return tile_width, tile_height, n_cols, n_rows
    
    def get_tile_coordinates(
        self,
        img_width: int,
        img_height: int,
        tile_width: int,
        tile_height: int,
        n_cols: int,
        n_rows: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate coordinates for all tiles.
        
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        tiles = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate tile position with overlap
                x_min = max(0, col * tile_width - col * self.tile_overlap)
                y_min = max(0, row * tile_height - row * self.tile_overlap)
                x_max = min(img_width, x_min + tile_width)
                y_max = min(img_height, y_min + tile_height)
                
                # Add all tiles (no minimum size filter)
                # Even small tiles are useful for boundary regions
                if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                    tiles.append((x_min, y_min, x_max, y_max))
        
        return tiles
    
    def remap_annotation_to_tile(
        self,
        annotation: Dict,
        tile_coords: Tuple[int, int, int, int]
    ) -> Dict:
        """
        Remap annotation coordinates to tile-local coordinates.
        
        Args:
            annotation: COCO annotation dict
            tile_coords: (x_min, y_min, x_max, y_max) of tile
            
        Returns:
            Remapped annotation or None if outside tile
        """
        x_min, y_min, x_max, y_max = tile_coords
        
        # Get bounding box
        bbox = annotation['bbox']  # [x, y, width, height]
        obj_x, obj_y, obj_w, obj_h = bbox
        
        # Check if object intersects with tile
        obj_x_max = obj_x + obj_w
        obj_y_max = obj_y + obj_h
        
        # Calculate intersection
        inter_x_min = max(x_min, obj_x)
        inter_y_min = max(y_min, obj_y)
        inter_x_max = min(x_max, obj_x_max)
        inter_y_max = min(y_max, obj_y_max)
        
        # No intersection
        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return None
        
        # Calculate intersection area ratio
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        obj_area = obj_w * obj_h
        
        # Keep only if >30% of object is in tile
        if inter_area / obj_area < 0.3:
            return None
        
        # Remap coordinates to tile space
        new_annotation = annotation.copy()
        
        # Remap bbox
        new_x = inter_x_min - x_min
        new_y = inter_y_min - y_min
        new_w = inter_x_max - inter_x_min
        new_h = inter_y_max - inter_y_min
        new_annotation['bbox'] = [new_x, new_y, new_w, new_h]
        
        # Remap segmentation
        if 'segmentation' in annotation:
            new_segmentation = []
            for polygon in annotation['segmentation']:
                # Polygon is list of [x1, y1, x2, y2, ...]
                new_polygon = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i] - x_min
                    py = polygon[i + 1] - y_min
                    
                    # Clip to tile boundaries
                    px = max(0, min(x_max - x_min, px))
                    py = max(0, min(y_max - y_min, py))
                    
                    new_polygon.extend([px, py])
                
                if len(new_polygon) >= 6:  # Valid polygon needs at least 3 points
                    new_segmentation.append(new_polygon)
            
            if new_segmentation:
                new_annotation['segmentation'] = new_segmentation
            else:
                return None  # Invalid segmentation after remapping
        
        # Update area
        new_annotation['area'] = new_w * new_h
        
        return new_annotation
    
    def process_image(
        self,
        img_info: Dict,
        annotations: List[Dict],
        coco: COCO,
        img_counter: Dict[str, int],
        split: str,
        debug: bool = False
    ) -> List[Dict]:
        """
        Process single image: create tiles and remap annotations.
        
        Returns:
            List of new image info dicts for COCO format
        """
        # Find the actual image file
        img_filename = img_info['file_name']
        
        # Try different possible paths
        possible_paths = [
            self.images_dir / img_filename,
            self.images_dir / Path(img_filename).name,
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            if debug:
                print(f"  ⚠ Warning: Image not found: {img_filename}, skipping...")
            return []
        
        # Load original image
        try:
            img = Image.open(img_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            if debug:
                print(f"  ⚠ Warning: Failed to load {img_path}: {e}, skipping...")
            return []
        
        img_width, img_height = img.size
        
        if debug:
            print(f"\n  Processing: {img_filename}")
            print(f"    Image size: {img_width}x{img_height}")
            print(f"    Annotations: {len(annotations)}")
            if len(annotations) > 0:
                print(f"    First annotation bbox: {annotations[0]['bbox']}")
                print(f"    First annotation area: {annotations[0].get('area', 'N/A')}")
                if 'segmentation' in annotations[0]:
                    print(f"    First annotation has segmentation: {type(annotations[0]['segmentation'])}")
        
        # Calculate tiling
        tile_w, tile_h, n_cols, n_rows = self.calculate_tile_grid(img_width, img_height)
        tile_coords_list = self.get_tile_coordinates(
            img_width, img_height, tile_w, tile_h, n_cols, n_rows
        )
        
        if debug:
            print(f"    Tile grid: {n_cols}x{n_rows} ({len(tile_coords_list)} tiles)")
            print(f"    Tile size: {tile_w}x{tile_h}")
            if len(tile_coords_list) > 0:
                print(f"    First tile coords: {tile_coords_list[0]}")
        
        new_images = []
        total_anns_mapped = 0
        
        for tile_idx, tile_coords in enumerate(tile_coords_list):
            x_min, y_min, x_max, y_max = tile_coords
            
            # Crop tile
            tile_img = img.crop((x_min, y_min, x_max, y_max))
            
            # Generate new image info
            img_counter[split] += 1
            new_img_id = img_counter[split]
            
            # Save tile
            original_name = Path(img_info['file_name']).stem
            tile_filename = f"{original_name}_tile_{tile_idx:02d}.png"
            
            output_img_dir = self.output_dir / split / 'images'
            output_img_dir.mkdir(parents=True, exist_ok=True)
            
            tile_img.save(output_img_dir / tile_filename)
            
            # Remap annotations for this tile
            tile_annotations = []
            ann_id_counter = new_img_id * 10000
            
            for ann in annotations:
                remapped_ann = self.remap_annotation_to_tile(ann, tile_coords)
                if remapped_ann is not None:
                    # Assign new IDs
                    ann_id_counter += 1
                    remapped_ann['id'] = ann_id_counter
                    remapped_ann['image_id'] = new_img_id
                    tile_annotations.append(remapped_ann)
            
            total_anns_mapped += len(tile_annotations)
            
            # Save ALL tiles (even if no annotations)
            # Create new image info
            new_img_info = {
                'id': new_img_id,
                'file_name': tile_filename,
                'width': x_max - x_min,
                'height': y_max - y_min,
                'original_image_id': img_info['id'],
                'tile_index': tile_idx,
                'tile_coords': tile_coords,
                'annotations': tile_annotations
            }
            
            new_images.append(new_img_info)
        
        if debug:
            print(f"    Result: {len(new_images)} tiles kept, {total_anns_mapped} annotations remapped")
        
        return new_images
    
    def preprocess(self):
        """
        Main preprocessing pipeline.
        """
        print("=" * 80)
        print("LIVECell Dataset Preprocessing")
        print("=" * 80)
        
        # Load original dataset
        print(f"\n1. Loading original dataset from {self.ann_file}")
        coco_train = COCO(self.ann_file)
        
        print(f"   Total images in COCO: {len(coco_train.imgs)}")
        print(f"   Total annotations: {len(coco_train.anns)}")
        
        # Select random subset of images that actually exist
        print(f"\n2. Finding images that exist on disk...")
        existing_img_ids = []
        
        for img_id in list(coco_train.imgs.keys()):
            img_info = coco_train.imgs[img_id]
            img_filename = img_info['file_name']
            
            # Try to find the image
            possible_paths = [
                self.images_dir / img_filename,
                self.images_dir / Path(img_filename).name,
            ]
            
            for path in possible_paths:
                if path.exists():
                    existing_img_ids.append(img_id)
                    break
        
        print(f"   Found {len(existing_img_ids)} images on disk")
        
        if len(existing_img_ids) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Select subset
        num_to_select = min(self.num_images, len(existing_img_ids))
        selected_img_ids = random.sample(existing_img_ids, num_to_select)
        print(f"   Selected {len(selected_img_ids)} images for processing")
        
        # Split into train/val/test
        print(f"\n3. Splitting into train/val/test ({self.splits})...")
        n_train = int(len(selected_img_ids) * self.splits[0])
        n_val = int(len(selected_img_ids) * self.splits[1])
        
        random.shuffle(selected_img_ids)
        train_ids = selected_img_ids[:n_train]
        val_ids = selected_img_ids[n_train:n_train + n_val]
        test_ids = selected_img_ids[n_train + n_val:]
        
        print(f"   Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        
        # Process each split
        splits_data = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        img_counter = {'train': 0, 'val': 0, 'test': 0}
        
        for split_name, img_ids in splits_data.items():
            print(f"\n4. Processing {split_name} split...")
            
            all_new_images = []
            all_new_annotations = []
            
            # Debug first image
            if len(img_ids) > 0:
                print(f"  [DEBUG] Processing first image with details...")
                img_info = coco_train.imgs[img_ids[0]]
                ann_ids = coco_train.getAnnIds(imgIds=img_ids[0])
                annotations = coco_train.loadAnns(ann_ids)
                
                new_images = self.process_image(
                    img_info, annotations, coco_train, img_counter, split_name, debug=True
                )
                
                for new_img in new_images:
                    all_new_images.append({
                        'id': new_img['id'],
                        'file_name': new_img['file_name'],
                        'width': new_img['width'],
                        'height': new_img['height']
                    })
                    all_new_annotations.extend(new_img['annotations'])
            
            # Process remaining images
            for img_id in tqdm(img_ids[1:], desc=f"  Tiling {split_name} images"):
                img_info = coco_train.imgs[img_id]
                ann_ids = coco_train.getAnnIds(imgIds=img_id)
                annotations = coco_train.loadAnns(ann_ids)
                
                # Process image into tiles
                new_images = self.process_image(
                    img_info, annotations, coco_train, img_counter, split_name, debug=False
                )
                
                # Collect all data
                for new_img in new_images:
                    all_new_images.append({
                        'id': new_img['id'],
                        'file_name': new_img['file_name'],
                        'width': new_img['width'],
                        'height': new_img['height']
                    })
                    all_new_annotations.extend(new_img['annotations'])
            
            # Create COCO format JSON
            coco_output = {
                'images': all_new_images,
                'annotations': all_new_annotations,
                'categories': coco_train.dataset['categories']
            }
            
            # Save annotations
            ann_output_dir = self.output_dir / 'annotations'
            ann_output_dir.mkdir(parents=True, exist_ok=True)
            
            ann_output_path = ann_output_dir / f'livecell_coco_{split_name}.json'
            with open(ann_output_path, 'w') as f:
                json.dump(coco_output, f)
            
            print(f"   ✓ {split_name}: {len(all_new_images)} tiles, "
                  f"{len(all_new_annotations)} annotations")
        
        # Save preprocessing info
        print("\n5. Saving preprocessing metadata...")
        metadata = {
            'source_dir': str(self.source_dir),
            'images_dir': str(self.images_dir),
            'annotation_file': str(self.ann_file),
            'num_original_images': self.num_images,
            'tiles_per_image': self.tiles_per_image,
            'tile_overlap': self.tile_overlap,
            'splits': {
                'train': len(train_ids),
                'val': len(val_ids),
                'test': len(test_ids)
            },
            'tiles_created': {
                'train': img_counter['train'],
                'val': img_counter['val'],
                'test': img_counter['test']
            },
            'random_seed': self.random_seed,
            'original_image_ids': {
                'train': train_ids,
                'val': val_ids,
                'test': test_ids
            }
        }
        
        with open(self.output_dir / 'preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ Preprocessing complete!")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Total tiles created: {img_counter['train'] + img_counter['val'] + img_counter['test']}")
        print(f"  Train: {img_counter['train']}")
        print(f"  Val:   {img_counter['val']}")
        print(f"  Test:  {img_counter['test']}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess LIVECell dataset')
    parser.add_argument('--source_dir', type=str, default='data',
                        help='Original LIVECell data directory')
    parser.add_argument('--output_dir', type=str, default='data_split',
                        help='Output directory for preprocessed data')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to select')
    parser.add_argument('--tiles_per_image', type=int, default=20,
                        help='Approximate number of tiles per image')
    parser.add_argument('--tile_overlap', type=int, default=50,
                        help='Overlap between tiles in pixels')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    preprocessor = LIVECellPreprocessor(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        tiles_per_image=args.tiles_per_image,
        tile_overlap=args.tile_overlap,
        random_seed=args.seed
    )
    
    preprocessor.preprocess()