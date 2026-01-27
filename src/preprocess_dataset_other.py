"""
LIVECell Dataset Preprocessor with Image Tiling (Deterministic Splits)
Logic Update:
- 'num_images_per_split' now represents the TOTAL number of images to process.
- Distribution: 70% Train, 15% Val, 15% Test.
- Selection: Files are selected alphabetically from their respective source folders.
- Tiling: 8x8 mini-tile grid, then 3x3 sliding window creates 36 tiles per image.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse


class LIVECellPreprocessor:
    """
    Preprocesses LIVECell dataset by:
    1. Selecting specific counts of images (70/15/15 split of total)
    2. Sorting alphabetically for deterministic selection
    3. Tiling large images into smaller patches using 8x8 grid + 3x3 sliding window
    4. Remapping annotations to tiles
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        total_images: int = 100,
        tiles_per_image: int = 20,
        tile_overlap: int = 50,
    ):
        """
        Args:
            source_dir: Original LIVECell data directory
            output_dir: Where to save preprocessed data
            total_images: TOTAL number of images to use across all splits
            tiles_per_image: How many tiles to create per image (approx) - NOTE: now fixed at 36
            tile_overlap: Overlap between adjacent tiles in pixels - NOTE: now determined by 3x3 window
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.total_images = total_images
        self.tiles_per_image = tiles_per_image
        self.tile_overlap = tile_overlap

        # Calculate limits per split based on 70/15/15 ratio
        n_train = int(self.total_images * 0.70)
        n_val = int(self.total_images * 0.15)
        # Assign remainder to test to ensure sum equals total_images exactly
        n_test = self.total_images - n_train - n_val

        self.split_limits = {
            "train": n_train,
            "val": n_val,
            "test": n_test
        }
        
        print(f"Target distribution (Total {self.total_images}):")
        print(f"  - Train (70%): {n_train}")
        print(f"  - Val   (15%): {n_val}")
        print(f"  - Test  (15%): {n_test}")

        self._detect_structure()

    def _detect_structure(self):
        """
        Auto-detect LIVECell dataset structure.
        """
        print(f"Detecting dataset structure in {self.source_dir}...")

        self.annotations_dir = self.source_dir / "annotations"

        # Split structure: train/val/test images
        if (self.source_dir / "train" / "images").exists():
            self.images_dirs = {
                "train": self.source_dir / "train" / "images",
                "val": self.source_dir / "val" / "images",
                "test": self.source_dir / "test" / "images",
            }
            print("  ✓ Detected split structure: train/val/test images/")
        # Flat structure: single images/ dir
        elif (self.source_dir / "images").exists():
            flat_dir = self.source_dir / "images"
            self.images_dirs = {
                "train": flat_dir,
                "val": flat_dir,
                "test": flat_dir,
            }
            print("  ✓ Detected flat structure: images/")
        else:
            raise ValueError(
                f"Cannot detect valid LIVECell structure in {self.source_dir}\n"
                f"Expected either:\n"
                f"  - {self.source_dir}/images/\n"
                f"  - {self.source_dir}/train/images/, val/images/, test/images/"
            )

        self.split_ann_files = {
            "train": self.annotations_dir / "livecell_coco_train.json",
            "val": self.annotations_dir / "livecell_coco_val.json",
            "test": self.annotations_dir / "livecell_coco_test.json",
        }

        for split, ann_path in self.split_ann_files.items():
            if not ann_path.exists():
                raise ValueError(f"Missing annotation file for {split}: {ann_path}")

    def calculate_tile_grid(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate 8x8 grid of mini-tiles.
        
        Returns:
            mini_tile_width: Width of each mini-tile
            mini_tile_height: Height of each mini-tile
            n_mini_cols: Number of mini-tile columns (8)
            n_mini_rows: Number of mini-tile rows (8)
        """
        # Fixed 8x8 grid of mini-tiles for 704x520 images
        n_mini_cols = 8
        n_mini_rows = 8

        # Calculate mini-tile dimensions (704/8 = 88, 520/8 = 65)
        mini_tile_width = img_width // n_mini_cols
        mini_tile_height = img_height // n_mini_rows

        return mini_tile_width, mini_tile_height, n_mini_cols, n_mini_rows
    
    def get_tile_coordinates(
        self,
        img_width: int,
        img_height: int,
        mini_tile_width: int,
        mini_tile_height: int,
        n_mini_cols: int,
        n_mini_rows: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate tile coordinates using 3x3 sliding window over 8x8 mini-tile grid.
        
        This creates overlapping tiles where each tile spans 3x3 mini-tiles.
        For an 8x8 grid, this produces 6x6 = 36 tiles total.
        
        Args:
            img_width: Full image width
            img_height: Full image height
            mini_tile_width: Width of one mini-tile
            mini_tile_height: Height of one mini-tile
            n_mini_cols: Number of mini-tile columns (8)
            n_mini_rows: Number of mini-tile rows (8)
            
        Returns:
            List of tile coordinates as (x_min, y_min, x_max, y_max)
        """
        tiles = []
        
        # Sliding window: 3x3 mini-tiles
        window_size = 3
        
        # Calculate how many positions the window can slide
        # For 8 mini-tiles and window of 3: positions 0-2, 1-3, 2-4, 3-5, 4-6, 5-7 = 6 positions
        n_positions_col = n_mini_cols - window_size + 1
        n_positions_row = n_mini_rows - window_size + 1
        
        # Slide window over the grid
        for row_start in range(n_positions_row):
            for col_start in range(n_positions_col):
                # Calculate pixel coordinates for this 3x3 window
                x_min = col_start * mini_tile_width
                y_min = row_start * mini_tile_height
                x_max = (col_start + window_size) * mini_tile_width
                y_max = (row_start + window_size) * mini_tile_height
                
                tiles.append((x_min, y_min, x_max, y_max))
        
        return tiles

    def remap_annotation_to_tile(
        self,
        annotation: Dict,
        tile_coords: Tuple[int, int, int, int],
    ) -> Dict | None:
        x_min, y_min, x_max, y_max = tile_coords

        obj_x, obj_y, obj_w, obj_h = annotation["bbox"]
        obj_x_max = obj_x + obj_w
        obj_y_max = obj_y + obj_h

        inter_x_min = max(x_min, obj_x)
        inter_y_min = max(y_min, obj_y)
        inter_x_max = min(x_max, obj_x_max)
        inter_y_max = min(y_max, obj_y_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return None

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        obj_area = obj_w * obj_h

        if inter_area / obj_area < 0.3:
            return None

        new_annotation = annotation.copy()

        new_x = inter_x_min - x_min
        new_y = inter_y_min - y_min
        new_w = inter_x_max - inter_x_min
        new_h = inter_y_max - inter_y_min
        new_annotation["bbox"] = [new_x, new_y, new_w, new_h]

        if "segmentation" in annotation:
            new_segmentation = []
            for polygon in annotation["segmentation"]:
                new_polygon = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i] - x_min
                    py = polygon[i + 1] - y_min

                    px = max(0, min(x_max - x_min, px))
                    py = max(0, min(y_max - y_min, py))

                    new_polygon.extend([px, py])

                if len(new_polygon) >= 6:
                    new_segmentation.append(new_polygon)

            if new_segmentation:
                new_annotation["segmentation"] = new_segmentation
            else:
                return None

        new_annotation["area"] = new_w * new_h
        return new_annotation

    def process_image(
        self,
        img_info: Dict,
        annotations: List[Dict],
        coco: COCO,
        img_counter: Dict[str, int],
        split: str,
    ) -> List[Dict]:
        img_filename = img_info["file_name"]

        img_dir = self.images_dirs[split]
        possible_paths = [
            img_dir / img_filename,
            img_dir / Path(img_filename).name,
        ]

        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break

        if img_path is None:
            print(f"  ⚠ Image not found: {img_filename}, skipping...")
            return []

        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            print(f"  ⚠ Failed to load {img_path}: {e}, skipping...")
            return []

        img_width, img_height = img.size

        mini_tile_w, mini_tile_h, n_mini_cols, n_mini_rows = self.calculate_tile_grid(img_width, img_height)
        tile_coords_list = self.get_tile_coordinates(
            img_width, img_height, mini_tile_w, mini_tile_h, n_mini_cols, n_mini_rows
        )

        new_images = []

        for tile_idx, tile_coords in enumerate(tile_coords_list):
            x_min, y_min, x_max, y_max = tile_coords

            tile_img = img.crop((x_min, y_min, x_max, y_max))

            img_counter[split] += 1
            new_img_id = img_counter[split]

            original_name = Path(img_info["file_name"]).stem
            tile_filename = f"{original_name}_tile_{tile_idx:02d}.png"

            output_img_dir = self.output_dir / split / "images"
            output_img_dir.mkdir(parents=True, exist_ok=True)

            tile_img.save(output_img_dir / tile_filename)

            tile_annotations = []
            ann_id_counter = new_img_id * 10000

            for ann in annotations:
                remapped_ann = self.remap_annotation_to_tile(ann, tile_coords)
                if remapped_ann is not None:
                    ann_id_counter += 1
                    remapped_ann["id"] = ann_id_counter
                    remapped_ann["image_id"] = new_img_id
                    tile_annotations.append(remapped_ann)

            new_img_info = {
                "id": new_img_id,
                "file_name": tile_filename,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "annotations": tile_annotations,
            }

            new_images.append(new_img_info)

        return new_images

    def preprocess(self):
        print("=" * 80)
        print("LIVECell Dataset Preprocessing (Deterministic + 8x8 Grid + 3x3 Window)")
        print("=" * 80)

        cocos = {}
        for split, ann_path in self.split_ann_files.items():
            coco = COCO(ann_path)
            cocos[split] = coco
            print(f"{split}: Loaded metadata for {len(coco.imgs)} potential images")

        img_counter = {"train": 0, "val": 0, "test": 0}

        for split_name, coco in cocos.items():
            target_limit = self.split_limits[split_name]
            print(f"\nProcessing {split_name} split (Target: {target_limit} images)...")

            if target_limit == 0:
                print(f"  ⚠ Skipping {split_name} as target count is 0")
                continue

            all_new_images = []
            all_new_annotations = []

            # 1. Get all image objects
            all_img_ids = coco.getImgIds()
            all_imgs = coco.loadImgs(all_img_ids)

            # 2. Filter images that exist on disk
            img_dir = self.images_dirs[split_name]
            valid_imgs = []
            
            # Using list comprehension for filtering valid images
            for img in all_imgs:
                img_filename = img["file_name"]
                if img_filename.startswith("A172"):
                    possible_paths = [img_dir / img_filename, img_dir / Path(img_filename).name]
                    if any(p.exists() for p in possible_paths):
                        valid_imgs.append(img)

            # 3. SORT ALPHABETICALLY BY FILE NAME
            # This ensures that selecting the "first X" is always deterministic
            valid_imgs.sort(key=lambda x: x['file_name'])

            # 4. Select top N images based on calculated limit
            selected_imgs = valid_imgs[:target_limit]
            
            print(f"  Selected {len(selected_imgs)} images (sorted alphabetically)")
            print(f"  Will generate ~36 tiles per image (8x8 grid, 3x3 sliding window)")
            if len(selected_imgs) < target_limit:
                 print(f"  ⚠ Warning: Requested {target_limit} but only found {len(selected_imgs)} valid images")

            # 5. Process the selected images
            for img_info in tqdm(selected_imgs, desc=f"  Tiling {split_name}"):
                img_id = img_info['id']
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)

                new_images = self.process_image(
                    img_info, annotations, coco, img_counter, split_name
                )

                for new_img in new_images:
                    all_new_images.append(
                        {
                            "id": new_img["id"],
                            "file_name": new_img["file_name"],
                            "width": new_img["width"],
                            "height": new_img["height"],
                        }
                    )
                    all_new_annotations.extend(new_img["annotations"])

            coco_output = {
                "images": all_new_images,
                "annotations": all_new_annotations,
                "categories": coco.dataset["categories"],
            }

            ann_output_dir = self.output_dir / "annotations"
            ann_output_dir.mkdir(parents=True, exist_ok=True)

            ann_output_path = ann_output_dir / f"livecell_coco_{split_name}.json"
            with open(ann_output_path, "w") as f:
                json.dump(coco_output, f)

            print(
                f"  ✓ {split_name} Complete: {len(all_new_images)} tiles generated from "
                f"{len(selected_imgs)} source images."
            )

        print("\n✓ Preprocessing complete!")
        print(f"Output directory: {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LIVECell dataset (deterministic)")
    parser.add_argument("--source_dir", type=str, default="data", help="Original LIVECell data directory")
    parser.add_argument("--output_dir", type=str, default="data_split", help="Output directory")
    
    # Updated help text to reflect new behavior
    parser.add_argument(
        "--num_images_per_split", 
        type=int, 
        default=100, 
        help="TOTAL number of source images to process across all splits (split 70/15/15)"
    )
    
    parser.add_argument("--tiles_per_image", type=int, default=36, help="Tiles per image (now fixed at 36 with 8x8 grid + 3x3 window)")
    parser.add_argument("--tile_overlap", type=int, default=0, help="Overlap determined by 3x3 sliding window")

    args = parser.parse_args()

    preprocessor = LIVECellPreprocessor(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        total_images=args.num_images_per_split, # Passing as total_images
        tiles_per_image=args.tiles_per_image,
        tile_overlap=args.tile_overlap,
    )

    preprocessor.preprocess()