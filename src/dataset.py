"""
LIVECell Tiled Dataset Loader with Data Augmentation
Works with preprocessed/tiled data and includes comprehensive augmentations
"""

import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import cv2


class LIVECellTiledDataset(Dataset):
    """
    Dataset for tiled LIVECell images with augmentation support.
    """
    
    def __init__(self, root_dir, split='train', augment=True):
        """
        Args:
            root_dir: Root directory containing preprocessed data
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Paths
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.ann_file = os.path.join(root_dir, 'annotations', f'livecell_coco_{split}.json')
        
        # Load COCO annotations
        print(f"Loading {split} annotations from {self.ann_file}")
        self.coco = COCO(self.ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        
        print(f"Loaded {len(self.img_ids)} tiled images")
        
        # Setup augmentations
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """
        Define augmentation pipeline using Albumentations.
        """
        if self.augment:
            # Training augmentations
            return A.Compose([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Elastic deformation (important for cells)
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
                
                # Slight rotations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                
                # Intensity transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                
                # Blur
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['labels'],
                min_visibility=0.3
            ))
        else:
            # Validation/test: only normalization
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco',
                label_fields=['labels']
            ))
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Load and augment image with annotations.
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare annotations for Albumentations
        boxes = []
        labels = []
        masks = []
        areas = []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            # Bounding box
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:  # Valid bbox
                boxes.append([x, y, w, h])  # COCO format
                labels.append(ann['category_id'])
                
                # Mask
                mask = self.coco.annToMask(ann)
                masks.append(mask)
                
                # Area
                areas.append(ann['area'])
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
            areas = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.array(masks, dtype=np.uint8)
            areas = np.array(areas, dtype=np.float32)
        
        # Apply augmentations
        if self.augment and len(boxes) > 0:
            # Combine masks for Albumentations
            mask_combined = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for i, mask in enumerate(masks):
                mask_combined[mask > 0] = i + 1
            
            try:
                transformed = self.transform(
                    image=image,
                    mask=mask_combined,
                    bboxes=boxes,
                    labels=labels
                )
                
                image = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['labels'], dtype=np.int64)
                
                # Reconstruct individual masks
                mask_combined = transformed['mask']
                new_masks = []
                for i in range(len(labels)):
                    new_masks.append((mask_combined == (i + 1)).astype(np.uint8))
                masks = np.array(new_masks, dtype=np.uint8)
                
            except Exception as e:
                # Fallback: apply only normalization
                print(f"Augmentation failed for image {img_id}: {e}")
                fallback_transform = A.Compose([
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2()
                ])
                image = fallback_transform(image=image)['image']
        else:
            # No augmentation
            image = self.transform(image=image, bboxes=[], labels=[])['image']
        
        # Convert boxes from COCO format [x, y, w, h] to [x1, y1, x2, y2]
        if len(boxes) > 0:
            boxes_xyxy = boxes.copy()
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes = boxes_xyxy
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        # Create target
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        return image, target


def collate_fn(batch):
    """
    Custom collate function for variable-size annotations.
    """
    return tuple(zip(*batch))


def get_dataloaders(
    root_dir,
    batch_size=4,
    num_workers=4,
    augment_train=True
):
    """
    Create dataloaders for all splits.
    
    Args:
        root_dir: Root directory with preprocessed data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment_train: Whether to augment training data
        
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = LIVECellTiledDataset(
                root_dir=root_dir,
                split=split,
                augment=(augment_train and split == 'train')
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            print(f"✓ {split:5s} dataloader: {len(dataset):4d} images, "
                  f"{len(dataloader):3d} batches")
            
        except Exception as e:
            print(f"✗ Failed to create {split} dataloader: {e}")
    
    return dataloaders


if __name__ == "__main__":
    """
    Test the tiled dataset loader
    """
    import sys
    
    data_dir = 'data'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    print("=" * 80)
    print("Testing LIVECell Tiled Dataset")
    print("=" * 80)
    print(f"Data directory: {data_dir}\n")
    
    # Test dataset
    print("1. Testing dataset loading...")
    print("-" * 80)
    try:
        train_dataset = LIVECellTiledDataset(data_dir, split='train', augment=True)
        print(f"✓ Dataset loaded: {len(train_dataset)} tiles\n")
        
        # Load sample
        img, target = train_dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Num instances: {len(target['boxes'])}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Masks shape: {target['masks'].shape}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Test dataloaders
    print("2. Testing dataloaders...")
    print("-" * 80)
    try:
        dataloaders = get_dataloaders(
            data_dir,
            batch_size=2,
            num_workers=0,
            augment_train=True
        )
        print()
        
        # Test batch loading
        images, targets = next(iter(dataloaders['train']))
        print(f"Batch size: {len(images)}")
        print(f"Image 0 shape: {images[0].shape}")
        print(f"Image 0 instances: {len(targets[0]['boxes'])}")
        print(f"Image 1 instances: {len(targets[1]['boxes'])}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)