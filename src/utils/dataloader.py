"""
LIVECell Dataset Loader for Instance Segmentation
Handles .tif images and COCO format annotations
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as T


class LIVECellDataset(Dataset):
    """
    Dataset class for LIVECell instance segmentation.
    Loads .tif images and COCO format annotations.
    """
    
    def __init__(self, root_dir, split='train', transforms=None):
        """
        Args:
            root_dir (str): Root directory containing data folder (e.g., 'data')
            split (str): 'train', 'val', or 'test'
            transforms: Optional transformations to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # Paths to images and annotations
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.ann_file = os.path.join(root_dir, 'annotations', f'livecell_coco_{split}.json')
        
        # Verify paths exist
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.ann_file):
            raise ValueError(f"Annotation file not found: {self.ann_file}")
        
        # Load COCO annotations
        print(f"Loading {split} annotations from {self.ann_file}")
        self.coco = COCO(self.ann_file)
        
        # Filter image IDs to only include those with actual image files
        all_img_ids = list(self.coco.imgs.keys())
        self.img_ids = []
        
        print(f"Filtering images that exist in {self.img_dir}...")
        for img_id in all_img_ids:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.img_ids.append(img_id)
        
        print(f"Found {len(self.img_ids)} images (out of {len(all_img_ids)} in annotations)")
        
        if len(self.img_ids) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Load image and target annotations.
        
        Returns:
            img (Tensor): Image tensor of shape (3, H, W)
            target (dict): Dictionary containing:
                - boxes (Tensor[N, 4]): Bounding boxes in [x_min, y_min, x_max, y_max] format
                - labels (Tensor[N]): Class labels
                - masks (Tensor[N, H, W]): Segmentation masks
                - image_id (Tensor[1]): Image ID
                - area (Tensor[N]): Area of each instance
                - iscrowd (Tensor[N]): Is crowd annotation
        """
        # Get image ID
        img_id = self.img_ids[idx]
        
        # Load image info
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load .tif image and convert to RGB
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare target components
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Skip invalid annotations
            if ann.get('iscrowd', 0):
                continue
                
            # Bounding box: COCO format is [x, y, width, height]
            # Convert to [x_min, y_min, x_max, y_max]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Category ID (LIVECell has single class for cells)
            labels.append(ann['category_id'])
            
            # Segmentation mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            
            # Area
            areas.append(ann['area'])
            
            # Is crowd (already filtered above, but keep for completeness)
            iscrowd.append(0)
        
        # Convert to tensors
        num_objs = len(boxes)
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle images with no valid annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Convert PIL image to tensor
        img = T.ToTensor()(img)
        
        # Apply transforms if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Since each image can have a different number of instances,
    we cannot stack them into a single tensor. Instead, we return
    lists of images and targets.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        tuple: (images, targets) where both are lists
    """
    return tuple(zip(*batch))


def get_dataloader(root_dir, split='train', batch_size=4, num_workers=4, shuffle=None):
    """
    Create a single dataloader for the specified split.
    
    Args:
        root_dir (str): Root directory containing data
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle data. If None, shuffles only for train split
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Create dataset
    dataset = LIVECellDataset(root_dir, split=split)
    
    # Determine shuffle behavior
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


def get_dataloaders(root_dir, batch_size=4, num_workers=4):
    """
    Create dataloaders for all splits (train, val, test).
    
    Args:
        root_dir (str): Root directory containing data folder
        batch_size (int): Batch size for all splits
        num_workers (int): Number of worker processes
        
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataloaders[split] = get_dataloader(
                root_dir=root_dir,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers
            )
            print(f"✓ Created {split} dataloader: {len(dataloaders[split].dataset)} images")
        except Exception as e:
            print(f"✗ Failed to create {split} dataloader: {e}")
    
    return dataloaders


if __name__ == "__main__":
    """
    Test the dataloader functionality
    """
    import sys
    
    # Default data directory
    data_dir = 'data'
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    print("=" * 80)
    print("Testing LIVECell DataLoader")
    print("=" * 80)
    print(f"Data directory: {data_dir}\n")
    
    # Test single dataset
    print("1. Testing single dataset (train)...")
    print("-" * 80)
    try:
        train_dataset = LIVECellDataset(data_dir, split='train')
        print(f"✓ Dataset loaded: {len(train_dataset)} images\n")
        
        # Load one sample
        print("2. Testing single sample loading...")
        print("-" * 80)
        img, target = train_dataset[0]
        
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Number of instances: {len(target['boxes'])}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Masks shape: {target['masks'].shape}")
        print(f"Labels: {target['labels'][:5]}..." if len(target['labels']) > 5 else f"Labels: {target['labels']}")
        print(f"Image ID: {target['image_id'].item()}\n")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        sys.exit(1)
    
    # Test dataloader
    print("3. Testing DataLoader with batching...")
    print("-" * 80)
    try:
        train_loader = get_dataloader(data_dir, split='train', batch_size=2, num_workers=0)
        
        # Get one batch
        images, targets = next(iter(train_loader))
        
        print(f"Batch size: {len(images)}")
        print(f"First image shape: {images[0].shape}")
        print(f"Second image shape: {images[1].shape}")
        print(f"First image instances: {len(targets[0]['boxes'])}")
        print(f"Second image instances: {len(targets[1]['boxes'])}\n")
        
    except Exception as e:
        print(f"✗ Error with DataLoader: {e}")
        sys.exit(1)
    
    # Test all dataloaders
    print("4. Testing all splits...")
    print("-" * 80)
    try:
        dataloaders = get_dataloaders(data_dir, batch_size=2, num_workers=0)
        print()
        
        for split, loader in dataloaders.items():
            print(f"{split:5s}: {len(loader.dataset):5d} images, {len(loader):4d} batches")
        
    except Exception as e:
        print(f"✗ Error creating all dataloaders: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)