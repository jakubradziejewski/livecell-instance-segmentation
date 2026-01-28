import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as T


class LIVECellTiledDataset(Dataset):
    """Dataset for tiled LIVECell microscopy images with COCO annotations."""
    
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.ann_file = os.path.join(root_dir, 'annotations', f'livecell_coco_{split}.json')
        
        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.ann_file):
            raise ValueError(f"Annotation file not found: {self.ann_file}")
        
        print(f"Loading {split} annotations from {self.ann_file}")
        self.coco = COCO(self.ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        print(f"Loaded {len(self.img_ids)} tiled images")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            # Convert COCO bbox [x, y, w, h] to [x_min, y_min, x_max, y_max]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann['area'])
            iscrowd.append(0)
        
        num_objs = len(boxes)
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        img = T.ToTensor()(img)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


def collate_fn(batch):
    """Custom collate for variable-length instance annotations."""
    return tuple(zip(*batch))


def get_dataloader(root_dir, split='train', batch_size=4, num_workers=4, shuffle=None):
    dataset = LIVECellTiledDataset(root_dir, split=split)
    
    if shuffle is None:
        shuffle = (split == 'train')
    
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
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataloaders[split] = get_dataloader(
                root_dir=root_dir,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers
            )
            print(f"Created {split} dataloader: {len(dataloaders[split].dataset)} images")
        except Exception as e:
            print(f"Failed to create {split} dataloader: {e}")
    
    return dataloaders


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_split'
    
    print("=" * 80)
    print("Testing LIVECell Tiled Dataset")
    print("=" * 80)
    print(f"Data directory: {data_dir}\n")
    
    print("1. Testing single dataset (train)...")
    print("-" * 80)
    try:
        train_dataset = LIVECellTiledDataset(data_dir, split='train')
        print(f"Dataset loaded: {len(train_dataset)} images\n")
        
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
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("3. Testing DataLoader with batching...")
    print("-" * 80)
    try:
        train_loader = get_dataloader(data_dir, split='train', batch_size=2, num_workers=0)
        images, targets = next(iter(train_loader))
        
        print(f"Batch size: {len(images)}")
        print(f"First image shape: {images[0].shape}")
        print(f"Second image shape: {images[1].shape}")
        print(f"First image instances: {len(targets[0]['boxes'])}")
        print(f"Second image instances: {len(targets[1]['boxes'])}\n")
        
    except Exception as e:
        print(f"Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("4. Testing all splits...")
    print("-" * 80)
    try:
        dataloaders = get_dataloaders(data_dir, batch_size=2, num_workers=0)
        print()
        
        for split, loader in dataloaders.items():
            print(f"{split:5s}: {len(loader.dataset):5d} images, {len(loader):4d} batches")
        
    except Exception as e:
        print(f"Error creating all dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)