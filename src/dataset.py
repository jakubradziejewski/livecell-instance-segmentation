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

