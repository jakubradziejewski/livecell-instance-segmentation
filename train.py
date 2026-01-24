"""
Main training script for LIVECell Instance Segmentation
"""

import os
import sys
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.dataloader import get_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train instance segmentation model on LIVECell')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Experiment tracking
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for tracking')
    
    return parser.parse_args()


def print_dataset_info(dataloaders):
    """Print information about the loaded datasets."""
    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    
    for split, loader in dataloaders.items():
        dataset = loader.dataset
        num_images = len(dataset)
        num_batches = len(loader)
        batch_size = loader.batch_size
        
        print(f"\n{split.upper()} SET:")
        print(f"  Total images:  {num_images:,}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Num batches:   {num_batches:,}")
        
        # Get sample to show instance statistics
        try:
            sample_img, sample_target = dataset[0]
            print(f"  Sample image:  {sample_img.shape}")
            print(f"  Sample instances: {len(sample_target['boxes'])}")
        except Exception as e:
            print(f"  Could not load sample: {e}")
    
    print("=" * 80)


def test_dataloader_iteration(dataloader, split='train', num_batches=2):
    """Test iterating through the dataloader."""
    print(f"\n{'=' * 80}")
    print(f"TESTING {split.upper()} DATALOADER ITERATION")
    print("=" * 80)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Number of images: {len(images)}")
        
        for img_idx, (img, target) in enumerate(zip(images, targets)):
            num_instances = len(target['boxes'])
            img_id = target['image_id'].item()
            print(f"    Image {img_idx + 1}: shape={img.shape}, instances={num_instances}, id={img_id}")
    
    print(f"\n✓ Successfully iterated through {num_batches} batches")
    print("=" * 80)


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 80)
    print("LIVECELL INSTANCE SEGMENTATION - TRAINING")
    print("=" * 80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg:20s}: {value}")
    print("=" * 80)
    
    # Check CUDA availability
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n✗ CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    
    try:
        dataloaders = get_dataloaders(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        print("\n✓ All dataloaders created successfully")
    except Exception as e:
        print(f"\n✗ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print dataset information
    print_dataset_info(dataloaders)
    
    # Test dataloader iteration
    if 'train' in dataloaders:
        test_dataloader_iteration(dataloaders['train'], split='train', num_batches=2)
    
    if 'val' in dataloaders:
        test_dataloader_iteration(dataloaders['val'], split='val', num_batches=1)
    
    # Placeholder for model training
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print("\n⚠ Model training not yet implemented")
    print("  - Dataloaders are ready")
    print("  - Add model definition and training loop next")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()