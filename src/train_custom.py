import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from pathlib import Path

from src.data.dataset import LIVECellDataset, get_transform, collate_fn
from src.models.custom_model import get_model_custom


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass - MaskRCNN automatically returns loss dict
        loss_dict = model(images, targets)
        
        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track losses
        total_loss += losses.item()
        for k, v in loss_dict.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    # Configuration
    config = {
        'batch_size': 2,  # Start small
        'num_epochs': 40,
        'lr': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'num_workers': 4,
    }
    
    # Initialize W&B
    wandb.init(
        project='livecell-instance-segmentation',
        name='model2-custom-attention',
        config=config
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = LIVECellDataset(
        root_dir='data',
        split='train',
        transforms=get_transform('train')
    )
    
    val_dataset = LIVECellDataset(
        root_dir='data',
        split='val',
        transforms=get_transform('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} images, {len(val_loader)} batches")
    
    # Create model
    print("Creating model...")
    model = get_model_custom(num_classes=9)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )
    
    # Training loop
    best_loss = float('inf')
    save_dir = Path('models/model2_custom')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Log to W&B
        log_dict = {
            'epoch': epoch,
            'train/total_loss': train_loss,
            'lr': optimizer.param_groups[0]['lr']
        }
        for k, v in train_components.items():
            log_dict[f'train/{k}'] = v
        
        wandb.log(log_dict)
        
        print(f"\nEpoch {epoch}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}")
        for k, v in train_components.items():
            print(f"    {k}: {v:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
        
        # Step scheduler
        lr_scheduler.step()
    
    wandb.finish()
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()