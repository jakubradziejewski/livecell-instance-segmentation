import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou
import argparse
import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.append('src')
from custom_maskrcnn import get_custom_model, count_parameters
from dataset import get_dataloaders


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch with training dynamics tracking."""
    model.train()
    
    total_loss = 0
    loss_rpn_cls = 0
    loss_box_cls = 0
    loss_box_reg = 0
    loss_mask = 0
    
    gradient_norms = []
    learning_rates = []
    memory_usages = []
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = torch.stack([img.to(device) for img in images])  
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norms.append(total_norm)
        
        optimizer.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            memory_usages.append(memory_mb)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            memory_usages.append(0.0)
        
        total_loss += losses.item()
        loss_rpn_cls += loss_dict.get('loss_rpn_cls', torch.tensor(0.0)).item()
        loss_box_cls += loss_dict.get('loss_box_cls', torch.tensor(0.0)).item()
        loss_box_reg += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
        loss_mask += loss_dict.get('loss_mask', torch.tensor(0.0)).item()
        
        progress_bar.set_postfix({
            'loss': losses.item(),
            'avg_loss': total_loss / (batch_idx + 1),
            'grad_norm': f'{total_norm:.2f}',
            'lr': f'{current_lr:.6f}',
            'mem_mb': f'{memory_usages[-1]:.0f}'
        })
        
        del images, targets, loss_dict, losses
        torch.cuda.empty_cache()
    
    n = len(dataloader)
    metrics = {
        'total_loss': total_loss / n,
        'loss_rpn_cls': loss_rpn_cls / n,
        'loss_box_cls': loss_box_cls / n,
        'loss_box_reg': loss_box_reg / n,
        'loss_mask': loss_mask / n,
        
        'gradient_norm_mean': np.mean(gradient_norms),
        'gradient_norm_max': np.max(gradient_norms),
        'gradient_norm_min': np.min(gradient_norms),
        'gradient_norm_std': np.std(gradient_norms),
        'learning_rate': learning_rates[-1],
        'memory_usage_mean_mb': np.mean(memory_usages),
        'memory_usage_max_mb': np.max(memory_usages),
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device, iou_threshold=0.5):
    """Evaluate model with IoU, precision, recall metrics."""
    model.eval()
    
    all_ious = []
    all_precisions = []
    all_recalls = []
    all_confidences = []
    total_gt_instances = 0
    total_pred_instances = 0
    total_true_positives = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for images, targets in progress_bar:
        images_device = torch.stack([img.to(device) for img in images])
        predictions = model(images_device)
        
        for pred, target in zip(predictions, targets):
            gt_boxes = target['boxes']
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            
            if len(pred_scores) > 0:
                all_confidences.extend(pred_scores.tolist())
            
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            
            total_gt_instances += len(gt_boxes)
            total_pred_instances += len(pred_boxes)
            
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            
            if len(iou_matrix) > 0:
                max_ious, max_indices = iou_matrix.max(dim=1)
                all_ious.extend(max_ious.tolist())
                
                true_positives = (max_ious > iou_threshold).sum().item()
                total_true_positives += true_positives
                
                precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
                
                all_precisions.append(precision)
                all_recalls.append(recall)
        
        del images_device, predictions
        torch.cuda.empty_cache()
    
    metrics = {
        'mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'f1_score': 2 * np.mean(all_precisions) * np.mean(all_recalls) / (np.mean(all_precisions) + np.mean(all_recalls)) if (all_precisions and all_recalls and (np.mean(all_precisions) + np.mean(all_recalls)) > 0) else 0.0,
        'mean_confidence': np.mean(all_confidences) if all_confidences else 0.0,
        'total_gt_instances': total_gt_instances,
        'total_pred_instances': total_pred_instances,
        'total_true_positives': total_true_positives,
    }
    
    return metrics


def save_training_plot(train_losses, val_metrics, save_path):
    """Save training and validation metrics plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    val_ious = [m['mean_iou'] for m in val_metrics]
    val_f1s = [m['f1_score'] for m in val_metrics]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, val_ious, 'g-', label='Val IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Validation IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(epochs, val_f1s, 'r-', label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Validation F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Custom Mask R-CNN')
    parser.add_argument('--model', type=str, default='custom', help='Model type (custom)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='livecell-instance-segmentation', help='W&B project name')
    args = parser.parse_args()
    
    print(f"Training {args.model.upper()} Model with Training Dynamics Tracking")
    
    data_dir = 'data_split'
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  W&B logging: {args.use_wandb}")
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model}_lr{args.lr}_bs{args.batch_size}_ep{args.num_epochs}",
            config={
                "model_type": args.model,
                "architecture": "Custom MaskRCNN with CBAM",
                "backbone": "ResNet-18",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "optimizer": "AdamW",
                "weight_decay": 0.0001,
                "scheduler": "StepLR",
                "dataset": "LIVECell",
            }
        )
        print("W&B initialized")
    
    print("\nLoading datasets...")
    dataloaders = get_dataloaders(
        root_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")
    print(f"  Test:  {len(test_loader.dataset)} images")
    
    print(f"\nCreating model...")
    model = get_custom_model(num_classes=num_classes)
    model.to(device)
    
    param_info = count_parameters(model)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters:      {param_info['total']:,}")
    print(f"  Backbone (ResNet-18):  {param_info['backbone']:,} ({100-param_info['custom_percentage']:.1f}%)")
    print(f"  Custom layers:")
    print(f"    - FPN:               {param_info.get('fpn', 0):,}")
    print(f"    - RPN:               {param_info.get('rpn', 0):,}")
    print(f"    - CBAM Attention:    {param_info.get('cbam', 0):,}")
    print(f"    - ROI Align:       {param_info.get('roi_align', 0):,}")
    print(f"    - Box Head:          {param_info.get('box_head', 0):,}")
    print(f"    - Mask Head:         {param_info.get('mask_head', 0):,}")
    print(f"  Total custom:          {param_info['custom']:,} ({param_info['custom_percentage']:.1f}%)")
    print(f"  Custom >50%:           {'YES' if param_info['custom_percentage'] > 50 else 'NO'}")
    print(f"  Memory size:           {param_info['memory_mb']:.2f} MB")
    
    if args.use_wandb:
        wandb.config.update({
            "total_params": param_info['total'],
            "custom_params": param_info['custom'],
            "cbam_params": param_info.get('cbam', 0),
            "custom_percentage": param_info['custom_percentage'],
            "model_memory_mb": param_info['memory_mb'],
        })
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    print("Starting Training with Training Dynamics Tracking")
    print("\nTracking:")
    print("  1. Learning Rate - optimizer step size")
    print("  2. Memory Usage - GPU memory consumption")
    print("  3. Gradient Norm - gradient magnitude")
    
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch} Training:")
        print(f"  Loss: {train_metrics['total_loss']:.4f}")
        print(f"    RPN Cls:  {train_metrics['loss_rpn_cls']:.4f}")
        print(f"    Box Cls:  {train_metrics['loss_box_cls']:.4f}")
        print(f"    Box Reg:  {train_metrics['loss_box_reg']:.4f}")
        print(f"    Mask:     {train_metrics['loss_mask']:.4f}")
        
        print(f"\n  Training Dynamics:")
        print(f"    Gradient Norm:  {train_metrics['gradient_norm_mean']:.4f} "
              f"(min: {train_metrics['gradient_norm_min']:.4f}, "
              f"max: {train_metrics['gradient_norm_max']:.4f})")
        print(f"    Learning Rate:  {train_metrics['learning_rate']:.6f}")
        print(f"    Memory Usage:   {train_metrics['memory_usage_mean_mb']:.1f} MB "
              f"(max: {train_metrics['memory_usage_max_mb']:.1f} MB)")
        
        train_losses.append(train_metrics['total_loss'])
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/total_loss": train_metrics['total_loss'],
                "train/rpn_cls_loss": train_metrics['loss_rpn_cls'],
                "train/box_cls_loss": train_metrics['loss_box_cls'],
                "train/box_reg_loss": train_metrics['loss_box_reg'],
                "train/mask_loss": train_metrics['loss_mask'],
                "dynamics/gradient_norm_mean": train_metrics['gradient_norm_mean'],
                "dynamics/gradient_norm_max": train_metrics['gradient_norm_max'],
                "dynamics/learning_rate": train_metrics['learning_rate'],
                "dynamics/memory_usage_mb": train_metrics['memory_usage_mean_mb'],
            })
        
        val_metrics = evaluate(model, val_loader, device)
        val_metrics_history.append(val_metrics)
        
        print(f"\n  Validation:")
        print(f"    IoU:       {val_metrics['mean_iou']:.4f}")
        print(f"    Precision: {val_metrics['mean_precision']:.4f}")
        print(f"    Recall:    {val_metrics['mean_recall']:.4f}")
        print(f"    F1 Score:  {val_metrics['f1_score']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "val/mean_iou": val_metrics['mean_iou'],
                "val/precision": val_metrics['mean_precision'],
                "val/recall": val_metrics['mean_recall'],
                "val/f1_score": val_metrics['f1_score'],
            })
        
        scheduler.step()
    
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{args.model}_maskrcnn_{args.num_epochs}epochs.pth'
    
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'param_info': param_info,
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    
    plot_path = f'outputs/{args.model}_training_plot.png'
    save_training_plot(train_losses, val_metrics_history, plot_path)
    
    print("\nTesting...")
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  IoU:       {test_metrics['mean_iou']:.4f}")
    print(f"  Precision: {test_metrics['mean_precision']:.4f}")
    print(f"  Recall:    {test_metrics['mean_recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1_score']:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "test/mean_iou": test_metrics['mean_iou'],
            "test/precision": test_metrics['mean_precision'],
            "test/recall": test_metrics['mean_recall'],
            "test/f1_score": test_metrics['f1_score'],
        })
        wandb.log({"training_plot": wandb.Image(plot_path)})
        wandb.finish()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()