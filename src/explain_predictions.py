import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import cv2
import sys

sys.path.append('src')
from custom_maskrcnn import get_custom_model
from dataset import LIVECellTiledDataset


class ComprehensivePredictionExplainer:
    """Visualizes complete prediction pipeline: Backbone → CBAM → FPN → RPN → ROI → Heads."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, (list, tuple)):
                    self.activations[name] = [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
                else:
                    self.activations[name] = output.detach()
            return hook
        
        self.hooks.append(self.model.layer1.register_forward_hook(get_activation('layer1')))
        self.hooks.append(self.model.layer2.register_forward_hook(get_activation('layer2')))
        self.hooks.append(self.model.layer3.register_forward_hook(get_activation('layer3')))
        self.hooks.append(self.model.layer4.register_forward_hook(get_activation('layer4')))
        
        self.hooks.append(self.model.cbam1.register_forward_hook(get_activation('cbam1')))
        self.hooks.append(self.model.cbam2.register_forward_hook(get_activation('cbam2')))
        self.hooks.append(self.model.cbam3.register_forward_hook(get_activation('cbam3')))
        self.hooks.append(self.model.cbam4.register_forward_hook(get_activation('cbam4')))
        
        self.hooks.append(self.model.fpn.register_forward_hook(get_activation('fpn')))
        self.hooks.append(self.model.rpn.register_forward_hook(get_activation('rpn')))
        self.hooks.append(self.model.box_head.register_forward_hook(get_activation('box_head')))
        self.hooks.append(self.model.mask_head.register_forward_hook(get_activation('mask_head')))
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def visualize_feature_map(self, feature_map, title="Feature Map"):
        """Visualize feature map by averaging across channels."""
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[0]
            
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
            
        avg_activation = torch.mean(feature_map, dim=0).cpu().numpy()
        avg_activation = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-8)
        
        colored = cv2.applyColorMap(np.uint8(255 * avg_activation), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored
    
    def compute_feature_importance(self, feature_map):
        """Compute feature importance based on activation magnitude."""
        if isinstance(feature_map, (list, tuple)):
            feature_map = feature_map[0]
            
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
        
        importance = torch.mean(torch.abs(feature_map)).item()
        return importance
    
    def explain_prediction_comprehensive(self, image, target, save_path, prediction_idx):
        """Create comprehensive visualization of entire prediction pipeline."""
        self.register_hooks()
        
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(self.device)
            predictions = self.model(image_batch)
            pred = predictions[0]
        
        layer1_feat = self.activations.get('layer1')
        layer2_feat = self.activations.get('layer2')
        layer3_feat = self.activations.get('layer3')
        layer4_feat = self.activations.get('layer4')
        
        cbam1_feat = self.activations.get('cbam1')
        cbam2_feat = self.activations.get('cbam2')
        cbam3_feat = self.activations.get('cbam3')
        cbam4_feat = self.activations.get('cbam4')
        
        fpn_feat = self.activations.get('fpn')
        rpn_output = self.activations.get('rpn')
        box_head_output = self.activations.get('box_head')
        mask_head_output = self.activations.get('mask_head')
        
        importance_scores = {}
        
        if layer1_feat is not None:
            importance_scores['Backbone-L1'] = self.compute_feature_importance(layer1_feat)
        if layer2_feat is not None:
            importance_scores['Backbone-L2'] = self.compute_feature_importance(layer2_feat)
        if layer3_feat is not None:
            importance_scores['Backbone-L3'] = self.compute_feature_importance(layer3_feat)
        if layer4_feat is not None:
            importance_scores['Backbone-L4'] = self.compute_feature_importance(layer4_feat)
            
        if cbam1_feat is not None:
            importance_scores['CBAM-1'] = self.compute_feature_importance(cbam1_feat)
        if cbam2_feat is not None:
            importance_scores['CBAM-2'] = self.compute_feature_importance(cbam2_feat)
        if cbam3_feat is not None:
            importance_scores['CBAM-3'] = self.compute_feature_importance(cbam3_feat)
        if cbam4_feat is not None:
            importance_scores['CBAM-4'] = self.compute_feature_importance(cbam4_feat)
            
        if fpn_feat is not None:
            importance_scores['FPN'] = self.compute_feature_importance(fpn_feat)
            
        if rpn_output is not None:
            if isinstance(rpn_output, (list, tuple)) and len(rpn_output) > 0:
                importance_scores['RPN'] = self.compute_feature_importance(rpn_output[0])
        
        if mask_head_output is not None:
            importance_scores['Mask-Head'] = self.compute_feature_importance(mask_head_output)
        
        total_importance = sum(importance_scores.values())
        importance_percentages = {k: (v / total_importance) * 100 for k, v in importance_scores.items()}
        
        layer1_vis = self.visualize_feature_map(layer1_feat) if layer1_feat is not None else None
        layer4_vis = self.visualize_feature_map(layer4_feat) if layer4_feat is not None else None
        cbam1_vis = self.visualize_feature_map(cbam1_feat) if cbam1_feat is not None else None
        cbam4_vis = self.visualize_feature_map(cbam4_feat) if cbam4_feat is not None else None
        fpn_vis = self.visualize_feature_map(fpn_feat) if fpn_feat is not None else None
        
        self.remove_hooks()
        
        keep = pred['scores'] > 0.5
        pred_boxes = pred['boxes'][keep].cpu()
        pred_scores = pred['scores'][keep].cpu()
        pred_masks = pred['masks'][keep].cpu()
        
        gt_boxes = target['boxes'].cpu()
        
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np_uint8 = (img_np * 255).astype(np.uint8)
        
        fig = plt.figure(figsize=(28, 21))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Input → Backbone
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_np)
        ax.set_title('Step 1: Input Image', fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, -0.12, f'Shape: {image.shape}\nRGB channels', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[0, 1])
        if layer1_vis is not None:
            ax.imshow(cv2.resize(layer1_vis, (img_np.shape[1], img_np.shape[0])))
        importance_pct = importance_percentages.get('Backbone-L1', 0)
        ax.set_title(f'Step 2a: Backbone L1\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Early features\nChannels: 64', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[0, 2])
        if cbam1_vis is not None:
            ax.imshow(cv2.resize(cbam1_vis, (img_np.shape[1], img_np.shape[0])))
        importance_pct = importance_percentages.get('CBAM-1', 0)
        ax.set_title(f'Step 2b: CBAM-1 Attention\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Channel + Spatial\nattention applied', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[0, 3])
        if layer4_vis is not None:
            ax.imshow(cv2.resize(layer4_vis, (img_np.shape[1], img_np.shape[0])))
        importance_pct = importance_percentages.get('Backbone-L4', 0)
        ax.set_title(f'Step 2c: Backbone L4\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'High-level features\nChannels: 512', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        # Row 2: CBAM + FPN + RPN
        ax = fig.add_subplot(gs[1, 0])
        if cbam4_vis is not None:
            ax.imshow(cv2.resize(cbam4_vis, (img_np.shape[1], img_np.shape[0])))
        importance_pct = importance_percentages.get('CBAM-4', 0)
        ax.set_title(f'Step 3a: CBAM-4 Attention\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Channel + Spatial\nattention on L4', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 1])
        if fpn_vis is not None:
            ax.imshow(cv2.resize(fpn_vis, (img_np.shape[1], img_np.shape[0])))
        importance_pct = importance_percentages.get('FPN', 0)
        ax.set_title(f'Step 3b: FPN Features\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Multi-scale fusion\n256 channels', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(img_np)
        
        if rpn_output is not None and isinstance(rpn_output, (list, tuple)):
            cls_scores = rpn_output[0]
            if isinstance(cls_scores, (list, tuple)):
                cls_scores = cls_scores[0]
            
            if cls_scores.dim() == 4:
                objectness = torch.sigmoid(cls_scores[0]).permute(1, 2, 0).reshape(-1).cpu()
                
                from custom_maskrcnn import AnchorGenerator
                anchor_gen = AnchorGenerator(sizes=(32, 64, 128), aspect_ratios=(0.5, 1.0, 2.0))
                feature_h, feature_w = cls_scores.shape[-2:]
                anchors = anchor_gen.generate_anchors((feature_h, feature_w), stride=4, device='cpu')
                
                top_k = min(50, len(objectness))
                top_scores, top_indices = torch.topk(objectness, top_k)
                top_proposals = anchors[top_indices]
                
                for box, score in zip(top_proposals[:50], top_scores[:50]):
                    x1, y1, x2, y2 = box.tolist()
                    color = 'yellow' if score > 0.7 else 'orange'
                    rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=0.6)
                    ax.add_patch(rect)
        
        importance_pct = importance_percentages.get('RPN', 0)
        ax.set_title(f'Step 3c: RPN Proposals\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Region proposals\nTop 50 shown', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[1, 3])
        ax.imshow(img_np)
        
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box.tolist()
            if score > 0.7:
                color = 'lime'
            elif score > 0.6:
                color = 'yellow'
            else:
                color = 'orange'
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{score:.2f}', 
                   color=color, fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        ax.set_title(f'Step 4a: Box Head Output\nDetections: {len(pred_boxes)}', 
                     fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, -0.12, f'Classification + BBox\nNMS applied', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        # Row 3: Masks + Analysis
        ax = fig.add_subplot(gs[2, 0])
        
        if len(pred_masks) > 0:
            masks_np = pred_masks.numpy()
            overlay = img_np_uint8.copy()
            
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(masks_np), 3))
            
            for i, (mask, color) in enumerate(zip(masks_np, colors)):
                mask_binary = (mask > 127).astype(np.uint8)
                colored_mask = np.zeros_like(overlay)
                for c in range(3):
                    colored_mask[:, :, c] = mask_binary * color[c]
                mask_area = mask_binary > 0
                overlay[mask_area] = cv2.addWeighted(
                    overlay[mask_area], 0.5, colored_mask[mask_area], 0.5, 0
                )
            
            ax.imshow(overlay)
        else:
            ax.imshow(img_np)
            ax.text(0.5, 0.5, 'No masks detected', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red', fontweight='bold')
        
        importance_pct = importance_percentages.get('Mask-Head', 0)
        ax.set_title(f'Step 4b: Mask Head Output\nImportance: {importance_pct:.1f}%', 
                     fontsize=14, fontweight='bold', pad=10,
                     color='red' if importance_pct == max(importance_percentages.values()) else 'black')
        ax.text(0.5, -0.12, f'Instance segmentation\n{len(pred_masks)} masks', 
                ha='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        ax = fig.add_subplot(gs[2, 1:3])
        
        sorted_components = sorted(importance_percentages.items(), key=lambda x: x[1], reverse=True)
        components = [c[0] for c in sorted_components]
        importances = [c[1] for c in sorted_components]
        
        colors_bar = ['red' if i == max(importances) else 'steelblue' for i in importances]
        
        bars = ax.barh(components, importances, color=colors_bar)
        ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
        ax.set_title('Component Importance Analysis', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=9)
        
        ax = fig.add_subplot(gs[2, 3])
        ax.axis('off')
        
        from torchvision.ops import box_iou
        
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            max_ious = iou_matrix.max(dim=1)[0]
            
            true_positives = (max_ious > 0.5).sum().item()
            false_positives = len(pred_boxes) - true_positives
            false_negatives = len(gt_boxes) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if len(pred_boxes) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if len(gt_boxes) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            true_positives = 0
            false_positives = len(pred_boxes)
            false_negatives = len(gt_boxes)
            precision = 0
            recall = 0
            f1 = 0
        
        metrics_text = "PERFORMANCE\n"
        metrics_text += "="*30 + "\n\n"
        metrics_text += f"GT Cells:    {len(gt_boxes):3d}\n"
        metrics_text += f"Predictions: {len(pred_boxes):3d}\n"
        metrics_text += f"True Pos:    {true_positives:3d}\n"
        metrics_text += f"False Pos:   {false_positives:3d}\n"
        metrics_text += f"False Neg:   {false_negatives:3d}\n\n"
        metrics_text += f"Precision: {precision:.3f}\n"
        metrics_text += f"Recall:    {recall:.3f}\n"
        metrics_text += f"F1-Score:  {f1:.3f}\n\n"
        
        if f1 > 0.8:
            metrics_text += "EXCELLENT"
        elif f1 > 0.6:
            metrics_text += "GOOD"
        elif f1 > 0.4:
            metrics_text += "FAIR"
        else:
            metrics_text += "POOR"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        most_important = max(importance_percentages.items(), key=lambda x: x[1])
        fig.suptitle(f'Prediction {prediction_idx}: Complete Pipeline Analysis\n' + 
                    f'Most Significant Component: {most_important[0].upper()} ({most_important[1]:.1f}% importance)', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved explanation to {save_path}")
        
        return {
            'n_predictions': len(pred_boxes),
            'n_ground_truth': len(gt_boxes),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': importance_percentages,
            'most_important_component': most_important[0]
        }


def main():
    """Generate comprehensive pipeline explanations for 3 predictions."""
    
    print("="*80)
    print("GENERATING COMPREHENSIVE PIPELINE EXPLANATIONS")
    print("Backbone → CBAM → FPN → RPN → ROI → Box/Mask Heads")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data_split'
    model_path = 'models/custom_maskrcnn_10epochs.pth'
    
    print(f"\nLoading model from {model_path}...")
    model = get_custom_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded")
    
    explainer = ComprehensivePredictionExplainer(model, device)
    print("Explainer initialized")
    
    print(f"\nLoading test dataset...")
    test_dataset = LIVECellTiledDataset(data_dir, split='test')
    print(f"Loaded {len(test_dataset)} test images")
    
    indices = [0, len(test_dataset) // 2, len(test_dataset) - 1]
    

    print("Explaining predictions for 3 test images...\n")
    
    results = []
    
    for i, idx in enumerate(indices, 1):
        print(f"\nProcessing prediction {i}/3 (image index {idx})...")
        
        image, target = test_dataset[idx]
        
        save_path = f'outputs/explanation_{i}.png'
        result = explainer.explain_prediction_comprehensive(
            image=image,
            target=target,
            save_path=save_path,
            prediction_idx=i
        )
        
        results.append(result)
        
        print(f"\n  Component Importance for Prediction {i}:")
        sorted_importance = sorted(result['feature_importance'].items(), 
                                  key=lambda x: x[1], reverse=True)
        for j, (component, importance) in enumerate(sorted_importance[:5], 1):
            marker = "★" if j == 1 else f"{j}."
            print(f"    {marker} {component:15s}: {importance:6.2f}%")

    
    print("\nOverall Statistics:")
    print(f"{'Pred':<6} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Most Important':<15}")
    print("-" * 95)
    
    for i, result in enumerate(results, 1):
        print(f"{'#' + str(i):<6} "
              f"{result['n_ground_truth']:>4} "
              f"{result['n_predictions']:>4} "
              f"{result['true_positives']:>4} "
              f"{result['false_positives']:>4} "
              f"{result['false_negatives']:>4} "
              f"{result['precision']:>6.3f} "
              f"{result['recall']:>6.3f} "
              f"{result['f1_score']:>6.3f} "
              f"{result['most_important_component']:<15}")
    
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1_score'] for r in results])
    
    print("-" * 95)
    print(f"{'Avg':<6} {'':>4} {'':>4} {'':>4} {'':>4} {'':>4} "
          f"{avg_precision:>6.3f} {avg_recall:>6.3f} {avg_f1:>6.3f}")
    

if __name__ == "__main__":
    main()