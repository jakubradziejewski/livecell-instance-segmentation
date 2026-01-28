import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops import RoIAlign, nms, box_iou

from src.components.cbam import CBAM
from src.components.fpn import FPN
from src.components.rpn import RPN
from src.components.mask_head import CustomMaskHead
from src.components.box_head import CustomBoxHead
from src.components.anchor_generator import AnchorGenerator
from src.utils.box_utils import encode_boxes
from src.utils.mask_utils import compute_mask_loss_from_gt
from src.utils.proposal_utils import (
    generate_training_proposals,
    generate_inference_proposals,
    sample_proposals
)

class CustomMaskRCNN(nn.Module):
    """Custom Mask R-CNN with ResNet-18 backbone, CBAM attention, FPN, RPN and anchor generator."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        del resnet

        self.fpn = FPN(in_channels_list=[64, 128, 256, 512], out_channels=256)
        self.rpn = RPN(in_channels=256, num_anchors=9)

        self.roi_align = RoIAlign(
            output_size=(7, 7), spatial_scale=1.0 / 4.0, sampling_ratio=2
        )

        self.box_head = CustomBoxHead(in_channels=256, num_classes=num_classes)
        self.mask_head = CustomMaskHead(in_channels=256, num_classes=num_classes)

        self.anchor_generator = AnchorGenerator(
            sizes=(32, 64, 128), aspect_ratios=(0.5, 1.0, 2.0)
        )

    def forward(self, images, targets=None):
        """Main forward pass"""
        if self.training:
            assert targets is not None, "Targets required during training"
            return self.forward_train(images, targets)
        else:
            return self.forward_inference(images)
        
    def extract_features(self, images):
        """Extract features using backbone and FPN."""
        if isinstance(images, list):
            images = torch.stack(images)
        
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.cbam1(self.layer1(x))
        c2 = self.cbam2(self.layer2(c1))
        c3 = self.cbam3(self.layer3(c2))
        c4 = self.cbam4(self.layer4(c3))
        
        features = self.fpn([c1, c2, c3, c4])
        return features, images
    
    def forward_train(self, images, targets):
        """Training-specific forward pass."""
        device = images[0].device if isinstance(images, list) else images.device
        
        # Extract features
        features, images_tensor = self.extract_features(images)
        
        # RPN forward
        cls_scores, bbox_deltas = self.rpn(features)
        
        # Generate anchors
        feature_map = features[0]
        feature_h, feature_w = feature_map.shape[-2:]
        anchors = self.anchor_generator.generate_anchors(
            (feature_h, feature_w), stride=4, device=device
        )
        
        # Compute RPN loss (now in RPN class)
        rpn_losses = self.rpn.compute_loss(
            cls_scores, bbox_deltas, anchors, targets, device
        )
        
        # Generate training proposals
        cls_score = cls_scores[0][0] 
        proposals = generate_training_proposals(
            cls_score, anchors, images_tensor.shape[-2:], device
        )
        
        if len(proposals) == 0:
            return self._empty_losses(rpn_losses, device)
        
        # Sample proposals
        proposals, _ = sample_proposals(proposals, num_samples=128)
        
        # ROI Align
        roi_features = self.roi_align(feature_map[:1], [proposals])
        
        # Box and mask heads
        cls_logits, box_regression = self.box_head(roi_features)
        mask_logits = self.mask_head(roi_features)
        
        # Compute box and mask losses
        box_losses = self._compute_box_losses(
            cls_logits, box_regression, proposals, targets, device
        )
        mask_loss = self._compute_mask_losses(
            mask_logits, proposals, targets, device
        )
        
        # Combine all losses
        losses = {
            "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
            "loss_box_cls": box_losses["loss_cls"],
            "loss_box_reg": box_losses["loss_reg"],
            "loss_mask": mask_loss,
        }
        
        return losses
    
    def forward_inference(self, images):
        """Inference-specific forward pass."""
        device = images[0].device if isinstance(images, list) else images.device
        
        # Extract features
        features, images_tensor = self.extract_features(images)
        batch_size = images_tensor.size(0)
        
        # RPN forward
        cls_scores, bbox_deltas = self.rpn(features)
        
        # Generate anchors
        feature_map = features[0]
        feature_h, feature_w = feature_map.shape[-2:]
        anchors = self.anchor_generator.generate_anchors(
            (feature_h, feature_w), stride=4, device=device
        )
        
        predictions = []
        
        for batch_idx in range(batch_size):
            # Generate proposals for this image
            cls_score = cls_scores[0][batch_idx]
            proposals, top_scores = generate_inference_proposals(
                cls_score, anchors, images_tensor.shape[-2:], device
            )
            
            if len(proposals) == 0:
                predictions.append(self._empty_prediction(images_tensor, device))
                continue
            
            # Process proposals through detection heads
            single_feature = feature_map[batch_idx : batch_idx + 1]
            roi_features = self.roi_align(single_feature, [proposals])
            
            cls_logits, box_regression = self.box_head(roi_features)
            
            # Filter by classification score
            cls_probs = F.softmax(cls_logits, dim=-1)
            box_scores = cls_probs[:, 1]
            
            keep_scores = box_scores > 0.4
            final_boxes = proposals[keep_scores]
            final_scores = box_scores[keep_scores]
            roi_features_kept = roi_features[keep_scores]
            
            # Apply NMS
            if len(final_boxes) > 0:
                keep_nms = nms(final_boxes, final_scores, iou_threshold=0.5)
                final_boxes = final_boxes[keep_nms]
                final_scores = final_scores[keep_nms]
                roi_features_kept = roi_features_kept[keep_nms]
            
            # Generate masks
            final_masks = self._generate_masks(
                roi_features_kept, final_boxes, images_tensor.shape[-2:], device
            )
            
            predictions.append({
                "boxes": final_boxes,
                "labels": torch.ones(len(final_boxes), dtype=torch.long, device=device),
                "scores": final_scores,
                "masks": final_masks,
            })
        
        return predictions
    
    def _compute_box_losses(self, cls_logits, box_regression, proposals, targets, device):
        """Compute box classification and regression losses."""
        gt_boxes = targets[0]["boxes"]
        
        if len(gt_boxes) == 0:
            return {
                "loss_cls": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_reg": torch.tensor(0.0, device=device, requires_grad=True),
            }
        
        ious = box_iou(proposals, gt_boxes)
        max_iou, matched_gt_idx = ious.max(dim=1)
        
        labels = torch.zeros(len(proposals), dtype=torch.long, device=device)
        labels[max_iou >= 0.4] = 1
        
        cls_loss = F.cross_entropy(cls_logits, labels)
        
        foreground_mask = labels == 1
        if foreground_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[matched_gt_idx[foreground_mask]]
            fg_proposals = proposals[foreground_mask]
            fg_box_regression = box_regression[foreground_mask]
            fg_box_deltas = fg_box_regression[:, 4:8]
            target_deltas = encode_boxes(matched_gt_boxes, fg_proposals)
            reg_loss = F.smooth_l1_loss(fg_box_deltas, target_deltas, reduction="mean")
        else:
            reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {"loss_cls": cls_loss, "loss_reg": reg_loss}
    
    def _compute_mask_losses(self, mask_logits, proposals, targets, device):
        """Compute mask segmentation loss."""
        gt_boxes = targets[0]["boxes"]
        
        if len(gt_boxes) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        ious = box_iou(proposals, gt_boxes)
        max_iou, _ = ious.max(dim=1)
        foreground_mask = max_iou >= 0.4
        
        if foreground_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        fg_mask_logits = mask_logits[foreground_mask]
        fg_proposals = proposals[foreground_mask]
        
        mask_loss = compute_mask_loss_from_gt(
            fg_mask_logits, fg_proposals, targets, device, mask_size=28
        )
        
        return mask_loss
    
    def _generate_masks(self, roi_features, boxes, image_size, device):
        """Generate final masks from ROI features."""
        img_h, img_w = image_size
        num_detections = len(boxes)
        
        if num_detections == 0:
            return torch.zeros((0, img_h, img_w), dtype=torch.uint8, device=device)
        
        mask_logits = self.mask_head(roi_features)
        mask_probs = torch.sigmoid(mask_logits[:, 1])
        
        final_masks = torch.zeros((num_detections, img_h, img_w), device=device)
        
        for i, (box, mask_prob) in enumerate(zip(boxes, mask_probs)):
            x1, y1, x2, y2 = box.int()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 > x1 and y2 > y1:
                box_h, box_w = y2 - y1, x2 - x1
                mask_resized = F.interpolate(
                    mask_prob.unsqueeze(0).unsqueeze(0),
                    size=(box_h, box_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                
                mask_binary = (mask_resized > 0.5).float()
                final_masks[i, y1:y2, x1:x2] = mask_binary
        
        return (final_masks * 255).to(torch.uint8)
    
    def _empty_losses(self, rpn_losses, device):
        """Return empty losses when no proposals."""
        return {
            "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
            "loss_box_cls": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_box_reg": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
        }
    
    def _empty_prediction(self, images_tensor, device):
        """Return empty prediction."""
        img_h, img_w = images_tensor.shape[-2:]
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "scores": torch.zeros((0,), device=device),
            "masks": torch.zeros((0, img_h, img_w), dtype=torch.uint8, device=device),
        }

def count_parameters(self):
    """Count model parameters"""
    total = sum(p.numel() for p in self.parameters())
    
    backbone_params = sum(
        p.numel() for n, p in self.named_parameters()
        if any(x in n for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'])
        and 'cbam' not in n 
    )
    
    cbam_params = sum(
        p.numel() for n, p in self.named_parameters()
        if 'cbam' in n
    )
    
    fpn_params = sum(p.numel() for n, p in self.named_parameters() if 'fpn' in n)
    rpn_params = sum(p.numel() for n, p in self.named_parameters() if 'rpn' in n)
    box_head_params = sum(p.numel() for n, p in self.named_parameters() if 'box_head' in n)
    mask_head_params = sum(p.numel() for n, p in self.named_parameters() if 'mask_head' in n)
    roi_align_params = sum(p.numel() for n, p in self.named_parameters() if 'roi_align' in n)
    
    custom_params = total - backbone_params
    
    memory_bytes = total * 4
    memory_mb = memory_bytes / (1024 ** 2)
    
    return {
        'total': total,
        'backbone': backbone_params,
        'fpn': fpn_params,
        'rpn': rpn_params,
        'cbam': cbam_params,
        'box_head': box_head_params,
        'mask_head': mask_head_params,
        'roi_align': roi_align_params,
        'custom': custom_params,
        'custom_percentage': (custom_params / total) * 100 if total > 0 else 0,
        'memory_mb': memory_mb
    }


def get_custom_model(num_classes=2):
    """Factory function"""
    model = CustomMaskRCNN(num_classes=num_classes)
    return model