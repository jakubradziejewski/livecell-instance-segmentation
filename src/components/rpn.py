import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

class RPN(nn.Module):
    """
    Region Proposal Network for objectness prediction.
    Applies a sliding window (3x3 conv) over each feature map from FPN.
    For every pixel in the feature map, it evaluates 'num_anchors' (pre-defined boxes).
    Cls_logits: Predicts the probability of an anchor containing a cell.
    Bbox_pred: Predicts 4 coordinates (deltas) to refine the anchor shape to fit the cell.
    Outputs potential "proposals" (regions of interest), then passed to ROI Align.
    """

    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True), 
        )

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        for layer in [self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        cls_scores = []
        bbox_deltas = []

        for feature in features:
            t = self.conv(feature)
            cls_scores.append(self.cls_logits(t))
            bbox_deltas.append(self.bbox_pred(t))

        return cls_scores, bbox_deltas
    
    def compute_loss(self, cls_scores_list, bbox_deltas_list, anchors, targets, device):
        """
        Compute RPN objectness classification loss.
        
        This method calculates the loss for the Region Proposal Network by:
        1. Matching anchors to ground truth boxes using IoU
        2. Sampling positive and negative anchors
        3. Computing binary cross-entropy loss for objectness classification
        """
        # Use first feature level for now (can be extended to multiple levels)
        cls_scores = cls_scores_list[0]
        cls_scores_flat = cls_scores.permute(0, 2, 3, 1).reshape(-1)

        # Collect all ground truth boxes from targets
        all_gt_boxes = []
        for target in targets:
            if len(target["boxes"]) > 0:
                all_gt_boxes.append(target["boxes"])

        # Handle case with no ground truth boxes
        if len(all_gt_boxes) == 0:
            return {
                "loss_rpn_cls": cls_scores_flat.sum() * 0.0 + 0.1,
            }

        gt_boxes_cat = torch.cat(all_gt_boxes)

        # Match anchors to ground truth boxes
        if len(gt_boxes_cat) > 0 and len(anchors) > 0:
            # Compute IoU between anchors and ground truth boxes
            ious = box_iou(anchors, gt_boxes_cat)
            max_ious, _ = ious.max(dim=1)

            # Define positive and negative anchors based on IoU thresholds
            pos_mask = max_ious >= 0.5  # High overlap = positive
            neg_mask = max_ious < 0.3   # Low overlap = negative

            # Sample balanced number of positive and negative anchors
            num_pos = min(pos_mask.sum().item(), 128)
            num_neg = min(neg_mask.sum().item(), 256 - num_pos)

            # Randomly sample positive anchors
            if num_pos > 0:
                pos_indices = torch.where(pos_mask)[0]
                pos_sampled = pos_indices[
                    torch.randperm(len(pos_indices), device=device)[:num_pos]
                ]
            else:
                pos_sampled = torch.tensor([], dtype=torch.long, device=device)

            # Randomly sample negative anchors
            if num_neg > 0:
                neg_indices = torch.where(neg_mask)[0]
                neg_sampled = neg_indices[
                    torch.randperm(len(neg_indices), device=device)[:num_neg]
                ]
            else:
                neg_sampled = torch.tensor([], dtype=torch.long, device=device)

            # Create labels: 1 for positive, 0 for negative
            labels = torch.zeros(len(anchors), dtype=torch.float32, device=device)
            labels[pos_sampled] = 1.0

            # Combine sampled indices
            sampled_indices = torch.cat([pos_sampled, neg_sampled])

            # Compute binary cross-entropy loss on sampled anchors
            if len(sampled_indices) > 0:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_scores_flat[sampled_indices], labels[sampled_indices]
                )

                return {
                    "loss_rpn_cls": cls_loss,
                }

        # Fallback loss if matching fails
        return {
            "loss_rpn_cls": cls_scores_flat.mean() * 0.1,
        }

