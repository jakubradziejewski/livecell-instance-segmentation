import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.ops import RoIAlign, nms, box_iou
import math

class ChannelAttention(nn.Module):
    """Channel attention with both avg and max pooling."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.fc(self.avg_pool(x).view(b, c))
        max_y = self.fc(self.max_pool(x).view(b, c))
        y = self.sigmoid(avg_y + max_y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention focusing on where objects are."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class FPN(nn.Module):
    """FPN"""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list]
        )

        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_list
            ]
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, features)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = []
        for lateral, output_conv in zip(laterals, self.output_convs):
            out = output_conv(lateral)
            outputs.append(out)

        return outputs

class RPN(nn.Module):
    """RPN with better objectness prediction."""

    def __init__(self, in_channels=256, num_anchors=9):
        super().__init__()

        # Single 3x3 conv (standard RPN design)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True), 
            
        )

        # Classification head: objectness score per anchor
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)

        # Regression head: 4 bbox deltas per anchor
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

        # Initialize with small weights (standard for detection heads)
        for layer in [self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        cls_scores = []
        bbox_deltas = []

        for feature in features:
            # Shared 3x3 conv
            t = self.conv(feature)

            # Parallel classification and regression
            cls_scores.append(self.cls_logits(t))
            bbox_deltas.append(self.bbox_pred(t))

        return cls_scores, bbox_deltas


class CustomMaskHead(nn.Module):
    """Mask head for predicting segmentation masks."""

    def __init__(self, in_channels=256, num_classes=2, mask_size=28):
        super().__init__()
        self.mask_size = mask_size

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.deconv_relu = nn.ReLU(inplace=True)

        self.mask_fcn_logits = nn.Conv2d(256, num_classes, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = self.deconv(x)
        x = self.deconv_relu(x)

        mask_logits = self.mask_fcn_logits(x)

        if mask_logits.shape[-1] != self.mask_size:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            )

        return mask_logits


class CustomBoxHead(nn.Module):
    """Box head for classification and bbox regression."""

    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()

        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        x = features.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_logits, bbox_deltas


class AnchorGenerator:
    """Anchor generator."""

    def __init__(self, sizes=(32, 64, 128), aspect_ratios=(0.5, 1.0, 2.0)):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors_per_location = len(sizes) * len(aspect_ratios)

    def generate_anchors(self, feature_map_size, stride, device):
        h, w = feature_map_size

        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                area = size * size
                h_anchor = math.sqrt(area / ratio)
                w_anchor = h_anchor * ratio

                base_anchors.append(
                    [-w_anchor / 2, -h_anchor / 2, w_anchor / 2, h_anchor / 2]
                )

        base_anchors = torch.tensor(base_anchors, device=device, dtype=torch.float32)

        shifts_x = torch.arange(0, w, device=device, dtype=torch.float32) * stride
        shifts_y = torch.arange(0, h, device=device, dtype=torch.float32) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(-1, 4)

        anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
        anchors = anchors.reshape(-1, 4)

        return anchors


def extract_mask_target(gt_mask, box, mask_size=28):
    """Extract mask target."""
    x1, y1, x2, y2 = box.int()

    h, w = gt_mask.shape
    x1 = max(0, min(x1.item(), w - 1))
    y1 = max(0, min(y1.item(), h - 1))
    x2 = max(x1 + 1, min(x2.item(), w))
    y2 = max(y1 + 1, min(y2.item(), h))

    mask_crop = gt_mask[y1:y2, x1:x2].float()

    if mask_crop.numel() == 0:
        return torch.zeros((mask_size, mask_size), device=gt_mask.device)

    mask_crop = mask_crop.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        mask_crop, size=(mask_size, mask_size), mode="bilinear", align_corners=False
    )

    return resized.squeeze()


def compute_mask_loss_from_gt(mask_logits, proposals, targets, device, mask_size=28):
    """Mask loss."""
    if len(proposals) == 0 or mask_logits is None:
        return torch.tensor(0.0, device=device)

    all_gt_boxes = []
    all_gt_masks = []
    all_gt_labels = []

    for target in targets:
        if len(target["boxes"]) > 0:
            all_gt_boxes.append(target["boxes"])
            all_gt_masks.append(target["masks"])
            all_gt_labels.append(target["labels"])

    if len(all_gt_boxes) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    gt_boxes = torch.cat(all_gt_boxes)
    gt_masks_list = torch.cat(all_gt_masks)
    gt_labels = torch.cat(all_gt_labels)

    ious = box_iou(proposals, gt_boxes)
    max_ious, matched_idxs = ious.max(dim=1)

    # LOOSER threshold for mask training
    positive_mask = max_ious > 0.3 

    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    positive_proposals = proposals[positive_mask]
    positive_mask_logits = mask_logits[positive_mask]
    matched_gt_idxs = matched_idxs[positive_mask]

    matched_gt_masks = gt_masks_list[matched_gt_idxs]
    matched_gt_labels = gt_labels[matched_gt_idxs]
    matched_gt_boxes = gt_boxes[matched_gt_idxs]

    mask_targets = []
    for gt_mask, gt_box in zip(matched_gt_masks, matched_gt_boxes):
        mask_target = extract_mask_target(gt_mask, gt_box, mask_size)
        mask_targets.append(mask_target)

    mask_targets = torch.stack(mask_targets)
    selected_mask_logits = positive_mask_logits[:, 1]

    mask_loss = F.binary_cross_entropy_with_logits(
        selected_mask_logits, mask_targets, reduction="mean"
    )

    return mask_loss


def encode_boxes(boxes, anchors):
    """Encode boxes with respect to anchors."""
    # Convert boxes to center format
    anchors_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchors_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchors_w = anchors[:, 2] - anchors[:, 0]
    anchors_h = anchors[:, 3] - anchors[:, 1]

    boxes_ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    boxes_ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    boxes_w = boxes[:, 2] - boxes[:, 0]
    boxes_h = boxes[:, 3] - boxes[:, 1]

    # Prevent division by zero
    anchors_w = torch.clamp(anchors_w, min=1.0)
    anchors_h = torch.clamp(anchors_h, min=1.0)
    boxes_w = torch.clamp(boxes_w, min=1.0)
    boxes_h = torch.clamp(boxes_h, min=1.0)

    # Encode as deltas (normalize by anchor size)
    dx = (boxes_ctr_x - anchors_ctr_x) / anchors_w
    dy = (boxes_ctr_y - anchors_ctr_y) / anchors_h
    dw = torch.log(boxes_w / anchors_w)
    dh = torch.log(boxes_h / anchors_h)

    deltas = torch.stack([dx, dy, dw, dh], dim=1)

    return deltas


class CustomMaskRCNN(nn.Module):
    """Custom Mask R-CNN using ResNet-18 backbone with FPN"""

    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        resnet = resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        

        # Add these CBAM modules
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        del resnet 
        # Custom layers
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

    def compute_rpn_loss(
        self, cls_scores_list, bbox_deltas_list, anchors, targets, device
    ):
        """RPN loss - simplified."""
        cls_scores = cls_scores_list[0]
        bbox_deltas = bbox_deltas_list[0]

        batch_size = cls_scores.size(0)

        cls_scores_flat = cls_scores.permute(0, 2, 3, 1).reshape(-1)
        bbox_deltas_flat = bbox_deltas.permute(0, 2, 3, 1).reshape(-1, 4)

        all_gt_boxes = []
        for target in targets:
            if len(target["boxes"]) > 0:
                all_gt_boxes.append(target["boxes"])

        if len(all_gt_boxes) == 0:
            return {
                "loss_rpn_cls": cls_scores_flat.sum() * 0.0 + 0.1,
                "loss_rpn_reg": bbox_deltas_flat.sum() * 0.0 + 0.1,
            }

        gt_boxes_cat = torch.cat(all_gt_boxes)

        if len(gt_boxes_cat) > 0 and len(anchors) > 0:
            ious = box_iou(anchors, gt_boxes_cat)
            max_ious, _ = ious.max(dim=1)

            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.3

            num_pos = min(pos_mask.sum().item(), 128)
            num_neg = min(neg_mask.sum().item(), 256 - num_pos)

            if num_pos > 0:
                pos_indices = torch.where(pos_mask)[0]
                pos_sampled = pos_indices[
                    torch.randperm(len(pos_indices), device=device)[:num_pos]
                ]
            else:
                pos_sampled = torch.tensor([], dtype=torch.long, device=device)

            if num_neg > 0:
                neg_indices = torch.where(neg_mask)[0]
                neg_sampled = neg_indices[
                    torch.randperm(len(neg_indices), device=device)[:num_neg]
                ]
            else:
                neg_sampled = torch.tensor([], dtype=torch.long, device=device)

            labels = torch.zeros(len(anchors), dtype=torch.float32, device=device)
            labels[pos_sampled] = 1.0

            sampled_indices = torch.cat([pos_sampled, neg_sampled])

            if len(sampled_indices) > 0:
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_scores_flat[sampled_indices], labels[sampled_indices]
                )

                if len(pos_sampled) > 0:
                    reg_loss = F.smooth_l1_loss(
                        bbox_deltas_flat[pos_sampled],
                        torch.zeros_like(bbox_deltas_flat[pos_sampled]),
                    )
                else:
                    reg_loss = bbox_deltas_flat.sum() * 0.0

                return {
                    "loss_rpn_cls": cls_loss,
                    "loss_rpn_reg": reg_loss * 0.5,
                }

        return {
            "loss_rpn_cls": cls_scores_flat.mean() * 0.1,
            "loss_rpn_reg": bbox_deltas_flat.mean() * 0.1,
        }

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)
        
        batch_size = images.size(0)
        device = images.device
        
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
                
        c1 = self.layer1(x)
        c1 = self.cbam1(c1)

        c2 = self.layer2(c1)
        c2 = self.cbam2(c2) 

        c3 = self.layer3(c2)
        c3 = self.cbam3(c3) 

        c4 = self.layer4(c3)
        c4 = self.cbam4(c4)  

        features = self.fpn([c1, c2, c3, c4])
        cls_scores, bbox_deltas = self.rpn(features)

        feature_map = features[0]
        cls_score = cls_scores[0]
        bbox_delta = bbox_deltas[0]

        feature_h, feature_w = feature_map.shape[-2:]
        anchors = self.anchor_generator.generate_anchors(
            (feature_h, feature_w), stride=4, device=device
        )

        if self.training:
            # Compute RPN losses
            rpn_losses = self.compute_rpn_loss(
                cls_scores, bbox_deltas, anchors, targets, device
            )

            # Generate proposals from RPN
            # Process first image in batch for simplicity
            objectness = torch.sigmoid(cls_score[0]).permute(1, 2, 0).reshape(-1)

            # Take many proposals for training
            top_k = min(500, len(objectness))
            top_scores, top_indices = torch.topk(objectness, top_k)

            # Lenient threshold for training
            score_keep = top_scores > 0.01  
            top_scores = top_scores[score_keep]
            top_indices = top_indices[score_keep]

            if len(top_indices) == 0:
                return {
                    "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
                    "loss_rpn_reg": rpn_losses["loss_rpn_reg"],
                    "loss_box_cls": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_box_reg": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                }

            # Get proposals from anchors
            proposals = anchors[top_indices]

            # Clamp proposals to image bounds
            img_h, img_w = images.shape[-2:]
            proposals[:, 0::2] = proposals[:, 0::2].clamp(0, img_w)
            proposals[:, 1::2] = proposals[:, 1::2].clamp(0, img_h)

            # Filter by size - very small minimum
            ws = proposals[:, 2] - proposals[:, 0]
            hs = proposals[:, 3] - proposals[:, 1]
            keep = (ws >= 5) & (hs >= 5)
            proposals = proposals[keep]

            if len(proposals) == 0:
                # No valid proposals - return zero losses
                return {
                    "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
                    "loss_rpn_reg": rpn_losses["loss_rpn_reg"],
                    "loss_box_cls": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_box_reg": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                }

            # Sample more proposals for training
            num_samples = min(128, len(proposals))
            sampled_indices = torch.randperm(len(proposals), device=device)[
                :num_samples
            ]
            sample_proposals = proposals[sampled_indices]

            # Extract ROI features from sampled proposals
            roi_features = self.roi_align(feature_map[:1], [sample_proposals])

            # Forward through detection heads
            cls_logits, box_regression = self.box_head(roi_features)
            mask_logits = self.mask_head(roi_features)

            # Match proposals to ground truth for loss computation
            gt_boxes = targets[0]["boxes"]

            if len(gt_boxes) == 0:
                # No GT boxes - return zero losses
                return {
                    "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
                    "loss_rpn_reg": rpn_losses["loss_rpn_reg"],
                    "loss_box_cls": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_box_reg": torch.tensor(
                        0.0, device=device, requires_grad=True
                    ),
                    "loss_mask": torch.tensor(0.0, device=device, requires_grad=True),
                }

            # Compute IoU between proposals and GT boxes
            ious = box_iou(sample_proposals, gt_boxes)
            max_iou, matched_gt_idx = ious.max(dim=1)

            # Matching threshold for foreground
            labels = torch.zeros(len(sample_proposals), dtype=torch.long, device=device)
            labels[max_iou >= 0.4] = 1

            # Box classification loss (foreground vs background)
            box_cls_loss = F.cross_entropy(cls_logits, labels)

            # Box regression loss (only for foreground proposals)
            foreground_mask = labels == 1
            if foreground_mask.sum() > 0:
                # Get matched GT boxes for foreground proposals
                matched_gt_boxes = gt_boxes[matched_gt_idx[foreground_mask]]
                fg_proposals = sample_proposals[foreground_mask]
                fg_box_regression = box_regression[foreground_mask]

                # box_regression shape: (N_fg, num_classes * 4) = (N_fg, 8)
                # Extract deltas for foreground class (class 1): columns 4:8
                fg_box_deltas = fg_box_regression[:, 4:8]  # Shape: (N_fg, 4)

                # Compute target deltas by encoding GT boxes relative to proposals
                target_deltas = encode_boxes(matched_gt_boxes, fg_proposals)

                # Now compute loss: predicted deltas vs target deltas
                box_reg_loss = F.smooth_l1_loss(
                    fg_box_deltas, target_deltas, reduction="mean"
                )
            else:
                box_reg_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Mask loss (only for foreground proposals)
            if foreground_mask.sum() > 0:
                fg_mask_logits = mask_logits[foreground_mask]
                fg_matched_gt_idx = matched_gt_idx[foreground_mask]

                # Compute mask loss using matched GT masks
                mask_loss = compute_mask_loss_from_gt(
                    fg_mask_logits,
                    sample_proposals[foreground_mask],
                    targets,
                    device,
                    mask_size=28,
                )
            else:
                mask_loss = torch.tensor(0.0, device=device, requires_grad=True)

            losses = {
                "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
                "loss_rpn_reg": rpn_losses["loss_rpn_reg"],
                "loss_box_cls": box_cls_loss,
                "loss_box_reg": box_reg_loss,
                "loss_mask": mask_loss,
            }
            return losses
        else:
            predictions = []

            for batch_idx in range(batch_size):
                objectness = (
                    torch.sigmoid(cls_score[batch_idx]).permute(1, 2, 0).reshape(-1)
                )

                top_k = min(250, len(objectness))
                top_scores, top_indices = torch.topk(objectness, top_k)

                score_keep = top_scores > 0.3
                top_scores = top_scores[score_keep]
                top_indices = top_indices[score_keep]

                proposals = anchors[top_indices]

                img_h, img_w = images.shape[-2:]
                proposals[:, 0::2] = proposals[:, 0::2].clamp(0, img_w)
                proposals[:, 1::2] = proposals[:, 1::2].clamp(0, img_h)

                ws = proposals[:, 2] - proposals[:, 0]
                hs = proposals[:, 3] - proposals[:, 1]
                keep = (ws >= 10) & (hs >= 10)
                proposals = proposals[keep]
                top_scores = top_scores[keep]

                if len(proposals) == 0:
                    predictions.append(
                        {
                            "boxes": torch.zeros((0, 4), device=device),
                            "labels": torch.zeros(
                                (0,), dtype=torch.long, device=device
                            ),
                            "scores": torch.zeros((0,), device=device),
                            "masks": torch.zeros(
                                (0, img_h, img_w), dtype=torch.uint8, device=device
                            ),
                        }
                    )
                    continue

                # More lenient NMS
                keep_nms = nms(proposals, top_scores, iou_threshold=0.4)
                proposals = proposals[keep_nms[:50]]

                single_feature = feature_map[batch_idx : batch_idx + 1]
                roi_features = self.roi_align(single_feature, [proposals])

                cls_logits, box_regression = self.box_head(roi_features)

                cls_probs = F.softmax(cls_logits, dim=-1)
                box_scores = cls_probs[:, 1]
                box_labels = torch.ones(
                    len(box_scores), dtype=torch.long, device=device
                )

                keep_scores = box_scores > 0.4
                final_boxes = proposals[keep_scores]
                final_scores = box_scores[keep_scores]
                final_labels = box_labels[keep_scores]
                roi_features_kept = roi_features[keep_scores]

                if len(final_boxes) > 0:
                    # More lenient final NMS
                    keep_final_nms = nms(
                        final_boxes, final_scores, iou_threshold=0.5
                    )  # Was 0.15
                    final_boxes = final_boxes[keep_final_nms]
                    final_scores = final_scores[keep_final_nms]
                    final_labels = final_labels[keep_final_nms]
                    roi_features_kept = roi_features_kept[keep_final_nms]

                num_detections = len(final_boxes)
                if num_detections > 0:
                    mask_logits = self.mask_head(roi_features_kept)
                    mask_probs = torch.sigmoid(mask_logits[:, 1])

                    final_masks = torch.zeros(
                        (num_detections, img_h, img_w), device=device
                    )

                    for i, (box, mask_prob) in enumerate(zip(final_boxes, mask_probs)):
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

                            # Lower mask threshold too
                            mask_binary = (mask_resized > 0.5).float()
                            final_masks[i, y1:y2, x1:x2] = mask_binary

                    final_masks = (final_masks * 255).to(torch.uint8)
                else:
                    final_masks = torch.zeros(
                        (0, img_h, img_w), dtype=torch.uint8, device=device
                    )

                predictions.append(
                    {
                        "boxes": final_boxes,
                        "labels": final_labels,
                        "scores": final_scores,
                        "masks": final_masks,
                    }
                )

            return predictions

    def count_parameters(self):
        """Count parameters."""
        total = sum(p.numel() for p in self.parameters())

        backbone_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if any(
                x in n for x in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
            )
        )

        custom_params = total - backbone_params

        return {
            "total": total,
            "backbone": backbone_params,
            "custom": custom_params,
            "custom_percentage": (custom_params / total) * 100,
        }


def get_custom_model(num_classes=2):
    """Factory function."""
    model = CustomMaskRCNN(num_classes=num_classes)
    return model
