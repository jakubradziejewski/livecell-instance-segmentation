import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def extract_mask_target(gt_mask, box, mask_size=28):
    """
    Extract and resize mask target from ground truth mask within bounding box.
    
    This function:
    1. Crops the ground truth mask to the bounding box region
    2. Resizes the cropped mask to a fixed size (e.g., 28x28)
    3. Returns the resized mask for supervision
    
    Args:
        gt_mask: Ground truth mask [H, W]
        box: Bounding box [4] in (x1, y1, x2, y2) format
        mask_size: Target size for the mask (default: 28)
        
    Returns:
        resized_mask: Resized mask [mask_size, mask_size]
    """
    x1, y1, x2, y2 = box.int()

    h, w = gt_mask.shape
    
    # Clip coordinates to valid image bounds
    x1 = max(0, min(x1.item(), w - 1))
    y1 = max(0, min(y1.item(), h - 1))
    x2 = max(x1 + 1, min(x2.item(), w))
    y2 = max(y1 + 1, min(y2.item(), h))

    # Crop mask to bounding box
    mask_crop = gt_mask[y1:y2, x1:x2].float()

    # Handle empty crops
    if mask_crop.numel() == 0:
        return torch.zeros((mask_size, mask_size), device=gt_mask.device)

    # Resize to fixed size
    mask_crop = mask_crop.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        mask_crop, size=(mask_size, mask_size), mode="bilinear", align_corners=False
    )

    return resized.squeeze()


def compute_mask_loss_from_gt(mask_logits, proposals, targets, device, mask_size=28):
    """
    Compute mask loss by matching proposals to ground truth.
    
    This function:
    1. Matches proposals to ground truth boxes using IoU
    2. Extracts mask targets for matched proposals
    3. Computes binary cross-entropy loss between predicted and target masks
    
    The mask head predicts masks for each proposal, and we only supervise
    the proposals that have sufficient overlap with ground truth boxes.
    
    Args:
        mask_logits: Predicted mask logits [N, num_classes, mask_size, mask_size]
        proposals: Proposal boxes [N, 4] in (x1, y1, x2, y2) format
        targets: List of target dictionaries with 'boxes', 'masks', 'labels'
        device: Device for computation
        mask_size: Size of the mask predictions (default: 28)
        
    Returns:
        mask_loss: Binary cross-entropy loss for mask prediction
    """
    if len(proposals) == 0 or mask_logits is None:
        return torch.tensor(0.0, device=device)

    # Collect all ground truth boxes and masks
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

    # Concatenate all ground truth
    gt_boxes = torch.cat(all_gt_boxes)
    gt_masks_list = torch.cat(all_gt_masks)

    # Match proposals to ground truth boxes using IoU
    ious = box_iou(proposals, gt_boxes)
    max_ious, matched_idxs = ious.max(dim=1)

    # Only supervise proposals with high IoU (positive examples)
    positive_mask = max_ious > 0.3 

    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Get predictions and targets for positive proposals
    positive_mask_logits = mask_logits[positive_mask]
    matched_gt_idxs = matched_idxs[positive_mask]

    matched_gt_masks = gt_masks_list[matched_gt_idxs]
    matched_gt_boxes = gt_boxes[matched_gt_idxs]

    # Extract and resize mask targets
    mask_targets = []
    for gt_mask, gt_box in zip(matched_gt_masks, matched_gt_boxes):
        mask_target = extract_mask_target(gt_mask, gt_box, mask_size)
        mask_targets.append(mask_target)

    mask_targets = torch.stack(mask_targets)
    
    # Use class 1 predictions (foreground class)
    # Shape: [N, mask_size, mask_size]
    selected_mask_logits = positive_mask_logits[:, 1]

    # Compute binary cross-entropy loss
    mask_loss = F.binary_cross_entropy_with_logits(
        selected_mask_logits, mask_targets, reduction="mean"
    )

    return mask_loss


def paste_masks_in_image(masks, boxes, image_size, threshold=0.5):
    """
    Paste predicted masks into full image size.
    
    Args:
        masks: Predicted masks [N, mask_size, mask_size]
        boxes: Bounding boxes [N, 4] in (x1, y1, x2, y2) format
        image_size: Tuple of (height, width)
        threshold: Threshold for binarizing masks (default: 0.5)
        
    Returns:
        full_masks: Masks pasted into full image [N, H, W]
    """
    img_h, img_w = image_size
    num_masks = len(masks)
    device = masks.device
    
    if num_masks == 0:
        return torch.zeros((0, img_h, img_w), dtype=torch.uint8, device=device)
    
    full_masks = torch.zeros((num_masks, img_h, img_w), device=device)
    
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 > x1 and y2 > y1:
            box_h, box_w = y2 - y1, x2 - x1
            
            # Resize mask to box size
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(box_h, box_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            
            # Binarize and paste
            mask_binary = (mask_resized > threshold).float()
            full_masks[i, y1:y2, x1:x2] = mask_binary
    
    return (full_masks * 255).to(torch.uint8)