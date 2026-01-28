# src/utils/proposal_utils.py
import torch
from torchvision.ops import nms
from src.utils.box_utils import clip_boxes_to_image, filter_small_boxes

def sample_proposals(proposals, num_samples=128):
    """Randomly sample proposals for training."""
    num_samples = min(num_samples, len(proposals))
    sampled_indices = torch.randperm(len(proposals), device=proposals.device)[:num_samples]
    return proposals[sampled_indices], sampled_indices

def generate_training_proposals(cls_scores, anchors, image_size, device,
                                num_proposals=500, score_threshold=0.01,
                                min_box_size=5):
    """Generate proposals for training."""
    objectness = torch.sigmoid(cls_scores).permute(1, 2, 0).reshape(-1)
    
    top_k = min(num_proposals, len(objectness))
    top_scores, top_indices = torch.topk(objectness, top_k)
    
    score_keep = top_scores > score_threshold
    top_scores = top_scores[score_keep]
    top_indices = top_indices[score_keep]
    
    proposals = anchors[top_indices]
    proposals = clip_boxes_to_image(proposals, image_size)
    
    keep = filter_small_boxes(proposals, min_box_size)
    proposals = proposals[keep]
    
    return proposals

def generate_inference_proposals(cls_scores, anchors, image_size, device,
                                 num_pre_nms=250, score_threshold=0.3,
                                 nms_threshold=0.4, num_post_nms=50,
                                 min_box_size=10):
    """Generate proposals for inference with NMS."""
    objectness = torch.sigmoid(cls_scores).permute(1, 2, 0).reshape(-1)
    
    top_k = min(num_pre_nms, len(objectness))
    top_scores, top_indices = torch.topk(objectness, top_k)
    
    score_keep = top_scores > score_threshold
    top_indices = top_indices[score_keep]
    top_scores = top_scores[score_keep]
    
    proposals = anchors[top_indices]
    proposals = clip_boxes_to_image(proposals, image_size)
    
    keep = filter_small_boxes(proposals, min_box_size)
    proposals = proposals[keep]
    top_scores = top_scores[keep]
    
    if len(proposals) > 0:
        keep_nms = nms(proposals, top_scores, iou_threshold=nms_threshold)
        proposals = proposals[keep_nms[:num_post_nms]]
        top_scores = top_scores[keep_nms[:num_post_nms]]
    
    return proposals, top_scores
