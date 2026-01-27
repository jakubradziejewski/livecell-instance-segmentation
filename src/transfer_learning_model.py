"""
Transfer Learning Model for Mask R-CNN
Uses pretrained ResNet-50-FPN backbone from torchvision
"""

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_maskrcnn(num_classes, pretrained=True):
    """
    Load Mask R-CNN model with ResNet-50-FPN backbone.
    (ResNet-18 is not available in torchvision's Mask R-CNN, so using ResNet-50)
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use COCO pretrained weights
        
    Returns:
        model: Mask R-CNN model
    """
    # Load pretrained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model