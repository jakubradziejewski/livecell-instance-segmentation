import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefinementModule(nn.Module):
    """
    Two-branch module for boundary-aware mask prediction
    
    Branch 1: Boundary Detection (predicts cell edges)
    Branch 2: Mask Refinement (uses boundary info to improve masks)
    
    Critical for separating touching/overlapping cells!
    """
    def __init__(self, in_channels=256, hidden_channels=128):
        super().__init__()
        
        # Branch 1: Boundary Detection
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # Binary boundary map
        )
        
        # Branch 2: Mask Refinement (takes features + boundary)
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),  # +1 for boundary
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Refined mask
        )
    
    def forward(self, roi_features):
        """
        Args:
            roi_features: (B, 256, H, W) - features from ROI align
        
        Returns:
            refined_mask: (B, 1, H*2, W*2) - upsampled refined mask
            boundary_map: (B, 1, H, W) - boundary prediction
        """
        # Detect boundaries
        boundary_map = torch.sigmoid(self.boundary_detector(roi_features))
        
        # Concatenate features with boundary
        combined = torch.cat([roi_features, boundary_map], dim=1)
        
        # Refine mask
        refined_features = self.mask_refiner(combined)
        
        # Upsample to final mask size (14x14 -> 28x28)
        refined_mask = F.interpolate(
            refined_features,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        refined_mask = torch.sigmoid(refined_mask)
        
        return refined_mask, boundary_map


def extract_boundaries(masks, kernel_size=3):
    """
    Extract boundaries from ground truth masks for training
    
    Boundary = Mask - Eroded(Mask)
    
    Args:
        masks: (N, H, W) binary masks
        kernel_size: Erosion kernel size
    
    Returns:
        boundaries: (N, H, W) boundary maps
    """
    import cv2
    import numpy as np
    
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    boundaries = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    for mask in masks:
        mask = (mask * 255).astype(np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - eroded
        boundaries.append(boundary / 255.0)
    
    return torch.tensor(np.array(boundaries), dtype=torch.float32)