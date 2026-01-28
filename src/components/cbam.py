import torch
import torch.nn as nn

# CBAM Attention Module 
# Based on: Woo et al., "CBAM: Convolutional Block Attention Module"
# Link to their paper: https://arxiv.org/abs/1807.06521
class ChannelAttention(nn.Module):
    """
    Channel attention using average and max pooling.
    Avg pooling - global descriptor.
    Max pooling - strongest activations.
    Focus on important feature channels.
    Produces channel-wise attention weights.
    """

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
    """Spatial attention for localizing objects.
       Applied after channel attention.
       Produces spatial attention map.
    """

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
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

