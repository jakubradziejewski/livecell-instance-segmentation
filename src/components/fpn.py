import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    Allows to detect objects at different scales.
    Uses Lateral connections - 1x1 conv to make channels uniform.
    Uses Top-down pathway - upsample higher-level features and add to lower-level.
    Output is a list of feature maps at different scales.
    Used in both RPN and highest resolution output is used in ROI Align.
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list]
        )

        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
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

        # Top-down pathway with lateral connections
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