import torch.nn as nn
import torch.nn.functional as F

class CustomMaskHead(nn.Module):
    """Mask prediction head for instance segmentation."""

    def __init__(self, in_channels=256, num_classes=2, mask_size=28):
        super().__init__()
        self.mask_size = mask_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
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
