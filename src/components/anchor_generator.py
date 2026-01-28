import torch
import math


class AnchorGenerator:
    """Generates anchor boxes at multiple scales and aspect ratios."""

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
