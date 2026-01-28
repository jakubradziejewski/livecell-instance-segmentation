import torch.nn as nn
import torch.nn.functional as F


class CustomBoxHead(nn.Module):
    """
    Box Head for classification and bounding box regression.
    Takes pooled ROI features and outputs class logits and bbox deltas.
    Consists of two fully connected layers followed by classification and regression heads.
    """

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
