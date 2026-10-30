"""MobileNetV2 transfer-learning classifier."""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
    return m


def freeze_backbone(m, freeze=True):
    for p in m.features.parameters():
        p.requires_grad = not freeze
