"""Model definitions for GeoLocate."""

import warnings

import torch.nn as nn
from torchvision import models

from config import BACKBONE_NAME


_RESNET_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


class Net(nn.Module):
    """Configurable ResNet backbone with classifier head for active class count."""

    def __init__(self, num_classes, pretrained=True, backbone_name=BACKBONE_NAME):
        super().__init__()
        if backbone_name not in _RESNET_BUILDERS:
            raise ValueError(
                "Unsupported BACKBONE_NAME. "
                "Expected one of: resnet18, resnet34, resnet50. "
                f"Got: {backbone_name}"
            )

        backbone_builder, default_weights = _RESNET_BUILDERS[backbone_name]
        weights = default_weights if pretrained else None
        try:
            self.backbone = backbone_builder(weights=weights)
        except (RuntimeError, OSError) as exc:
            # Fall back to random init if pretrained weights cannot be loaded.
            warnings.warn(
                f"Could not load pretrained {backbone_name} weights; continuing with "
                f"random initialization. Details: {exc}",
                stacklevel=2,
            )
            self.backbone = backbone_builder(weights=None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.backbone_name = backbone_name

    def freeze_backbone(self):
        """Freeze feature extractor layers and keep classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze the full network for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)