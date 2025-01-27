import torch
import torch.nn as nn
from torchvision import models

class TrafficLightEncoder(nn.Module):
    def __init__(self, feature_dim=128, pretrained=True):
        super(TrafficLightEncoder, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.fc = nn.Linear(num_features, feature_dim)

    def forward(self, x):
        # Select the front camera (index 0 along camera dimension) and current time (index 2 along time dimension)
        current_time_frame = x[:, 2, 0, :, :, :]  # Shape: [batch, channels, height, width]

        # Extract features using ResNet-50
        features = self.backbone(current_time_frame)  # Shape: [batch, num_features]

        # Reduce dimensionality
        features = self.fc(features)  # Shape: [batch, feature_dim]

        return features