import torch
import torch.nn as nn
from torchvision.models import resnet18

# Stacked GRU Model 정의
class StackedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(StackedGRUModel, self).__init__()
        # Stacked GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout only applied when num_layers > 1
        )
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU forward pass
        gru_output, _ = self.gru(x)  # [Batch, Time, Hidden]
        # Use the last time step's output
        last_output = gru_output[:, -1, :]  # [Batch, Hidden]
        # Fully connected layer for output
        output = self.fc(last_output)  # [Batch, Output]
        return output

# ResNet 기반 BEV Decoder
class BEVDecoderWithStackedGRU(nn.Module):
    def __init__(self, input_channels, hidden_size, seq_len, spatial_dim, num_layers):
        super().__init__()
        # ResNet backbone (adapted for 2-channel input)
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Remove fully connected layer

        # Stacked GRU for temporal modeling
        spatial_feature_size = 512  # ResNet 마지막 특징 크기
        self.stacked_gru = StackedGRUModel(
            input_size=spatial_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3,  # accel, brake, steer
            dropout=0.2
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # ResNet feature extraction
        resnet_features = self.resnet(x)  # [B*T, 512]

        # Reshape for Stacked GRU
        resnet_features = resnet_features.view(B, T, -1)  # [B, T, 512]

        # Pass through Stacked GRU
        control_command = self.stacked_gru(resnet_features)
        return control_command

# Example usage
model = BEVDecoderWithStackedGRU(input_channels=2, hidden_size=128, seq_len=32, spatial_dim=50, num_layers=3)
input_tensor = torch.randn(1, 32, 2, 50, 50)  # [Batch, Time, Channels, Height, Width]
output = model(input_tensor)
print(output.shape)  # torch.Size([1, 3])
