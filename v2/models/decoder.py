import torch
import torch.nn as nn

class TrafficSignClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TrafficSignClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer
        self.fc2 = nn.Linear(64, 32)        # Hidden layer
        self.fc_out = nn.Linear(32, num_classes)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # [1, 128] -> [1, 64]
        x = self.relu(self.fc2(x))  # [1, 64] -> [1, 32]
        x = self.fc_out(x)          # [1, 32] -> [1, num_classes]
        return x
    
class EgoStateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=21):
        """
        GRU에서 나온 128차원 feature를 21차원의 미래 ego 상태 값으로 변환하는 MLP 헤더.
        """
        super(EgoStateHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 128 -> 64
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 64 -> 21
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, future_steps, 128)
        Returns:
            output: (batch_size, future_steps, 21)
        """
        return self.fc(x)