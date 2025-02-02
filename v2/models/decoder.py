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