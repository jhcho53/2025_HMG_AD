import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class FeatureEmbedding(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=16):
        super(FeatureEmbedding, self).__init__()
        # 각 입력 그룹에 대해 MLP 정의
        self.position_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.orientation_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.enu_velocity_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.velocity_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.angular_velocity_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.acceleration_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.scalar_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)  # accel, brake, steer
        
    def forward(self, data):

        batch_size, time_steps, _ = data.shape
        data = data.view(batch_size * time_steps, -1)  # Flatten batch and time for consistent processing

        position = data[:, :3]
        orientation = data[:, 3:6]
        enu_velocity = data[:, 6:9]
        velocity = data[:, 9:12]
        angular_velocity = data[:, 12:15]
        acceleration = data[:, 15:18]
        scalars = data[:, 18:21]  # Ensure correct slicing
        
        # 각 그룹별 MLP 처리
        position_embed = self.position_mlp(position)
        orientation_embed = self.orientation_mlp(orientation)
        enu_velocity_embed = self.enu_velocity_mlp(enu_velocity)
        velocity_embed = self.velocity_mlp(velocity)
        angular_velocity_embed = self.angular_velocity_mlp(angular_velocity)
        acceleration_embed = self.acceleration_mlp(acceleration)
        scalar_embed = self.scalar_mlp(scalars)
        
        # 임베딩 결합
        combined = torch.cat((
            position_embed,
            orientation_embed,
            enu_velocity_embed,
            velocity_embed,
            angular_velocity_embed,
            acceleration_embed,
            scalar_embed
        ), dim=1)
        
        combined = combined.view(batch_size, time_steps, -1) 
        
        return combined
