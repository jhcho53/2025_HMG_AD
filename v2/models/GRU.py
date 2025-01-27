import torch
import torch.nn as nn

class BEVGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, height, width):
        super(BEVGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fully Connected Layer to project GRU output for each time step
        self.fc = nn.Linear(hidden_dim, output_dim * height * width)

    def forward(self, x, current_index=2, future_index=3):
        """
        Forward pass to process all time steps and extract current & future concatenated features.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, channel, height, width].
            current_index (int): Index of the current time step.
            future_index (int): Index of the future time step.
        
        Returns:
            output (torch.Tensor): Processed features for all time steps [batch, seq_len, output_dim, height, width].
            combined_bev (torch.Tensor): Concatenated BEV features for current and future time steps 
                [batch, 2 * output_dim, height, width].
        """
        # Input dimensions
        batch_size, seq_len, channel, height, width = x.size()

        # Flatten spatial dimensions into a single feature dimension
        x = x.view(batch_size, seq_len, -1)  # [batch, seq_len, channel * height * width]

        # Pass through GRU
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim]

        # Project GRU output to spatial dimensions for all time steps
        output = self.fc(gru_out)  # [batch, seq_len, output_dim * height * width]

        # Reshape to [batch, seq_len, output_dim, height, width]
        output = output.view(batch_size, seq_len, -1, self.height, self.width)

        # Extract current and future BEV features
        current_bev = output[:, current_index].unsqueeze(1)  # [batch, 1, output_dim, height, width]
        future_bev = output[:, future_index].unsqueeze(1)    # [batch, 1, output_dim, height, width]

        # Concatenate current and future features along the channel dimension
        combined_bev = torch.cat([current_bev, future_bev], dim=1)  # [batch, 2, output_dim, height, width]

        return output, combined_bev
    
class EgoStateGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        """
        Args:
            input_dim (int): Input feature dimension (e.g., 112 for ego state embedding).
            hidden_dim (int): Hidden state size of the GRU.
            output_dim (int): Output feature dimension for the GRU output projection.
            num_layers (int): Number of GRU layers.
        """
        super(EgoStateGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Project to desired output dimension

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 2, output_dim], containing the outputs for the last two time steps.
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # [batch_size, seq_len, hidden_dim]

        # Select the last two time steps
        last_two_outputs = gru_out[:, -2:, :]  # [batch_size, 2, hidden_dim]

        # Apply the fully connected layer to project to output_dim
        output = self.fc(last_two_outputs)  # [batch_size, 2, output_dim]

        return output
    
class FutureControlGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: GRU 입력 차원 (현재와 미래 프레임 정보 포함)
            hidden_dim: GRU의 hidden state 크기
            output_dim: 출력 차원 ([throttle, steer, brake])
        """
        super(FutureControlGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # GRU hidden state에서 제어값 추출

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] - 현재와 미래 프레임 정보를 포함한 시퀀스 입력
        Returns:
            future_control_value: [batch_size, output_dim] - 미래 제어값
        """
        output, _ = self.gru(x)  # GRU 모든 시점 출력: [batch_size, seq_len, hidden_dim]
        future_control_value = self.fc(output[:, -1])  # 마지막 시점 출력 사용: [batch_size, output_dim]
        return future_control_value