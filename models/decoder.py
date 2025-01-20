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
        # initialize hidden state
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
        # GRU forward pass
        gru_output, _ = self.gru(x, h0)  # [Batch, Time, Hidden]
        # Use the last time step's output
        last_output = gru_output[:, -1, :]  # [Batch, Hidden]
        # Fully connected layer for output
        output = self.fc(last_output)  # [Batch, Output]
        return output

class BEVDecoderWithStackedGRU(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_size: int,
                 seq_len: int,
                 spatial_dim: int,
                 num_layers: int = 3,
                 control_dim: int = 3):
        """
        Args:
            input_channels: GRU에 들어가기 전 feature의 채널 수
            hidden_size   : GRU hidden dim
            seq_len       : 시계열 길이 (T)
            spatial_dim   : H, W(=spatial_dim)일 때, 평면을 flatten한 크기는 spatial_dim*spatial_dim
            num_layers    : GRU 스택 수
            control_dim   : 최종 컨트롤 예측 차원 (예: 조향, 가속, 브레이크 등 3개라면 3)
        """
        super().__init__()
        self.seq_len = seq_len
        self.spatial_dim = spatial_dim
        self.num_layers = num_layers

        # GRU 입력 = (input_channels * spatial_dim * spatial_dim)
        self.rnn = nn.GRU(
            input_size=input_channels * (spatial_dim * spatial_dim),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # GRU 출력 -> (B, T, hidden_size) -> FC -> (B, T, control_dim)
        self.fc = nn.Linear(hidden_size, control_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B, input_channels, T, H, W)
        """
        B, C, T, H, W = x.shape
        
        # GRU 입력을 위해 (B, T, C*H*W)로 변환
        # batch_first=True이므로 (B, T, feature_dim) 형태가 되어야 함
        x = x.reshape(B, T, C * H * W)

        # GRU 진행
        # output: (B, T, hidden_size)
        # hidden_state: (num_layers, B, hidden_size)
        output, hidden_state = self.rnn(x)

        # 전체 타임스텝(T)에 대해 FC 적용 -> (B, T, control_dim)
        output = self.fc(output)

        return output
