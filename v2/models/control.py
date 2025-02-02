import torch
import torch.nn as nn

class FutureControlMLP(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim):
        """
        Args:
            seq_len (int): 시퀀스 길이 (현재 + 미래 프레임 수)
            input_dim (int): 각 시점의 feature 차원
            hidden_dim (int): MLP 내부 hidden layer 크기
            output_dim (int): 최종 출력 차원 ([throttle, steer, brake])
        """
        super(FutureControlMLP, self).__init__()
        # (seq_len * input_dim)을 입력으로 받아 hidden_dim으로 매핑
        self.fc1 = nn.Linear(seq_len * input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # hidden_dim에서 최종 제어값 (output_dim=3 등)으로 매핑
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, input_dim]
                             현재와 미래 프레임 정보를 포함한 시퀀스 입력
        Returns:
            future_control_value (torch.Tensor): [batch_size, output_dim]
                                                미래 제어값 (throttle, steer, brake 등)
        """
        b, s, d = x.shape  # batch_size, seq_len, input_dim

        # 시퀀스 차원을 펼쳐서 MLP 입력으로 사용
        x = x.view(b, s * d)  # [batch_size, seq_len * input_dim]

        # MLP 통과
        x = self.fc1(x)       # [batch_size, hidden_dim]
        x = self.relu(x)
        future_control_value = self.fc2(x)  # [batch_size, output_dim]

        return future_control_value
