# main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import numpy as np

# ================================
# 1. 데이터 로드 및 전처리
# ================================

# Global Path 로드 (CSV 파일)
global_path = pd.read_csv("/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0/global_path.csv", delimiter=",")  # (N, 2) 형태
# X, Y 좌표만 추출
global_path = global_path[['PositionX (m)', 'PositionY (m)']].to_numpy()


def parse_file(file_path):
    """단일 파일에서 입력 벡터와 출력 벡터 생성"""
    with open(file_path, 'r') as f:
        data = {}
        for line in f:
            key, value = line.split(": ", 1)  # ":" 기준으로 나눔
            if key.strip() in ["linkid", "trafficlightid", "turn_signal_lamp"]:
                continue  # 문자열 필드는 무시
            try:
                data[key.strip()] = list(map(float, value.strip().split()))
            except ValueError:
                # 숫자로 변환할 수 없는 데이터가 있을 경우 무시
                continue
    
    # 입력 벡터 생성 (현재 상태)
    input_vector = [
        *data['position'],                  # x, y, z
        data['orientation'][-1],           # yaw
        *data['enu_velocity'],             # v_x, v_y, v_z
        *data['acceleration'],             # a_x, a_y, a_z
        *data['angularVelocity']           # ω_x, ω_y, ω_z
    ]
    
    # 출력 벡터 생성 (제어 명령)
    output_vector = [
        data['accel'][0],                  # accel
        data['brake'][0],                  # brake
        data['steer'][0]                   # steer
    ]
    
    return input_vector, output_vector

def get_next_path_points(global_path, current_position, n_points=5):
    """현재 위치에서 가장 가까운 경로 점 n개 가져오기"""
    distances = np.linalg.norm(global_path - np.array(current_position[:2]), axis=1)
    nearest_idx = np.argmin(distances)
    next_points = global_path[nearest_idx:nearest_idx + n_points]
    # 패딩 처리
    if len(next_points) < n_points:
        padding = np.zeros((n_points - len(next_points), global_path.shape[1]))
        next_points = np.vstack((next_points, padding))
    return next_points

def compute_relative_path(next_points, current_position):
    """경로 점과 현재 위치 간의 상대 좌표 계산"""
    relative_points = next_points - np.array(current_position[:2])
    return relative_points.flatten()  # 1D 벡터 반환

def create_input_vector(current_state, global_path, n_points=5):
    """현재 상태와 Global Path 정보를 결합하여 입력 벡터 생성"""
    next_points = get_next_path_points(global_path, current_state, n_points)
    relative_path = compute_relative_path(next_points, current_state)
    input_vector = np.concatenate([current_state, relative_path])
    return input_vector

def load_dataset(file_paths, global_path, n_points=5):
    """전체 데이터셋 로드 및 입력/출력 벡터 생성"""
    inputs, outputs = [], []
    for file_path in file_paths:
        current_state, output_vector = parse_file(file_path)
        input_vector = create_input_vector(current_state, global_path, n_points)
        inputs.append(input_vector)
        outputs.append(output_vector)
    return np.array(inputs), np.array(outputs)

# ================================
# 2. 모델 정의
# ================================

class DrivingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DrivingMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # accel, brake, steer
        )

    def forward(self, x):
        return self.model(x)

# ================================
# 3. 데이터셋 클래스 정의
# ================================

class DrivingDataset(Dataset):
    def __init__(self, file_paths, global_path, n_points=5):
        # file_paths, global_path를 받아서 데이터를 처리
        inputs, outputs = load_dataset(file_paths, global_path, n_points)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# ================================
# 4. 학습 설정 및 루프
# ================================

# 데이터 로드
file_paths = sorted(glob.glob("/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0/EGO_INFO/*.txt"))  # 시간 순서대로 정렬
inputs, outputs = load_dataset(file_paths, global_path, n_points=5)

# 데이터셋 및 데이터로더
dataset = DrivingDataset(file_paths=file_paths, global_path=global_path, n_points=5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 초기화
input_dim = inputs.shape[1]  # 입력 벡터 크기 (현재 상태 + Global Path)
model = DrivingMLP(input_dim=input_dim, hidden_dim=64)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_outputs in dataloader:
        # Forward pass
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ================================
# 5. 평가
# ================================

model.eval()
with torch.no_grad():
    test_predictions = model(torch.tensor(inputs, dtype=torch.float32))
    test_loss = criterion(test_predictions, torch.tensor(outputs, dtype=torch.float32))
    print(f"Test Loss: {test_loss.item():.4f}")