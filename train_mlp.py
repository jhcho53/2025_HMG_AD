# main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import numpy as np
from dataloader.driving_dataset import DrivingDataset 

# ================================
# 1. 데이터 로드 및 전처리
# ================================

# Global Path 로드 (CSV 파일)
global_path = pd.read_csv("/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0/global_path.csv", delimiter=",")  # (N, 2) 형태
# X, Y 좌표만 추출
global_path = global_path[['PositionX (m)', 'PositionY (m)']].to_numpy()

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
# 3. 학습 설정 및 루프
# ================================

# 데이터 로드
file_paths = sorted(glob.glob("/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0/EGO_INFO/*.txt"))  # 시간 순서대로 정렬

# 데이터셋 및 데이터로더
dataset = DrivingDataset(file_paths=file_paths, global_path=global_path, n_points=5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 초기화
input_dim = dataset.inputs.shape[1]  # 입력 벡터 크기 (현재 상태 + Global Path)
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
# 4. 평가
# ================================

model.eval()
with torch.no_grad():
    test_predictions = model(dataset.inputs.clone().detach())
    test_loss = criterion(test_predictions, dataset.outputs.clone().detach())  
    print(f"Test Loss: {test_loss.item():.4f}")
