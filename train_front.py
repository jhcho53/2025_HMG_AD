import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader 
from dataloader.dataloader_front import DataLoader
from models.model_front import FullE2EModel 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_root = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E"
    map_name = "R_KR_PG_KATRI__HMG"
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # 데이터 로더 초기화
    data_loader = DataLoader(base_root, map_name, batch_size=batch_size)

    # 통합 모델 초기화
    full_model = FullE2EModel(
        batch_size=batch_size,
        ego_input_dim=29,
        ego_hidden_dim=64,
        ego_feature_dim=128,
        image_feature_dim=512, # ResNet18 특징 차원
        mlp_hidden_dim=128,
        output_dim=3
    ).to(device)  # 모델을 GPU로 이동
    full_model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)

    save_path = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E/sample"
    os.makedirs(save_path, exist_ok=True)

    for scenario_path in data_loader.scenario_paths:
        data_loader.set_scenario(scenario_path)
        print(f"Training on Map: {map_name}, Scenario: {data_loader.current_scenario}")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for camera_images, ego_inputs, gt_data in data_loader:
                # camera_images -> PyTorch Tensor 변환 & GPU 이동
                camera_images = torch.tensor(camera_images, dtype=torch.float32).to(device)
                # permute(차원 변경):
                camera_images = camera_images.permute(0, 3, 1, 2)  # (N,H,W,C)->(N,C,H,W) , (32, 224, 224, 3)

                # ego_inputs -> PyTorch Tensor 변환 & GPU 이동
                ego_inputs_tensor = torch.tensor(ego_inputs, dtype=torch.float32).to(device)

                # gt_data -> PyTorch Tensor 변환 & GPU 이동
                target = torch.tensor(gt_data, dtype=torch.float32).to(device)

                # 모델 Forward
                predictions = full_model(camera_images, ego_inputs_tensor)

                # 패딩 때문에 shapes가 다를 경우 슬라이싱
                min_batch_size = min(predictions.shape[0], target.shape[0])
                predictions = predictions[:min_batch_size]
                target = target[:min_batch_size]

                # 손실 계산
                loss = criterion(predictions, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Scenario: {data_loader.current_scenario}, "
                  f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(full_model.state_dict(), os.path.join(save_path, "full_e2e_model.pth"))
    print("Training complete! Model saved at:")
    print(f" - Full E2E Model: {os.path.join(save_path, 'full_e2e_model.pth')}")

if __name__ == "__main__":
    main()
