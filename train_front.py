import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader  # 예시
from dataloader.dataloader_front import DataLoader
from models.model_front import FeatureExtractor, EGOFeatureExtractor, CombinedMLP  # 여기서 import

def main():
    base_root = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E"
    map_name = "R_KR_PG_KATRI__HMG"  # 처리할 Map 이름
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # 데이터 로더 초기화 (예시)
    data_loader = DataLoader(base_root, map_name, batch_size=batch_size)

    # 특징 추출기 및 MLP 모델 초기화
    feature_extractor = FeatureExtractor(batch_size=batch_size)
    ego_feature_extractor = EGOFeatureExtractor(
        ego_input_dim=29,  # 상태 + 경로 입력 차원
        hidden_dim=64,
        ego_feature_dim=128
    )
    combined_mlp = CombinedMLP(
        image_feature_dim=512,
        ego_feature_dim=128,
        hidden_dim=128,
        output_dim=3
    )

    # 학습 모드로 설정
    ego_feature_extractor.train()
    combined_mlp.train()

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(ego_feature_extractor.parameters()) + list(combined_mlp.parameters()), 
        lr=learning_rate
    )

    # 모델 저장 경로
    save_path = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E/sample"
    os.makedirs(save_path, exist_ok=True)

    # 모든 Scenario 처리 및 학습
    for scenario_path in data_loader.scenario_paths:
        # 시나리오 세팅
        data_loader.set_scenario(scenario_path)
        print(f"Training on Map: {map_name}, Scenario: {data_loader.current_scenario}")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # 예시: data_loader에서 (camera_images, ego_inputs, gt_data)를 yield하는 구조
            for camera_images, ego_inputs, gt_data in data_loader:
                # 이미지 특징 추출
                image_features = feature_extractor.extract_features(camera_images)

                # EGO 입력 벡터 (현재 상태와 Global Path 포함)
                ego_inputs_with_path = torch.tensor(ego_inputs, dtype=torch.float32)

                # EGO 특징 추출
                ego_features = ego_feature_extractor(ego_inputs_with_path)

                # 배치 크기 불일치 처리 (데이터셋에서 잘리는 경우 등)
                min_batch_size = min(image_features.size(0), ego_features.size(0), gt_data.shape[0])
                image_features = image_features[:min_batch_size]
                ego_features = ego_features[:min_batch_size]
                gt_data = gt_data[:min_batch_size]

                # 모델 추론
                predictions = combined_mlp(image_features, ego_features)
                
                # 손실 계산
                target = torch.tensor(gt_data, dtype=torch.float32)
                loss = criterion(predictions, target)

                # 옵티마이저 업데이트
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Scenario: {data_loader.current_scenario}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 학습 완료 후 모델 저장
    torch.save(ego_feature_extractor.state_dict(), os.path.join(save_path, "ego_feature_extractor.pth"))
    torch.save(combined_mlp.state_dict(), os.path.join(save_path, "combined_mlp.pth"))
    
    print("Training complete! Model saved at:")
    print(f" - EGO Feature Extractor: {os.path.join(save_path, 'ego_feature_extractor.pth')}")
    print(f" - Combined MLP: {os.path.join(save_path, 'combined_mlp.pth')}")

if __name__ == "__main__":
    main()
