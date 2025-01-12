import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
from dataloader.dataloader_front import DataLoader  # DataLoader 클래스를 import


class FeatureExtractor:
    def __init__(self, batch_size=32):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.batch_size = batch_size

    def extract_features(self, camera_images_batch):
        camera_images_batch = torch.stack([self.transform(image) for image in camera_images_batch])
        if camera_images_batch.size(0) < self.batch_size:
            padding_size = self.batch_size - camera_images_batch.size(0)
            padding = torch.zeros(padding_size, *camera_images_batch.shape[1:])
            camera_images_batch = torch.cat((camera_images_batch, padding), dim=0)
        with torch.no_grad():
            features = self.model(camera_images_batch)
        features = features.view(features.size(0), -1)
        return features


class EGOFeatureExtractor(nn.Module):
    def __init__(self, ego_input_dim, hidden_dim, ego_feature_dim):
        super(EGOFeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ego_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ego_feature_dim),
        )

    def forward(self, ego_inputs):
        return self.model(ego_inputs)


class CombinedMLP(nn.Module):
    def __init__(self, image_feature_dim, ego_feature_dim, hidden_dim, output_dim):
        super(CombinedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_feature_dim + ego_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, image_features, ego_features):
        combined_input = torch.cat((image_features, ego_features), dim=1)
        return self.model(combined_input)


def main():
    base_path = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E/R_KR_PG_KATRI__HMG_Scenario_0"
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    # 데이터 로더 초기화
    data_loader = DataLoader(base_path, batch_size=batch_size)

    # 특징 추출기 및 MLP 모델 초기화
    feature_extractor = FeatureExtractor(batch_size=batch_size)
    ego_feature_extractor = EGOFeatureExtractor(ego_input_dim=19, hidden_dim=64, ego_feature_dim=128)
    combined_mlp = CombinedMLP(image_feature_dim=512, ego_feature_dim=128, hidden_dim=128, output_dim=3)
    ego_feature_extractor.train()  # 학습 모드
    combined_mlp.train()  # 학습 모드

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(ego_feature_extractor.parameters()) + list(combined_mlp.parameters()), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for camera_images, ego_inputs, gt_data in data_loader:
            # 이미지 특징 추출
            image_features = feature_extractor.extract_features(camera_images)

            # EGO 특징 추출
            ego_features = ego_feature_extractor(torch.tensor(ego_inputs, dtype=torch.float32))

            # 배치 크기 확인 및 일치화
            if image_features.size(0) != ego_features.size(0):
                min_batch_size = min(image_features.size(0), ego_features.size(0))
                image_features = image_features[:min_batch_size]
                ego_features = ego_features[:min_batch_size]
                gt_data = gt_data[:min_batch_size]

            # 결합된 입력 데이터를 MLP에 통과
            predictions = combined_mlp(image_features, ego_features)

            # 손실 계산
            target = torch.tensor(gt_data, dtype=torch.float32)
            loss = criterion(predictions, target)

            # 옵티마이저 초기화 및 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
