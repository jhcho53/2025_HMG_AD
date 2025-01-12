import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class FeatureExtractor:
    def __init__(self, batch_size=32):
        # 사전 학습된 ResNet18 모델 불러오기
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        # 마지막 FC 레이어 이전까지 잘라서 사용
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.batch_size = batch_size

    def extract_features(self, camera_images_batch):
        """
        camera_images_batch: (batch_size, H, W, C) 형태의 텐서 (또는 리스트)
        """
        # 변환 적용
        camera_images_batch = torch.stack([
            self.transform(image) for image in camera_images_batch
        ])
        # 배치 크기를 self.batch_size에 맞춰서 맞춤 (부족할 경우 0 패딩)
        if camera_images_batch.size(0) < self.batch_size:
            padding_size = self.batch_size - camera_images_batch.size(0)
            padding = torch.zeros(padding_size, *camera_images_batch.shape[1:])
            camera_images_batch = torch.cat((camera_images_batch, padding), dim=0)
        
        with torch.no_grad():
            features = self.model(camera_images_batch)
        
        # features는 (batch_size, 512, 1, 1) 형태이므로 (batch_size, 512)로 변형
        features = features.view(features.size(0), -1)
        return features


class EGOFeatureExtractor(nn.Module):
    def __init__(self, ego_input_dim, hidden_dim, ego_feature_dim):
        """
        ego_input_dim: Ego 상태(자차 상태 + 경로 정보 등)의 입력 차원
        hidden_dim: 내부 FC 레이어 차원
        ego_feature_dim: EGO가 추출할 최종 feature 차원
        """
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
        """
        image_feature_dim: 이미지 FeatureExtractor에서 추출되는 특징 차원 (예: 512)
        ego_feature_dim: EGOFeatureExtractor에서 추출되는 특징 차원 (예: 128)
        hidden_dim: 내부 FC 레이어 차원
        output_dim: 최종 출력 차원 (예: 회전, 가속도, 브레이크 등 3차원)
        """
        super(CombinedMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_feature_dim + ego_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, image_features, ego_features):
        """
        image_features: (batch_size, image_feature_dim)
        ego_features: (batch_size, ego_feature_dim)
        """
        combined_input = torch.cat((image_features, ego_features), dim=1)
        return self.model(combined_input)
