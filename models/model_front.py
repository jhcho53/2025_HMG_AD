import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np

class FeatureExtractor:
    def __init__(self, batch_size=32, device=torch.device("cuda")):
        self.device = device  # GPU 또는 CPU 지정
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.train()

        # 마지막 FC 레이어 이전까지 잘라서 사용
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # 모델을 지정된 디바이스로 이동
        self.model.to(self.device)

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
        camera_images_batch: (N, C, H, W) 형태의 GPU 텐서
        """
        # 모델의 디바이스와 일치하도록 확인
        if not camera_images_batch.is_cuda:
            raise RuntimeError("camera_images_batch is not on GPU!")

        # 모델 Forward로 feature 추출
        features = self.model(camera_images_batch)

        # features는 (batch_size, 512, 1, 1) 형태이므로 (batch_size, 512)로 변환
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


class FullE2EModel(nn.Module):
    """
    3가지 forward 흐름을 하나로 통합하여,
    이미지 + EGO 상태를 입력받고 최종 출력을 내는 E2E 모델.
    """
    def __init__(self, 
                 batch_size=32,
                 ego_input_dim=29, 
                 ego_hidden_dim=64, 
                 ego_feature_dim=128,
                 image_feature_dim=512,
                 mlp_hidden_dim=128,
                 output_dim=3):
        super(FullE2EModel, self).__init__()
        
        # 1) 이미지 FeatureExtractor
        self.feature_extractor = FeatureExtractor(batch_size=batch_size)
        
        # 2) EGO FeatureExtractor
        self.ego_feature_extractor = EGOFeatureExtractor(
            ego_input_dim=ego_input_dim,
            hidden_dim=ego_hidden_dim,
            ego_feature_dim=ego_feature_dim
        )
        
        # 3) Combined MLP
        self.combined_mlp = CombinedMLP(
            image_feature_dim=image_feature_dim,
            ego_feature_dim=ego_feature_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim
        )

    def forward(self, camera_images_batch, ego_inputs):
        """
        camera_images_batch: (batch_size, H, W, C) 형태의 텐서 (또는 리스트)
        ego_inputs: (batch_size, ego_input_dim) 형태의 텐서
        """
        # 1) 이미지 특징 추출
        image_features = self.feature_extractor.extract_features(camera_images_batch)
        
        # 2) EGO 특징 추출 (EGO states -> EGO features)
        ego_features = self.ego_feature_extractor(ego_inputs)
        
        # 3) 이미지+EGO 특징 합쳐서 MLP로 예측
        output = self.combined_mlp(image_features, ego_features)
        return output
