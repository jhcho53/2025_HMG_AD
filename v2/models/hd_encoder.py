import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet 기반의 HD Map Feature Extractor
    """
    def __init__(self, input_channels=6, output_channels=2048):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet50(pretrained=True)
        # 첫 번째 Conv 레이어 수정
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # ResNet Feature Extractor

    def forward(self, x):
        return self.feature_extractor(x)


class FeatureAdjuster(nn.Module):
    """
    ResNet 출력 크기 조정을 위한 모듈
    - 채널 수 축소: 2048 -> 128
    - 해상도 조정: 8x8 -> 32x32
    """
    def __init__(self, input_channels=2048, output_channels=128, output_size=(32, 32)):
        super(FeatureAdjuster, self).__init__()
        self.channel_match = nn.Conv2d(input_channels, output_channels, kernel_size=1)  # 채널 축소
        self.output_size = output_size

    def forward(self, x):
        # 채널 축소
        x = self.channel_match(x)
        # 해상도 조정
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class HDMapFeaturePipeline(nn.Module):
    """
    HD Map 데이터로부터 최종 Feature 추출
    """
    def __init__(self, input_channels=6, resnet_output_channels=2048, final_channels=128, final_size=(32, 32)):
        super(HDMapFeaturePipeline, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(input_channels, resnet_output_channels)
        self.feature_adjuster = FeatureAdjuster(resnet_output_channels, final_channels, final_size)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape  # 입력 텐서의 크기 추출
        x = x.flatten(0, 1)  # Batch와 Time 차원을 결합 -> [batch*time, channels, height, width]

        # ResNet으로 Feature 추출
        x = self.feature_extractor(x)  # [batch*time, 2048, H', W']

        # Feature 크기 조정
        x = self.feature_adjuster(x)  # [batch*time, 128, 32, 32]

        # 배치와 시간 차원을 다시 분리
        x = x.view(batch_size, time_steps, 128, 32, 32)  # [batch, time, channels, height, width]
        return x