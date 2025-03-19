import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm  # Vision Transformer 활용을 위해 설치 필요

# ------------------------------
# 1) Vision Transformer Feature Extractor
# ------------------------------
class VisionTransformerFeatureExtractor(nn.Module):
    """
    - timm 라이브러리의 ViT 모델 사용
    - 입력: (N, 3, 224, 224)  (한 프레임)
    - 출력: (N, feature_dim)  (비전 트랜스포머 임베딩)
    """
    def __init__(self, model_name="vit_base_patch16_224", batch_size=32, device=torch.device("cuda")):
        super(VisionTransformerFeatureExtractor, self).__init__()
        self.device = device
        self.batch_size = batch_size

        # timm 모델 로드 (pretrained weights)
        self.vit = timm.create_model(model_name, pretrained=True)

        # timm의 ViT는 (batch_size, 3, 224, 224)를 입력으로 받아
        # (batch_size, embed_dim) 형태의 cls_token 임베딩을 반환함.
        # 일부 모델은 fc나 head를 가지고 있으므로 제거할 수 있음.
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()  # 최종 분류 레이어 제거 (임베딩만 추출)
        if hasattr(self.vit, 'head_dist'):
            self.vit.head_dist = nn.Identity()

        self.vit.to(self.device)
        self.vit.eval()  # 기본적으로 eval 모드 (fine-tuning 시 train 모드로 변경)

    def forward(self, x):
        """
        x: (batch_size, 3, 224, 224)
        """
        # (batch_size, embed_dim)
        with torch.no_grad():
            features = self.vit(x)  # ViT 임베딩
        return features  # shape: (batch_size, feature_dim)


# ------------------------------
# 2) Temporal Transformer
# ------------------------------
class TemporalTransformer(nn.Module):
    """
    - 여러 프레임(시퀀스)의 Feature에 대해 시간 차원 Attention을 수행
    - 입력: (batch_size, seq_len, feature_dim)
    - 출력: (batch_size, feature_dim)  (마지막 혹은 평균 풀링 등)
    """
    def __init__(self, d_model, num_layers=2, nhead=4, dim_feedforward=256, max_seq_len=1000):
        super(TemporalTransformer, self).__init__()

        # Transformer 인풋에 맞춰 (seq_len, batch_size, feature_dim) 형태로 사용
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, features):
        """
        features: (batch_size, seq_len, feature_dim)
        """
        batch_size, seq_len, feature_dim = features.shape

        # (seq_len, batch_size, feature_dim) 형태로 변환
        features = features.permute(1, 0, 2)  # (seq_len, batch_size, feature_dim)

        # Positional Embedding 추가
        # pos_embedding: (1, max_seq_len, feature_dim)
        # features: (seq_len, batch_size, feature_dim)
        pos_embedding = self.pos_embedding[:, :seq_len, :]  # (1, seq_len, feature_dim)
        pos_embedding = pos_embedding.permute(1, 0, 2)  # (seq_len, 1, feature_dim)
        pos_embedding = pos_embedding.expand(seq_len, batch_size, feature_dim)  # (seq_len, batch_size, feature_dim)

        features = features + pos_embedding  # Positional Embedding 추가

        # TransformerEncoder
        # out: (seq_len, batch_size, feature_dim)
        out = self.transformer_encoder(features)

        # seq_len 차원에 대해 평균 풀링(또는 마지막 프레임만 취하기 등)
        # 여기서는 간단히 평균 풀링
        out = out.mean(dim=0)  # (batch_size, feature_dim)

        return out




# ------------------------------
# 3) EGO Feature Extractor
# ------------------------------
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


# ------------------------------
# 4) Combined MLP
# ------------------------------
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
        """
        image_features: (batch_size, feature_dim)
        ego_features: (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
        """
        # 만약 ego_features에 seq_len 차원이 있다면 평균 풀링 수행
        if len(ego_features.shape) == 3:
            ego_features = ego_features.mean(dim=1)  # (batch_size, feature_dim)

        # 결합
        combined_input = torch.cat((image_features, ego_features), dim=1)  # (batch_size, combined_feature_dim)
        return self.model(combined_input)


# ------------------------------
# 5) FullE2EModel
# ------------------------------
class FullE2EModel(nn.Module):
    def __init__(self, 
                 batch_size=32,
                 ego_input_dim=29, 
                 ego_hidden_dim=64, 
                 ego_feature_dim=128,
                 image_feature_dim=512,
                 transformer_d_model=768,
                 transformer_nhead=8, 
                 transformer_num_layers=6, 
                 mlp_hidden_dim=128,
                 output_dim=3,
                 max_seq_len=20):
        super(FullE2EModel, self).__init__()
        
        # 1) 이미지 FeatureExtractor
        self.feature_extractor = VisionTransformerFeatureExtractor(
            batch_size=batch_size
        )
        
        # 2) EGO FeatureExtractor
        self.ego_feature_extractor = EGOFeatureExtractor(
            ego_input_dim=ego_input_dim,
            hidden_dim=ego_hidden_dim,
            ego_feature_dim=ego_feature_dim
        )
        
        # 3) Linear Layer to match Transformer input dimension
        self.feature_projector = nn.Linear(768, transformer_d_model)

        # 4) Transformer Encoder (Temporal Attention)
        self.temporal_transformer = TemporalTransformer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            max_seq_len=max_seq_len
        )
        
        # 5) Combined MLP
        self.combined_mlp = CombinedMLP(
            image_feature_dim=transformer_d_model,
            ego_feature_dim=ego_feature_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim
        )

    def forward(self, camera_images_batch, ego_inputs):
        # 1) 이미지 특징 추출
        batch_size, seq_len, c, h, w = camera_images_batch.shape
        camera_images_batch = camera_images_batch.view(batch_size * seq_len, c, h, w)
        frame_features = self.feature_extractor(camera_images_batch)  # (N * seq_len, 512)

        frame_features = frame_features.view(batch_size, seq_len, -1)  # (N, seq_len, 512)

        # 2) Feature Projection to match Transformer input size
        frame_features = self.feature_projector(frame_features)  # (N, seq_len, 768)

        # 3) Transformer Encoder for temporal features
        time_features = self.temporal_transformer(frame_features)  # (N, 768)
        
        # 4) EGO 특징 추출
        ego_features = self.ego_feature_extractor(ego_inputs)  # (N, 128)
        
        # 5) 이미지+EGO 특징 합쳐서 MLP로 예측
        output = self.combined_mlp(time_features, ego_features)
        return output
