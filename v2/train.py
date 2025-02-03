import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder import Encoder, HDMapFeaturePipeline, FeatureEmbedding, TrafficLightEncoder
from models.decoder import TrafficSignClassificationHead, EgoStateHead, Decoder
from models.GRU import BEVGRU, EgoStateGRU 
from models.backbones.efficientnet import EfficientNetExtractor
from models.control import FutureControlMLP
from utils.attention import FeatureFusionAttention
from dataloader.dataloader import camDataLoader

class EndToEndModel(nn.Module):
    """
    End-to-End 모델 클래스.
    입력 배치에서 이미지, intrinsics, extrinsics, HD map, ego_info 등을 받아
    각 서브모듈(HD map 파이프라인, Ego GRU, BEV Encoder, Front-view Encoder,
    Fusion Attention, 분류 및 제어 헤드)을 거쳐 최종 제어 및 분류 출력을 생성합니다.
    """
    def __init__(self, config):
        """
        config: dict 형태의 설정값. (이미지 크기, BEV 크기, decoder_blocks 등)
        """
        super(EndToEndModel, self).__init__()
        # 기본 설정
        image_h = config.get("image_h", 135)
        image_w = config.get("image_w", 240)
        bev_h = config.get("bev_h", 150)
        bev_w = config.get("bev_w", 150)
        bev_h_meters = config.get("bev_h_meters", 50)
        bev_w_meters = config.get("bev_w_meters", 50)
        bev_offset = config.get("bev_offset", 0)
        decoder_blocks = config.get("decoder_blocks", [128, 128, 64])

        # Backbone 초기화 (EfficientNetExtractor)
        self.backbone = EfficientNetExtractor(
            model_name="efficientnet-b4",
            layer_names=["reduction_2", "reduction_4"],
            image_height=image_h,
            image_width=image_w,
        )
        
        # CrossViewAttention 관련 설정 (Encoder 내부에서 사용됨)
        cross_view_config = {
            "heads": 4,
            "dim_head": 32,
            "qkv_bias": True,
            "skip": True,
            "no_image_features": False,
            "image_height": image_h,
            "image_width": image_w,
        }
        
        # BEVEmbedding 관련 설정
        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": bev_h,
            "bev_width": bev_w,
            "h_meters": bev_h_meters,
            "w_meters": bev_w_meters,
            "offset": bev_offset,
            "decoder_blocks": decoder_blocks,
        }
        
        # Encoder 초기화
        self.encoder = Encoder(
            backbone=self.backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128,
            scale=1.0,
            middle=[2, 2],
        )
        
        # BEV GRU 모델 초기화
        # 입력 채널 수: 256 (hd map feature 128 + encoder output 128)
        # 출력 채널 수: 128
        input_dim = 256
        hidden_dim = 256
        output_dim = 128
        height, width = 18, 18
        self.bev_gru = BEVGRU(input_dim, hidden_dim, output_dim, height, width)
        
        # Ego GRU 모델 및 Feature Embedding 초기화
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.ego_gru = EgoStateGRU(input_dim=112, hidden_dim=256, output_dim=128, num_layers=1)
        
        # HD Map Feature Pipeline 초기화
        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=6, final_channels=128, final_size=(18, 18))
        
        # Front-view (Traffic) Encoder 초기화
        self.traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True)
        
        # Feature Fusion Attention 초기화
        self.fusion_model = FeatureFusionAttention(feature_dim=128, bev_dim=128, time_steps=2, spatial_dim=32)
        
        # Traffic Sign Classification Head 초기화
        self.classification_head = TrafficSignClassificationHead(input_dim=128, num_classes=10)
        
        # Future Control Head (MLP) 초기화
        self.control_mlp = FutureControlMLP(seq_len=2, input_dim=128, hidden_dim=64, output_dim=3)
        
        # Future Ego Head 초기화
        self.ego_header = EgoStateHead(input_dim=128, hidden_dim=64, output_dim=21)
        
        # BEV decoder 초기화
        self.bev_decoder = Decoder(dim=128, blocks=decoder_blocks, residual=True, factor=2)
    
    def forward(self, batch):
        """
        batch: dict with keys
            - "image": [B, num_views, C, H, W]
            - "intrinsics": [B, num_views, 3, 3]
            - "extrinsics": [B, num_views, 4, 4]
            - "hd_map": [B, time_steps, ...]
            - "ego_info": [B, ...]
        """
        # HD Map Encoding
        hd_features = self.hd_map_pipeline(batch["hd_map"])  
        # hd_features shape: [B, time_steps, 128, 18, 18]

        # Ego Encoding
        ego_embedding = self.feature_embedding(batch["ego_info"])  
        # ego_embedding shape: [B, seq_len, 112] (예시)
        ego_gru_output = self.ego_gru(ego_embedding)  
        # ego_gru_output shape: [B, future_steps, 128]
        
        # Ego decoding
        future_ego = self.ego_header(ego_gru_output)
        # future_ego shape = [B, future_steps, 21]
        
        # BEV Encoding
        bev_output = self.encoder(batch)  
        # bev_output shape: [B, time_steps, 128, 18, 18]
        
        # BEV Decoding
        bev_decoding = self.bev_decoder(bev_output)
        # bev_decoding shape: [B, time_steps, 64, 144, 144]
        
        # HD map feature와 BEV feature를 채널 차원에서 concat (256 채널)
        concat_bev = torch.cat([hd_features, bev_output], dim=2)  
        # concat_bev shape: [B, time_steps, 256, 18, 18]
        # GRU를 통해 과거, 현재, 미래 정보를 추출 (여기서는 미래 정보만 사용)
        _, gru_bev = self.bev_gru(concat_bev)  
        # gru_bev shape: [B, future_steps, 128, 18, 18]

        # Front-view Encoding (Traffic Light 관련 특징 추출)
        front_feature = self.traffic_encoder(batch["image"])  
        # front_feature shape: [B, 128]

        # Feature Fusion Attention: front, BEV, Ego 정보를 융합
        fusion_output = self.fusion_model(front_feature, gru_bev, ego_gru_output)  
        # fusion_output shape: [B, future_steps, 128]

        # Traffic Sign Classification Head (front feature 사용)
        classification_output = self.classification_head(front_feature)  
        # classification_output shape: [B, num_classes]

        # Future Control 예측 (fusion된 feature 사용)
        control_output = self.control_mlp(fusion_output)  
        # control_output shape: [B, 3]

        return {
            "control": control_output,                      # control_output shape: [B, 3]
            "classification": classification_output,        # classification_output shape: [B, num_classes]
            "future_ego": future_ego,                         # future_ego shape = [B, future_steps, 21]
            "bev_seg": bev_decoding                         # bev_seg shape: [B, time_steps, 64, 144, 144]
        }


def get_dataloader(root_dir, num_timesteps=3, batch_size=1):
    """
    데이터셋과 DataLoader 초기화 함수.
    """
    dataset = camDataLoader(root_dir, num_timesteps=num_timesteps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


def train():
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 설정 값
    config = {
        "image_h": 135,
        "image_w": 240,
        "bev_h": 150,
        "bev_w": 150,
        "bev_h_meters": 50,
        "bev_w_meters": 50,
        "bev_offset": 0,
        "decoder_blocks": [128, 128, 64],
    }
    
    # End-to-End 모델 초기화
    model = EndToEndModel(config)
    
    # 여러 GPU 사용을 위한 DataParallel 래핑
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 옵티마이저 및 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 각 헤드별 손실 함수
    # - classification: CrossEntropyLoss (one-hot이 아닌 class index가 필요하다면, one-hot vector를 argmax하거나 loss 함수를 조정)
    classification_loss_fn = nn.CrossEntropyLoss()
    # - control 및 ego 예측: MSELoss
    control_loss_fn = nn.MSELoss()
    ego_loss_fn = nn.MSELoss()
    # - bev segmentation: MSELoss (실제 적용 시 Dice loss, CrossEntropy 등으로 변경 가능)
    seg_loss_fn = nn.MSELoss()
    
    # 데이터 로더 초기화 (root_dir 경로는 실제 데이터셋 경로로 수정)
    root_dir = "/home/vip1/hd/2025_HMG_AD/v2/Dataset_sample"
    dataloader = get_dataloader(root_dir, num_timesteps=3, batch_size=1)
    
    num_epochs = 10
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            # 데이터의 각 항목을 device로 이동
            batch = {
                "image": data["images_input"].to(device),           # [B, num_views, C, H, W]
                "intrinsics": data["intrinsic_input"].to(device),     # [B, num_views, 3, 3]
                "extrinsics": data["extrinsic_input"].to(device),     # [B, num_views, 4, 4]
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }
            
            # Ground Truth (GT)
            ego_info_future_gt = data["ego_info_future"].to(device)   # [B, future_steps, 21]
            bev_seg_gt = data["gt_hd_map_input"].to(device)           # [B, T, 6, 144, 144]
            # classification GT: 만약 one-hot 벡터이면, argmax를 통해 class index로 변환 (예시)
            traffic_gt = data["traffic"].to(device)                   # [B, num_classes]
            traffic_gt_indices = torch.argmax(traffic_gt, dim=1)        # [B]
            control_gt = data["control"].to(device)                   # [B, 3]
            
            optimizer.zero_grad()
            outputs = model(batch)
            # 예측 결과
            control_pred = outputs["control"]         # [B, 3]
            classification_pred = outputs["classification"]   # [B, num_classes]
            future_ego_pred = outputs["future_ego"]     # [B, future_steps, 21]
            bev_seg_pred = outputs["bev_seg"]           # [B, time_steps, 64, 144, 144]
            
            # 손실 계산 (여러 헤드에 대해 개별 손실을 계산한 후, 가중합)
            loss_classification = classification_loss_fn(classification_pred, traffic_gt_indices)
            loss_control = control_loss_fn(control_pred, control_gt)
            loss_ego = ego_loss_fn(future_ego_pred, ego_info_future_gt)
            loss_seg = seg_loss_fn(bev_seg_pred, bev_seg_gt)
            
            # 총 손실 (가중치는 상황에 맞게 조정)
            loss = loss_classification + loss_control + loss_ego + loss_seg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}")
    
    # 학습 후 모델 저장
    torch.save(model.state_dict(), "end_to_end_model_parallel.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()
