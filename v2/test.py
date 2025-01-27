import torch
from torch.utils.data import Dataset, DataLoader
from models.bev_encoder import Encoder
from models.hd_encoder import HDMapFeaturePipeline
from models.ego_encoder import FeatureEmbedding
from models.front_encoder import TrafficLightEncoder
from models.GRU import BEVGRU, EgoStateGRU, FutureControlGRU
from models.backbones.efficientnet import EfficientNetExtractor
from utils.attention import CrossViewAttention, FeatureFusionAttention
from utils.utils import BEVEmbedding
from dataloader.dataloader import camDataLoader

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 설정 직접 선언
    image_h, image_w = 270, 480  # 이미지 크기
    bev_h, bev_w = 256, 256  # BEV 크기
    bev_h_meters, bev_w_meters = 50, 50
    bev_offset = 0
    decoder_blocks = [128, 128, 64]

    # Backbone 초기화
    backbone = EfficientNetExtractor(
        model_name="efficientnet-b4",
        layer_names=["reduction_2", "reduction_4"],
        image_height=image_h,
        image_width=image_w,
    ).to(device)

    # CrossViewAttention 설정
    cross_view_config = {
        "heads": 4,
        "dim_head": 32,
        "qkv_bias": True,
        "skip": True,
        "no_image_features": False,
        "image_height": image_h,
        "image_width": image_w,
    }

    # BEVEmbedding 설정
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
    encoder = Encoder(
        backbone=backbone,
        cross_view=cross_view_config,
        bev_embedding=bev_embedding_config,
        dim=128,
        scale=1.0,
        middle=[2, 2],
    ).to(device)

    # BEV GRU 모델 초기화
    input_dim = 256 * 32 * 32  # channel * height * width
    hidden_dim = 256  # GRU hidden state 크기
    output_dim = 128  # 최종 출력 채널 수
    height, width = 32, 32

    bev_gru_model = BEVGRU(input_dim, hidden_dim, output_dim, height, width).to(device)

    # EGO GRU 모델 초기화
    ego_gru_model = EgoStateGRU(input_dim=112, hidden_dim=256, output_dim=128, num_layers=1).to(device)

    # Traffic Light Encoder 초기화
    Traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True).to(device)
    
    # Feature fusion attention 초기화
    fusion_model = FeatureFusionAttention(feature_dim=128, bev_dim=128, time_steps=2, spatial_dim=32).to(device)

    # Future Control GRU 초기화
    future_control_gru = FutureControlGRU(input_dim=128, hidden_dim=64, output_dim=3).to(device)
    
    root_dir = "/home/vip1/hd/2025_HMG_AD/v2/Dataset_sample"  # Replace with the actual root directory path
    num_timesteps = 4
    calibration_dataset = camDataLoader(root_dir, num_timesteps=num_timesteps)
    calibration_dataloader = DataLoader(calibration_dataset, batch_size=1, shuffle=False)

    # Initialize placeholders for batch data
    for batch_idx, data in enumerate(calibration_dataloader):
        # Simulate image data (random tensor for example)
        batch = {
            "image": data["images"].to(device),       # (batch_size, num_views, channels, height, width)
            "intrinsics": data["intrinsic"].to(device),  # (batch_size, num_views, 3, 3)
            "extrinsics": data["extrinsic"].to(device),  # (batch_size, num_views, 4, 4)
            "hd_map": data["hd_map"].to(device),
            "ego_info": data["ego_info"].to(device),
        }

        # HD Map Encoding
        hd_map_pipeline = HDMapFeaturePipeline(input_channels=6, final_channels=128, final_size=(32, 32)).to(device)
        hd_features = hd_map_pipeline(batch["hd_map"])  # torch.Size([1, 4, 128, 32, 32])

        # Ego Encoding
        model = FeatureEmbedding(hidden_dim=32, output_dim=16).to(device)
        embedding = model(batch["ego_info"])  # Embedding Shape: torch.Size([1, 4, 112])
        ego_gru_output = ego_gru_model(embedding)  # torch.size([1, 2, 128])

        # BEV Encoding
        output = encoder(batch)  # torch.Size([1, 4, 128, 32, 32])
        concat_bev = torch.cat([hd_features, output], dim=2)  # torch.size([1, 4, 256, 32, 32])

        gru_bev_train, gru_bev = bev_gru_model(concat_bev)  # torch.size([1, 4, 128, 32, 32]) // torch.size(1, 2, 128, 32, 32)

        # Front-view Encoding
        front_feature = Traffic_encoder(batch["image"])  # torch.Size([1, 128])
        fusion_output = fusion_model(front_feature, gru_bev, ego_gru_output)  # torch.Size([1, 2, 128])

        # Control Output
        control = future_control_gru(fusion_output)  # torch.size([1,3])

if __name__ == "__main__":
    main()