import torch
from torch.utils.data import DataLoader
from utils.backbones.efficientnet import EfficientNetExtractor
from dataloader.bev_dataloader import MultiCamDataLoader

# Import the models
from utils.encoder import SequenceEncoder, SingleFrameEncoder, TemporalTransformer
from utils.BEVHDmapFusionNet import BEVHDMapFusionNet
from utils.decoder import BEVDecoderWithStackedGRU
def main():
    # Dataset and DataLoader settings
    base_root = "/home/vip/hd/Dataset"
    map_name = "R_KR_PG_KATRI__HMG"
    camera_dirs = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "CAMERA_5"]
    hd_map_dir = "HD_MAP"
    loader = MultiCamDataLoader(
        base_root=base_root,
        map_name=map_name,
        camera_dirs=camera_dirs,
        batch_size=1,  # B=1
        hd_map_dir=hd_map_dir,
        img_size=(224, 480),  # Image resolution
        time_steps=2,  # T=2
        map_size=(256, 256)
    )

    if not loader.scenario_paths:
        print("No scenarios found!")
        return

    scenario_path = loader.scenario_paths[0]
    loader.set_scenario(scenario_path)
    print(f"Loaded scenario: {scenario_path}")

    # Initialize the backbone (EfficientNetExtractor)
    backbone = EfficientNetExtractor(
        layer_names=['reduction_2', 'reduction_4'],
        image_height=224,
        image_width=480,
        model_name='efficientnet-b4'
    )

    # CrossViewAttention configuration
    cross_view_config = {
        "heads": 4,
        "dim_head": 32,
        "qkv_bias": True,
        "skip": True,
        "no_image_features": False,
        "image_height": 224,
        "image_width": 480,
    }

    # BEV Embedding configuration
    bev_embedding_config = {
        "sigma": 1.0,
        "bev_height": 200,
        "bev_width": 200,
        "h_meters": 100.0,
        "w_meters": 100.0,
        "offset": 0.0,
        "decoder_blocks": [2, 2],
    }

    # Instantiate encoders
    single_frame_encoder = SingleFrameEncoder(
        backbone=backbone,
        cross_view=cross_view_config,
        bev_embedding=bev_embedding_config,
        dim=128
    )

    temporal_transformer = TemporalTransformer(d_model=128, nhead=8, num_layers=2)

    sequence_encoder = SequenceEncoder(
        base_encoder=single_frame_encoder,
        temporal_module=temporal_transformer
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_encoder.to(device)
    
    model = BEVHDMapFusionNet(bev_dim=128, hd_map_dim=6, ego_dim=19, fused_dim=64, output_dim=32).to(device)
    model = model.to(device)
    model2 = BEVDecoderWithStackedGRU(input_channels=2, hidden_size=128, seq_len=32, spatial_dim=50, num_layers=3)
    model2 = model2.to(device)
    # Optimizer 및 Loss 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss() 

    model.eval() 
    with torch.no_grad(): 
        for batch_idx, (camera_images, intrinsics, extrinsics, hd_map_tensors, ego_inputs, _) in enumerate(loader):
            # (1) SequenceEncoder를 통해 BEV Features 추출
            batch = {
                "image": camera_images.to(device),
                "intrinsics": intrinsics.to(device),
                "extrinsics": extrinsics.to(device),
            }
            bev_features = sequence_encoder(batch)  # BEV Features 추출 (B, T, D, H, W)

            # (2) HD Map, Ego 정보 준비
            hd_map = hd_map_tensors.to(device)  # HD Map (B, T, C, H_map, W_map)
            ego_info = ego_inputs.to(device)    # Ego 정보 (B, T, ego_dim)

            # (3) 모델 출력 확인
            output = model(bev_features, hd_map, ego_info)  # 모델 예측 (B, T, output_dim, H, W)
            output = output.permute(0, 2, 1, 3, 4)
            output = model2(output)
            print(f"--- Batch {batch_idx} ---")
            print(f"Model Output Shape: {output.shape}")

            # 첫 번째 배치만 확인 후 종료
            break

if __name__ == "__main__":
    main()