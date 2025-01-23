import torch
from torch.utils.data import Dataset, DataLoader
from models.bev_encoder import Encoder
from models.hd_encoder import HDMapFeaturePipeline
from models.ego_encoder import FeatureEmbedding
from models.backbones.efficientnet import EfficientNetExtractor
from utils.attention import CrossViewAttention
from utils.utils import BEVEmbedding
from dataloader.dataloader import camDataLoader

def main():
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
    )

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
    )

    root_dir = "/home/vip/hd_jh/v2/Dataset_sample"  # Replace with the actual root directory path
    num_timesteps = 4
    calibration_dataset = camDataLoader(root_dir,num_timesteps=num_timesteps)
    calibration_dataloader = DataLoader(calibration_dataset, batch_size=1, shuffle=False)

    # Initialize placeholders for batch data
    for batch_idx, data in enumerate(calibration_dataloader):
        # Simulate image data (random tensor for example)
        batch = {
            "image": data["images"],  # (batch_size, num_views, channels, height, width)
            "intrinsics": data["intrinsic"],  # (batch_size, num_views, 3, 3)
            "extrinsics": data["extrinsic"],  # (batch_size, num_views, 4, 4)
            "hd_map": data["hd_map"],
            "ego_info": data["ego_info"],
        }

        # print(f"Batch {batch_idx + 1}:")
        # print("Image Shape:", batch["image"].shape)
        # print("Intrinsics Shape:", batch["intrinsics"].shape)
        # print("Extrinsics Shape:", batch["extrinsics"].shape)
        # print("HD Map Shape:", batch["hd_map"].shape)
        # print("Ego_info Shape:", batch["ego_info"].shape)

    # hd_map encoding
    hd_map_pipeline = HDMapFeaturePipeline(input_channels=6, final_channels=128, final_size=(32, 32))
    hd_features = hd_map_pipeline(batch["hd_map"]) # torch.Size([1, 4, 128, 32, 32])

    # ego encoding
    model = FeatureEmbedding(hidden_dim=32, output_dim=16)
    embedding = model(batch["ego_info"]) # Embedding Shape: torch.Size([1, 4, 112])
    
    # BEV encoding
    output = encoder(batch) # torch.Size([1, 4, 128, 32, 32])

if __name__ == "__main__":
    main()
