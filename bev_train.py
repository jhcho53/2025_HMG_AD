import torch
from torch.utils.data import DataLoader
from utils.backbones.efficientnet import EfficientNetExtractor
from dataloader.bev_dataloader import MultiCamDataLoader

# Import the models
from utils.encoder import SequenceEncoder, SingleFrameEncoder, TemporalTransformer


def main():
    # Dataset and DataLoader settings
    base_root = "/home/jaehyeon/Desktop/VIPLAB/HD_E2E"
    map_name = "R_KR_PG_KATRI__HMG"
    camera_dirs = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "CAMERA_5"]

    loader = MultiCamDataLoader(
        base_root=base_root,
        map_name=map_name,
        camera_dirs=camera_dirs,
        batch_size=1,  # B=1
        img_size=(224, 480),  # Image resolution
        time_steps=2,  # T=2
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

    # Process data through the SequenceEncoder
    for batch_idx, (camera_images, intrinsics, extrinsics, _, _) in enumerate(loader):
        batch = {
            "image": camera_images.to(device),
            "intrinsics": intrinsics.to(device),
            "extrinsics": extrinsics.to(device),
        }

        # Forward pass through the SequenceEncoder
        output_seq = sequence_encoder(batch)
        print(f"--- Batch {batch_idx} ---")
        print(f"Output Shape: {output_seq.shape}")

        # Process only the first batch for demonstration
        break


if __name__ == "__main__":
    main()
