import torch
from torch.utils.data import DataLoader
from utils.backbones.efficientnet import EfficientNetExtractor
from dataloader.bev_dataloader import MultiCamDataLoader

# Import the models
from utils.encoder import SequenceEncoder, SingleFrameEncoder, TemporalTransformer
from utils.BEVHDmapFusionNet import BEVHDMapFusionNet
from utils.decoder import BEVDecoderWithStackedGRU
from utils.VIP import CombinedModel
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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel().to(device)
    # Optimizer 및 Loss 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss() 

    model.eval() 
    with torch.no_grad(): 
        for batch_idx, (camera_images, intrinsics, extrinsics, hd_map_tensors, ego_inputs, _) in enumerate(loader):
            camera_images  = camera_images.to(device)
            intrinsics     = intrinsics.to(device)
            extrinsics     = extrinsics.to(device)
            hd_map_tensors = hd_map_tensors.to(device)
            ego_inputs     = ego_inputs.to(device)

            output = model(camera_images, intrinsics, extrinsics, hd_map_tensors, ego_inputs)
            print(f"[Batch {batch_idx}] Output Shape: {output.shape}")

            # 첫 번째 배치만 확인 후 종료
            break

if __name__ == "__main__":
    main()