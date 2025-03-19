import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils.VIP import CombinedModel
from dataloader.bev_dataloader import MultiCamDataLoader

def setup_ddp(rank, world_size):
    """
    DDP 환경 초기화.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # 포트 번호는 사용 가능한 것으로 설정
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """
    DDP 환경 정리.
    """
    torch.distributed.destroy_process_group()

def train(rank, world_size):
    """
    각 GPU에서 독립적으로 작업을 수행하는 함수.
    """
    setup_ddp(rank, world_size)

    # 데이터 및 DataLoader 설정
    base_root = "/home/vip/hd/Dataset"
    map_name = "R_KR_PG_KATRI__HMG"
    camera_dirs = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "CAMERA_5"]
    hd_map_dir = "HD_MAP"

    loader = MultiCamDataLoader(
        base_root=base_root,
        map_name=map_name,
        camera_dirs=camera_dirs,
        batch_size=1,  # 각 GPU에서 처리할 배치 크기
        hd_map_dir=hd_map_dir,
        img_size=(144, 256),
        time_steps=2,
        map_size=(150, 150)
    )

    if not loader.scenario_paths:
        print("No scenarios found!")
        return

    # 시나리오 나누기: 각 GPU에 다른 시나리오 할당
    scenario_paths = loader.scenario_paths
    local_scenario_paths = scenario_paths[rank::world_size]  # GPU별 시나리오 분배

    # 모델 및 학습 설정
    device = torch.device(f"cuda:{rank}")
    model = CombinedModel().to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    num_epochs = 2  # 에폭 설정

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # GPU마다 할당된 시나리오 학습
        for scenario_path in local_scenario_paths:
            loader.set_scenario(scenario_path)
            print(f"[Rank {rank}] Epoch [{epoch+1}/{num_epochs}] - Scenario: {scenario_path}")

            for batch_idx, (camera_images, intrinsics, extrinsics, hd_map_tensors, ego_inputs, target) in enumerate(loader):
                # 데이터 장치로 이동
                camera_images = camera_images.to(device, non_blocking=True)
                intrinsics = intrinsics.to(device, non_blocking=True)
                extrinsics = extrinsics.to(device, non_blocking=True)
                hd_map_tensors = hd_map_tensors.to(device, non_blocking=True)
                ego_inputs = ego_inputs.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Mixed Precision 학습
                with autocast():
                    output = model(camera_images, intrinsics, extrinsics, hd_map_tensors, ego_inputs)
                    loss = criterion(output, target)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"[Rank {rank}] Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(loader)}], "
                          f"Loss: {loss.item():.6f}")

        # 에폭 종료 후 로그 출력
        avg_loss = running_loss / len(loader)
        print(f"[Rank {rank}] Epoch [{epoch+1}/{num_epochs}] Complete: Average Loss: {avg_loss:.6f}")

        # 모델 저장 (Rank 0에서만 저장)
        if rank == 0:
            save_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.module.state_dict(), save_path)  # DDP에서는 model.module 사용
            print(f"[Rank {rank}] Model saved at {save_path}")
            
    cleanup_ddp()

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")

    spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
