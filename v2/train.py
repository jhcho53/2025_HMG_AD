import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# (기존의 import 구문 유지)
from models.encoder import Encoder, HDMapFeaturePipeline, FeatureEmbedding, TrafficLightEncoder
from models.decoder import TrafficSignClassificationHead, EgoStateHead, Decoder
from models.GRU import BEVGRU, EgoStateGRU 
from models.backbones.efficientnet import EfficientNetExtractor
from models.control import FutureControlMLP, ControlMLP
from utils.attention import FeatureFusionAttention
from dataloader.dataloader import camDataLoader


class EndToEndModel(nn.Module):
    """
    End-to-End 모델 클래스.
    입력 배치에서 이미지, intrinsics, extrinsics, HD map, ego_info 등을 받아
    각 서브모듈을 거쳐 최종 제어 및 분류 출력을 생성합니다.
    """
    def __init__(self, config):
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
        input_dim = 256
        hidden_dim = 256
        output_dim = 128
        height, width = 18, 18
        self.bev_gru = BEVGRU(input_dim, hidden_dim, output_dim, height, width)
        
        # Ego GRU 및 Feature Embedding 초기화
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.ego_gru = EgoStateGRU(input_dim=112, hidden_dim=256, output_dim=128, num_layers=1)
        
        # HD Map Feature Pipeline 초기화
        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=6, final_channels=128, final_size=(18, 18))
        
        # Front-view (Traffic) Encoder 초기화
        self.traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True)
        
        # Traffic Sign Classification Head 초기화
        self.classification_head = TrafficSignClassificationHead(input_dim=128, num_classes=10)
        
        # Future Control Head (MLP) 초기화
        self.control = ControlMLP(future_steps=2, control_dim=3)
        
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

        # Traffic Sign Classification Head (front feature 사용)
        classification_output = self.classification_head(front_feature)  
        # classification_output shape: [B, num_classes]

        # Future Control 예측 (fusion된 feature 사용)
        control_output = self.control(front_feature, gru_bev, ego_gru_output)
        # control_output shape: [B, 3]

        return {
            "control": control_output,                      # [B, 3]
            "classification": classification_output,        # [B, num_classes]
            "future_ego": future_ego,                         # [B, future_steps, 21]
            "bev_seg": bev_decoding                         # [B, time_steps, 64, 144, 144]
        }


def get_dataloader(dataset, batch_size=16, sampler=None):
    """
    데이터셋과 DataLoader 초기화 함수.
    DistributedSampler가 제공되면 이를 사용.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    return dataloader


def train(local_rank, args, distributed=False):
    if distributed:
        # 분산 환경을 위한 일부 환경 변수 설정 (실행 환경에 따라 torchrun 등에서 자동 설정될 수 있음)
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        
        # 분산 환경 초기화 (init_method='env://'를 사용)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            print(f"Using distributed training with {world_size} processes.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        print(f"Using single GPU mode on device: {device}")

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
    
    # End-to-End 모델 초기화 및 device로 이동
    model = EndToEndModel(config).to(device)
    
    # 분산 모드이면 DDP로 모델 래핑
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    # 옵티마이저 및 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    classification_loss_fn = nn.CrossEntropyLoss()
    control_loss_fn = nn.MSELoss()
    ego_loss_fn = nn.MSELoss()
    seg_loss_fn = nn.MSELoss()
    
    # 데이터셋 초기화 (하나의 dataset 인스턴스를 사용)
    root_dir = "/home/vip/hd/Dataset"  # 실제 데이터셋 경로로 수정
    dataset = camDataLoader(root_dir, num_timesteps=3)
    
    # 분산 모드이면 DistributedSampler 사용, 아니면 기본 DataLoader 사용
    if distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = get_dataloader(dataset, batch_size=16, sampler=sampler)
    else:
        sampler = None
        dataloader = get_dataloader(dataset, batch_size=16, sampler=None)
    
    num_epochs = 1
    model.train()
    
    for epoch in range(num_epochs):
        # 매 epoch마다 sampler의 시드를 변경해 데이터 셔플링 보장 (distributed 모드에서만)
        if distributed:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            if data is None:  # None 데이터가 전달되었는지 확인
                print(f"Warning: Skipping batch {batch_idx} due to empty data.")
                break  # 배치가 None이면 skip.

            # 데이터의 각 항목을 device로 이동
            batch = {
                "image": data["images_input"].to(device),           # [B, T, num_views, C, H, W]
                "intrinsics": data["intrinsic_input"].to(device),     # [B, num_views, 3, 3]
                "extrinsics": data["extrinsic_input"].to(device),     # [B, num_views, 4, 4]
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }

            # Ground Truth (GT)
            ego_info_future_gt = data["ego_info_future"].to(device)   # [B, future_steps, 21]
            bev_seg_gt = data["gt_hd_map_input"].to(device)            # [B, T, 6, 144, 144]
            traffic_gt = data["traffic"].to(device)                    # [B, num_classes]
            traffic_gt_indices = torch.argmax(traffic_gt, dim=1)         # [B]
            control_gt = data["control"].to(device)                    # [B, 3]
            
            optimizer.zero_grad()
            outputs = model(batch)
            control_pred = outputs["control"]
            classification_pred = outputs["classification"]
            future_ego_pred = outputs["future_ego"]
            bev_seg_pred = outputs["bev_seg"]
            
            # 손실 계산 (각 손실을 합산)
            loss_classification = classification_loss_fn(classification_pred, traffic_gt_indices)
            loss_control = control_loss_fn(control_pred, control_gt)
            loss_ego = ego_loss_fn(future_ego_pred, ego_info_future_gt)
            loss_seg = seg_loss_fn(bev_seg_pred, bev_seg_gt)
            loss = loss_classification + loss_control + loss_ego + loss_seg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # rank 0에서만 출력 (단일 모드에서는 항상 rank 0)
            if batch_idx % 10 == 0 and rank == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}")
                print(f"  Classification Loss: {loss_classification.item():.4f}")
                print(f"  Control Loss      : {loss_control.item():.4f}")
                print(f"  Ego State Loss    : {loss_ego.item():.4f}")
                print(f"  BEV Segmentation Loss: {loss_seg.item():.4f}")
                print(f"  Total Loss        : {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}")
    
    # rank 0에서만 모델 저장 (DDP로 래핑한 경우 실제 모델은 model.module에 위치)
    if rank == 0:
        torch.save(model.module.state_dict() if distributed else model.state_dict(), "end_to_end_model.pth")
        print("Training complete and model saved.")
    
    # 분산 모드이면 분산 프로세스 그룹 종료
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Provided by distributed launcher if using distributed training.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training across multiple GPUs.")
    args = parser.parse_args()

    # distributed 모드일 경우, torchrun 또는 기타 분산 런처에 의해 LOCAL_RANK가 설정되어 있으면 우선 사용
    if args.distributed:
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            local_rank = args.local_rank
    else:
        local_rank = 0  # 단일 GPU 모드에서는 0번 GPU 사용

    train(local_rank, args, distributed=args.distributed)

# 실행 예시:
# 분산 모드: torchrun --nproc_per_node=2 train.py --distributed
# 단일 GPU 모드: python train.py
