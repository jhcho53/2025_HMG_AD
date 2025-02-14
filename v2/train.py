import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# (기존의 import 구문 유지)
from models.encoder import Encoder, HDMapFeaturePipeline, FeatureEmbedding, TrafficLightEncoder
from models.decoder import TrafficSignClassificationHead, EgoStateHead, Decoder
from models.GRU import BEVGRU, EgoStateGRU 
from models.backbones.efficientnet import EfficientNetExtractor
from models.control import FutureControlMLP, ControlMLP
from utils.attention import FeatureFusionAttention
from utils.utils import BEV_Ego_Fusion
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
        image_h = config.get("image_h", 270)
        image_w = config.get("image_w", 480)
        bev_h = config.get("bev_h", 200)
        bev_w = config.get("bev_w", 200)
        bev_h_meters = config.get("bev_h_meters", 50)
        bev_w_meters = config.get("bev_w_meters", 50)
        bev_offset = config.get("bev_offset", 0)
        decoder_blocks = config.get("decoder_blocks", [128, 128, 64])

        # Backbone 초기화
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
        
        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": bev_h,
            "bev_width": bev_w,
            "h_meters": bev_h_meters,
            "w_meters": bev_w_meters,
            "offset": bev_offset,
            "decoder_blocks": decoder_blocks,
        }
        
        self.encoder = Encoder(
            backbone=self.backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128,
            scale=1.0,
            middle=[2, 2],
        )
        
        input_dim = 256
        hidden_dim = 256
        output_dim = 256
        height, width = 25, 25
        self.bev_gru = BEVGRU(input_dim, hidden_dim, output_dim, height, width)
        
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.ego_gru = EgoStateGRU(input_dim=176, hidden_dim=256, output_dim=256, num_layers=1)
        self.ego_fusion = BEV_Ego_Fusion()

        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=7, final_channels=128, final_size=(25, 25))
        self.traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True)
        self.classification_head = TrafficSignClassificationHead(input_dim=128, num_classes=10)
        self.control = ControlMLP(future_steps=2, control_dim=3)
        self.ego_header = EgoStateHead(input_dim=256, hidden_dim=128, output_dim=12)
        self.bev_decoder = Decoder(dim=128, blocks=decoder_blocks, residual=True, factor=2)
    
    def forward(self, batch):
        hd_features = self.hd_map_pipeline(batch["hd_map"])
        ego_embedding = self.feature_embedding(batch["ego_info"])
        bev_output = self.encoder(batch)
        fusion_ego = self.ego_fusion(bev_output, ego_embedding)
        ego_gru_output, ego_gru_output_2 = self.ego_gru(fusion_ego)
        future_ego = self.ego_header(ego_gru_output)
        bev_decoding = self.bev_decoder(bev_output)
        concat_bev = torch.cat([hd_features, bev_output], dim=2)
        _, gru_bev = self.bev_gru(concat_bev)
        front_feature = self.traffic_encoder(batch["image"])
        classification_output = self.classification_head(front_feature)
        control_output = self.control(front_feature, gru_bev, ego_gru_output_2, batch["ego_info"])
        return {
            "control": control_output,
            "classification": classification_output,
            "future_ego": future_ego,
            "bev_seg": bev_decoding
        }


def get_dataloader(dataset, batch_size=16, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))


def validate_model(model, dataloader, device, classification_loss_fn, control_loss_fn, ego_loss_fn, seg_loss_fn, logger):
    """
    validation 단계에서 모델의 성능을 평가하는 함수.
    """
    model.eval()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_control_loss = 0.0
    total_ego_loss = 0.0
    total_seg_loss = 0.0
    count = 0

    with torch.no_grad():
        for data in dataloader:
            batch = {
                "image": data["images_input"].to(device),
                "intrinsics": data["intrinsic_input"].to(device),
                "extrinsics": data["extrinsic_input"].to(device),
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }
            ego_info_future_gt = data["ego_info_future"].to(device)
            bev_seg_gt = data["gt_hd_map_input"].to(device)
            traffic_gt = data["traffic"].to(device)
            traffic_gt_indices = torch.argmax(traffic_gt, dim=1)
            control_gt = data["control"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(batch)
                control_pred = outputs["control"]
                classification_pred = outputs["classification"]
                future_ego_pred = outputs["future_ego"]
                bev_seg_pred = outputs["bev_seg"]

                loss_classification = classification_loss_fn(classification_pred, traffic_gt_indices)
                loss_control = control_loss_fn(control_pred, control_gt)
                loss_ego = ego_loss_fn(future_ego_pred, ego_info_future_gt)
                loss_seg = seg_loss_fn(bev_seg_pred, bev_seg_gt)
                loss = loss_classification + loss_control + loss_ego + loss_seg

            total_loss += loss.item()
            total_classification_loss += loss_classification.item()
            total_control_loss += loss_control.item()
            total_ego_loss += loss_ego.item()
            total_seg_loss += loss_seg.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else 0
    avg_classification_loss = total_classification_loss / count if count > 0 else 0
    avg_control_loss = total_control_loss / count if count > 0 else 0
    avg_ego_loss = total_ego_loss / count if count > 0 else 0
    avg_seg_loss = total_seg_loss / count if count > 0 else 0

    logger.info(f"Validation - Avg Total Loss: {avg_loss:.4f}, Classification: {avg_classification_loss:.4f}, "
                f"Control: {avg_control_loss:.4f}, Ego: {avg_ego_loss:.4f}, BEV Seg: {avg_seg_loss:.4f}")
    return avg_loss, avg_classification_loss, avg_control_loss, avg_ego_loss, avg_seg_loss


def train(local_rank, args, distributed=False):
    # 분산 환경 설정
    if distributed:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    # logger 설정 (rank 0에서만 파일에 기록)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    if rank == 0:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # 파일 핸들러
        fh = logging.FileHandler("training.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info(f"Using {'distributed' if distributed else 'single GPU'} training with {world_size} process(es) on device: {device}")
    else:
        # rank 0가 아닌 프로세스는 별도의 핸들러를 추가하지 않습니다.
        logger.addHandler(logging.NullHandler())

    config = {
        "image_h": 270,
        "image_w": 480,
        "bev_h": 200,
        "bev_w": 200,
        "bev_h_meters": 50,
        "bev_w_meters": 50,
        "bev_offset": 0,
        "decoder_blocks": [128, 128, 64],
    }
    
    model = EndToEndModel(config).to(device)
    
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # 분산 환경에서는 SyncBatchNorm으로 변환하여 각 GPU의 배치 정규화 통계를 동기화합니다.
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, gradient_as_bucket_view=False)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    classification_loss_fn = nn.CrossEntropyLoss()
    control_loss_fn = nn.MSELoss()
    ego_loss_fn = nn.MSELoss()
    seg_loss_fn = nn.MSELoss()
    
    # AMP: GradScaler와 autocast 사용
    scaler = torch.amp.GradScaler(device="cuda")
    root_dir = "/home/vip/hd/Dataset"
    # root_dir = "/home/vip/2025_HMG_AD/v2/Dataset_sample"  # 실제 데이터셋 경로로 수정
    dataset = camDataLoader(root_dir, num_timesteps=3)
    
    # 학습 데이터와 검증 데이터를 80:20 비율로 분할합니다.
    total_samples = len(dataset)
    train_samples = int(0.8 * total_samples)
    val_samples = total_samples - train_samples
    train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=4, sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 예시로 num_epochs를 50으로 설정 (필요에 따라 조정)
    num_epochs = 50
    model.train()
    
    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_classification = 0.0
        running_loss_control = 0.0
        running_loss_ego = 0.0
        running_loss_seg = 0.0

        for batch_idx, data in enumerate(train_loader):
            if data is None:
                logger.warning(f"Warning: Skipping batch {batch_idx} due to empty data.")
                break

            batch = {
                "image": data["images_input"].to(device),
                "intrinsics": data["intrinsic_input"].to(device),
                "extrinsics": data["extrinsic_input"].to(device),
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }
            ego_info_future_gt = data["ego_info_future"].to(device)
            bev_seg_gt = data["gt_hd_map_input"].to(device)
            traffic_gt = data["traffic"].to(device)
            traffic_gt_indices = torch.argmax(traffic_gt, dim=1)
            control_gt = data["control"].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(batch)
                control_pred = outputs["control"]
                classification_pred = outputs["classification"]
                future_ego_pred = outputs["future_ego"]
                bev_seg_pred = outputs["bev_seg"]
                
                loss_classification = classification_loss_fn(classification_pred, traffic_gt_indices)
                loss_control = control_loss_fn(control_pred, control_gt)
                loss_ego = ego_loss_fn(future_ego_pred, ego_info_future_gt)
                loss_seg = seg_loss_fn(bev_seg_pred, bev_seg_gt)
                loss = loss_classification + loss_control + loss_ego + loss_seg
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            running_loss_classification += loss_classification.item()
            running_loss_control += loss_control.item()
            running_loss_ego += loss_ego.item()
            running_loss_seg += loss_seg.item()
            
            if batch_idx % 10 == 0 and rank == 0:
                logger.info(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx} - "
                            f"Classification Loss: {loss_classification.item():.4f}, "
                            f"Control Loss: {loss_control.item():.4f}, "
                            f"Ego State Loss: {loss_ego.item():.4f}, "
                            f"BEV Segmentation Loss: {loss_seg.item():.4f}, "
                            f"Total Loss: {loss.item():.4f}")
        
        avg_epoch_loss = running_loss / len(train_loader)
        avg_epoch_class_loss = running_loss_classification / len(train_loader)
        avg_epoch_control_loss = running_loss_control / len(train_loader)
        avg_epoch_ego_loss = running_loss_ego / len(train_loader)
        avg_epoch_seg_loss = running_loss_seg / len(train_loader)

        if rank == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Training Summary:")
            logger.info(f"  Avg Total Loss           : {avg_epoch_loss:.4f}")
            logger.info(f"  Avg Classification Loss  : {avg_epoch_class_loss:.4f}")
            logger.info(f"  Avg Control Loss         : {avg_epoch_control_loss:.4f}")
            logger.info(f"  Avg Ego State Loss       : {avg_epoch_ego_loss:.4f}")
            logger.info(f"  Avg BEV Segmentation Loss: {avg_epoch_seg_loss:.4f}")
        
        # 10 epoch마다 검증 단계 수행
        if (epoch + 1) % 10 == 0 and rank == 0:
            logger.info("Running validation...")
            validate_model(model, val_loader, device, classification_loss_fn, control_loss_fn, ego_loss_fn, seg_loss_fn, logger)
            # 검증 후 다시 학습 모드로 전환
            model.train()
    
    if rank == 0:
        torch.save(model.module.state_dict() if distributed else model.state_dict(), "end_to_end_model_test.pth")
        logger.info("Training complete and model saved.")
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Provided by distributed launcher if using distributed training.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training across multiple GPUs.")
    args = parser.parse_args()

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        local_rank = 0

    train(local_rank, args, distributed=args.distributed)

# 분산 모드: torchrun --nproc_per_node=2 train.py --distributed
# 단일 GPU 모드: python train.py