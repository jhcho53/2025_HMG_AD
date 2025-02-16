import os
import argparse
import logging
import math  # cosine annealing 스케줄러에 필요
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # tqdm import 추가

# 경고 무시 (필요한 경우)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", message="Detected call of lr_scheduler.step() before optimizer.step()")

# (기존의 import 구문 유지)
from models.encoder import Encoder, HDMapFeaturePipeline, FeatureEmbedding, TrafficLightEncoder
from models.decoder import TrafficSignClassificationHead, EgoStateHead, Decoder
from models.GRU import BEVGRU, EgoStateGRU 
from models.backbones.efficientnet import EfficientNetExtractor
from models.control import FutureControlMLP, ControlMLP
from utils.attention import FeatureFusionAttention
from utils.utils import BEV_Ego_Fusion
from dataloader.dataloader import camDataLoader
from models.decoder_v2 import PlannerModule

# 메모리 단편화 문제를 완화하기 위한 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class EndToEndModel(nn.Module):
    """
    End-to-End 모델 클래스.
    """
    def __init__(self, config):
        super(EndToEndModel, self).__init__()
        image_h = config.get("image_h", 270)
        image_w = config.get("image_w", 480)
        bev_h = config.get("bev_h", 200)
        bev_w = config.get("bev_w", 200)
        bev_h_meters = config.get("bev_h_meters", 50)
        bev_w_meters = config.get("bev_w_meters", 50)
        bev_offset = config.get("bev_offset", 0)
        decoder_blocks = config.get("decoder_blocks", [128, 128, 64])

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
        
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=7, final_channels=128, final_size=(25, 25))
        self.bev_decoder = Decoder(dim=128, blocks=decoder_blocks, residual=True, factor=2)
        self.decoder = PlannerModule(embed_dims=256, num_reg_fcs=2, 
                                     ego_fut_mode=3, fut_steps=2)
    
    def forward(self, batch):
        # HD Map Encoding
        hd_features = self.hd_map_pipeline(batch["hd_map"])  
        # Ego Encoding
        ego_embedding = self.feature_embedding(batch["ego_info"])
        # BEV Encoding
        bev_output = self.encoder(batch)
        # BEV Decoding
        bev_decoding = self.bev_decoder(bev_output)
        # HD map feature와 BEV feature를 채널 차원에서 concat
        concat_bev = torch.cat([hd_features, bev_output], dim=2)
        # PlannerModule을 통해 control 예측 (planner shape: [B, T, 5])
        _, planner = self.decoder(concat_bev, batch["ego_info"][:, -1, :])
        return {
            "control": planner,      # control 예측 결과
            "bev_seg": bev_decoding   # BEV segmentation 결과
        }

def get_dataloader(dataset, batch_size=16, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))

def validate_model(model, dataloader, device, control_loss_fn, seg_loss_fn, logger, val_logger=None, control_logger=None, max_batches=100):
    """
    Validation: control과 segmentation loss를 계산하고, validation 데이터 중 첫 sample의 control 예측 결과를 control_logger에 기록합니다.
    전체 데이터셋 대신, 지정된 배치 수(max_batches)만 처리합니다.
    """
    model.eval()
    total_loss = 0.0
    total_control_loss = 0.0
    total_seg_loss = 0.0
    count = 0
    first_sample_logged = False

    with torch.no_grad():
        # validation loop에도 tqdm 적용 (진행률 표시)
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, data in enumerate(pbar):
            # 지정한 배치 수만 validation 진행
            if batch_idx >= max_batches:
                break

            batch = {
                "image": data["images_input"].to(device),
                "intrinsics": data["intrinsic_input"].to(device),
                "extrinsics": data["extrinsic_input"].to(device),
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }
            bev_seg_gt = data["gt_hd_map_input"].to(device)
            control_gt = data["control"].to(device)

            # positional 인자로 "cuda" 사용 (device_type 인자 대신)
            with torch.amp.autocast("cuda"):
                outputs = model(batch)
                control_pred = outputs["control"]
                bev_seg_pred = outputs["bev_seg"]

                loss_control = control_loss_fn(control_pred, control_gt)
                loss_seg = seg_loss_fn(bev_seg_pred, torch.argmax(bev_seg_gt, dim=1))
                loss = loss_control + loss_seg

            total_loss += loss.item()
            total_control_loss += loss_control.item()
            total_seg_loss += loss_seg.item()
            count += 1

            # validation의 첫 sample에 대해 control 예측 결과 기록 (한 sample만 기록)
            if not first_sample_logged and control_logger is not None:
                control_logger.info(
                    f"Validation Sample - Control Prediction: {control_pred.detach().cpu().tolist()}, "
                    f"Ground Truth: {control_gt.detach().cpu().tolist()}"
                )
                first_sample_logged = True

    avg_loss = total_loss / count if count > 0 else 0
    avg_control_loss = total_control_loss / count if count > 0 else 0
    avg_seg_loss = total_seg_loss / count if count > 0 else 0

    log_msg = (f"Validation - Avg Total Loss: {avg_loss:.4f}, "
               f"Control: {avg_control_loss:.4f}, "
               f"BEV Seg: {avg_seg_loss:.4f}")
    logger.info(log_msg)
    if val_logger is not None:
        val_logger.info(log_msg)
    return avg_loss, avg_control_loss, avg_seg_loss

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

    # logger 설정 (rank 0에서만 기록)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if rank == 0:
        fh = logging.FileHandler("training.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info(f"Using {'distributed' if distributed else 'single GPU'} training with {world_size} process(es) on device: {device}")
        
        # validation logger
        val_logger = logging.getLogger("val_logger")
        val_logger.setLevel(logging.INFO)
        val_fh = logging.FileHandler("validation.log")
        val_fh.setLevel(logging.INFO)
        val_fh.setFormatter(formatter)
        val_logger.addHandler(val_fh)
        
        # control 전용 logger (control.log)
        control_logger = logging.getLogger("control_logger")
        control_logger.setLevel(logging.INFO)
        control_fh = logging.FileHandler("control.log")
        control_fh.setLevel(logging.INFO)
        control_fh.setFormatter(formatter)
        control_logger.addHandler(control_fh)
    else:
        logger.addHandler(logging.NullHandler())
        val_logger = None
        control_logger = None

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

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            gradient_as_bucket_view=False, find_unused_parameters=True
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    control_loss_fn = nn.MSELoss()
    seg_loss_fn = nn.CrossEntropyLoss()
    
    scaler = torch.amp.GradScaler(device="cuda")
    root_dir = "/workspace/dataset" 
    # root_dir = "/home/vip/hd/Dataset"
    dataset = camDataLoader(root_dir, num_timesteps=3)
    
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
    
    # hyperparameter: 총 epoch 수, iteration 단위 validation interval
    num_epochs = getattr(args, "num_epochs", 20)
    validation_interval = getattr(args, "validation_interval", 100)  # 기본 100 iteration마다 validation
    best_val_loss = float("inf")
    early_stop_counter = 0
    early_stop_patience = args.early_stop_patience

    # 총 iteration 수 및 warmup iteration 설정
    total_iterations = num_epochs * len(train_loader)
    warmup_iterations = 300  # warmup iteration 수
    cosine_iterations = total_iterations - warmup_iterations

    # warmup: learning rate를 매우 작은 값(1e-3배)에서 시작하여 1.0 배율까지 선형 증가
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iterations
    )
    # cosine annealing: warmup 이후 cosine annealing 적용
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_iterations, eta_min=1e-6
    )
    # 두 scheduler를 순차적으로 적용
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iterations]
    )
    
    model.train()
    iteration = 0
    stop_training = False

    first_step = True 

    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        # tqdm을 사용하여 epoch 내의 배치 진행률 표시
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        running_loss_control = 0.0
        running_loss_seg = 0.0

        for batch_idx, data in pbar:
            if data is None:
                logger.warning(f"Warning: Skipping batch {batch_idx} due to empty data.")
                continue

            batch = {
                "image": data["images_input"].to(device),
                "intrinsics": data["intrinsic_input"].to(device),
                "extrinsics": data["extrinsic_input"].to(device),
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),
            }
            bev_seg_gt = data["gt_hd_map_input"].to(device)
            control_gt = data["control"].to(device)
            gt_indices = torch.argmax(bev_seg_gt, dim=1)
            
            optimizer.zero_grad()
            # "cuda" positional 인자로 autocast 사용
            with torch.amp.autocast("cuda"):
                outputs = model(batch)
                control_pred = outputs["control"]
                bev_seg_pred = outputs["bev_seg"]

                loss_control = control_loss_fn(control_pred, control_gt)
                loss_seg = seg_loss_fn(bev_seg_pred, gt_indices)
                loss = loss_control + loss_seg
            
            # optimizer.step() 후 scheduler.step() 순서 (경고 해결)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if not first_step: 
                scheduler.step()
            
            first_step = False
            
            running_loss += loss.item()
            running_loss_control += loss_control.item()
            running_loss_seg += loss_seg.item()
            
            # tqdm progress bar 업데이트 (현재 배치 loss 표시)
            pbar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "Control Loss": f"{loss_control.item():.4f}",
                "BEV Seg Loss": f"{loss_seg.item():.4f}"
            })
            
            # 10 배치마다 loss 정보를 training.log에 기록
            if batch_idx % 10 == 0 and rank == 0:
                logger.info(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx} - "
                            f"Control Loss: {loss_control.item():.4f}, "
                            f"BEV Seg Loss: {loss_seg.item():.4f}, "
                            f"Total Loss: {loss.item():.4f}")
                # control 관련 정보는 control 전용 로그에 기록
                control_logger.info(f"Training [Epoch {epoch+1}/{num_epochs}] Batch {batch_idx} - "
                                      f"Control Prediction: {control_pred.detach().cpu().tolist()}, "
                                      f"Ground Truth: {control_gt.detach().cpu().tolist()}")
            
            iteration += 1
            
            # iteration 단위 validation 수행 (일부 배치만 사용)
            if rank == 0 and iteration % validation_interval == 0:
                logger.info(f"Iteration {iteration}: Running validation...")
                val_loss, avg_control_loss, avg_seg_loss = validate_model(
                    model, val_loader, device, control_loss_fn, seg_loss_fn, logger, val_logger, control_logger, max_batches=10
                )
                model.train()  # validation 후 train 모드로 전환

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    logger.info(f"Validation loss improved to {val_loss:.4f}.")
                else:
                    early_stop_counter += 1
                    logger.info(f"No improvement in validation loss for {early_stop_counter} iteration(s).")
                    if early_stop_counter >= early_stop_patience:
                        logger.info(f"Early stopping triggered at iteration {iteration}.")
                        stop_training = True
                        break  # inner loop 종료

        # epoch 종료 후 모델 저장 (매 epoch마다)
        if rank == 0:
            model_state = model.module.state_dict() if distributed else model.state_dict()
            save_path = f"end_to_end_model_epoch_{epoch+1}.pth"
            torch.save(model_state, save_path)
            logger.info(f"Epoch {epoch+1} complete. Model saved to {save_path}.")

        if stop_training:
            break

        avg_epoch_loss = running_loss / len(train_loader)
        avg_epoch_control_loss = running_loss_control / len(train_loader)
        avg_epoch_seg_loss = running_loss_seg / len(train_loader)

        if rank == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] Training Summary:")
            logger.info(f"  Avg Total Loss           : {avg_epoch_loss:.4f}")
            logger.info(f"  Avg Control Loss         : {avg_epoch_control_loss:.4f}")
            logger.info(f"  Avg BEV Seg Loss         : {avg_epoch_seg_loss:.4f}")

    if rank == 0 and not stop_training:
        # 최종 모델 저장
        save_dict = model.module.state_dict() if distributed else model.state_dict()
        torch.save(save_dict, "end_to_end_model_test.pth")
        logger.info("Training complete and final model saved.")
    
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Provided by distributed launcher if using distributed training.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training across multiple GPUs.")
    parser.add_argument("--early_stop_patience", type=int, default=4,
                        help="Early stopping patience (number of validations with no improvement).")
    parser.add_argument("--validation_interval", type=int, default=300,
                        help="Run validation every N iterations (default: 100).")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs.")
    args = parser.parse_args()

    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    else:
        local_rank = 0

    train(local_rank, args, distributed=args.distributed)

# 분산 모드: torchrun --nproc_per_node=2 train.py --distributed
# 단일 GPU 모드: python train.py
