import torch
import torch.nn as nn

from models.backbones.efficientnet import EfficientNetExtractor
from models.encoder import SequenceEncoder, SingleFrameEncoder, TemporalTransformer
from models.BEVHDmapFusionNet import BEVHDMapFusionNet
from models.decoder import BEVDecoderWithStackedGRU

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # (1) Backbone & Config
        backbone = EfficientNetExtractor(
            layer_names=['reduction_2', 'reduction_4'],
            image_height=224,
            image_width=480,
            model_name='efficientnet-b4'
        )

        cross_view_config = {
            "heads": 4,
            "dim_head": 32,
            "qkv_bias": True,
            "skip": True,
            "no_image_features": False,
            "image_height": 224,
            "image_width": 480,
        }

        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": 200,
            "bev_width": 200,
            "h_meters": 100.0,
            "w_meters": 100.0,
            "offset": 0.0,
            "decoder_blocks": [2, 2],
        }

        # (2) Encoders
        single_frame_encoder = SingleFrameEncoder(
            backbone=backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128
        )
        temporal_transformer = TemporalTransformer(d_model=128, nhead=8, num_layers=2)

        # 시퀀스 인코더 (BEV Feature 추출)
        self.sequence_encoder = SequenceEncoder(
            base_encoder=single_frame_encoder,
            temporal_module=temporal_transformer
        )

        # (3) Fusion & Decoder
        self.bev_fusion_net = BEVHDMapFusionNet(
            bev_dim=128,
            hd_map_dim=6,
            ego_dim=19,
            fused_dim=64,
            output_dim=32
        )
        self.bev_decoder = BEVDecoderWithStackedGRU(
            input_channels=2,
            hidden_size=128,
            seq_len=32,
            spatial_dim=50,
            num_layers=3
        )

    def forward(self,
                camera_images: torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor,
                hd_map_tensors: torch.Tensor,
                ego_inputs: torch.Tensor):
        """
        Args:
            camera_images :  (B, T, N_cam, C, H, W)
            intrinsics    :  (B, T, N_cam, 3, 3)
            extrinsics    :  (B, T, N_cam, 4, 4)
            hd_map_tensors:  (B, T, C_map, H_map, W_map)
            ego_inputs    :  (B, T, ego_dim)

        Returns:
            output: (B, Control)
        """
        # 1) 시퀀스 인코더로부터 BEV Feature 추출
        batch_dict = {
            "image": camera_images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics
        }
        bev_features = self.sequence_encoder(batch_dict)  # (B, T, D, H, W)

        # 2) Fusion (BEV Feature + HD Map + Ego Info)
        fusion_output = self.bev_fusion_net(bev_features, hd_map_tensors, ego_inputs)  # (B, T, out_dim, H, W)

        # 3) Decoder
        fusion_output = fusion_output.permute(0, 2, 1, 3, 4)  # (B, out_dim, T, H, W)
        output = self.bev_decoder(fusion_output)              # (B, Control)

        return output
