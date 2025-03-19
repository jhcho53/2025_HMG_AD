import torch
import torch.nn as nn
from models.utils.planner import CustomTransformerDecoder, EgoFutDecoder

class PlannerModule(nn.Module):
    """
    BEV feature, ego_state, gt_ego_cmd 3개의 입력을 받아 Ego 미래 trajectory를 예측하는 모듈.
    (고수준 명령을 이용한 query 보정은 제거되었습니다.)
    
    Args:
        embed_dims (int): 임베딩 차원 (default: 256)
        num_reg_fcs (int): 회귀를 위한 FC layer 개수 (default: 2)
        ego_fut_mode (int): 미래 모드 수 (default: 3)
        fut_steps (int): 미래 예측 스텝 수 (default: 2)
    """
    def __init__(self, embed_dims=256, num_reg_fcs=2, ego_fut_mode=3, fut_steps=2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.ego_fut_mode = ego_fut_mode
        self.fut_steps = fut_steps

        # Query embedding (단일 learnable query)
        self.query_feat_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embed_dims)
        # Transformer decoder: bev feature (이미지 컨텍스트)를 활용
        self.ego_img_decoder = CustomTransformerDecoder()
        # Ego 미래 trajectory 디코더: ego 정보와 추가 ego_state 정보를 결합하여 미래 궤적 예측
        self.ego_fut_decoder = EgoFutDecoder(
            embed_dims=embed_dims, 
            num_reg_fcs=num_reg_fcs, 
            fut_steps=fut_steps, 
        )

    def forward(self, bev_feature, ego_state):
        """
        Args:
            bev_feature: Tensor of shape (B, T, C, H, W)
            ego_state: Tensor of shape (B, C_e)  (예: ego 상태 정보, 여기서는 C_e=12)
            gt_ego_cmd: Tensor of shape (B, 3)  (원래 고수준 명령 정보; 현재는 사용되지 않음)
        Returns:
            outputs_ego_trajs: Tensor of shape (B, ego_fut_mode * fut_steps * 2)
            outputs_ego_trajs_2: Tensor of shape (B, fut_steps, 5)
        """
        B, T, C, H, W = bev_feature.shape
        # bev feature의 첫 번째 타임스텝에서 flatten 및 차원 변환 → (B, H*W, C)
        img_context = bev_feature[:, 0].flatten(-2, -1).permute(0, 2, 1)
        
        # Query 생성: 단일 query embedding을 배치 크기만큼 복제 후 unsqueeze → (B, 1, embed_dims)
        ego_query = self.query_feat_embedding.weight.repeat(B, 1).unsqueeze(1)
        
        # Transformer 디코더: tgt는 ego_query, memory는 이미지 컨텍스트
        ego_query_final = self.ego_img_decoder(
            tgt=ego_query,
            memory=img_context
        )  # (B, 1, embed_dims)
        
        # ego_state (예: 12차원)와 결합: 최종 디코더 입력 생성 → (B, embed_dims+12)
        ego_query_cat = torch.cat([ego_query_final[:, 0], ego_state], dim=-1)
        
        # Ego 미래 trajectory 디코더를 통해 미래 궤적 예측
        outputs_ego_trajs = self.ego_fut_decoder(ego_query_cat)
        outputs_ego_trajs_2 = outputs_ego_trajs.reshape(B, self.fut_steps, 5)   # [B, fut_steps, 5]
        
        return outputs_ego_trajs, outputs_ego_trajs_2

# 아래는 위 모듈을 dummy input으로 테스트하는 예시입니다.
def main():
    # 하이퍼파라미터 설정
    embed_dims = 256
    num_reg_fcs = 2
    ego_fut_mode = 3
    fut_steps = 2

    # dummy bev feature: (B, T, C, H, W)
    B, T, C, H, W = 2, 3, 256, 25, 25  
    bev_feature = torch.randn(B, T, C, H, W)
    
    # dummy ego_state (예: ego 상태, 채널 수 12)
    ego_state = torch.randn(B, 12)
    
    # PlannerModule 인스턴스 생성 및 forward 수행
    planner = PlannerModule(embed_dims=embed_dims, num_reg_fcs=num_reg_fcs, 
                            ego_fut_mode=ego_fut_mode, fut_steps=fut_steps)
    
    outputs_ego_trajs, outputs_ego_trajs_2 = planner(bev_feature, ego_state)
    
    # 각 텐서의 shape 출력
    print("bev_feature shape:", bev_feature.shape)              # (B, T, C, H, W)
    print("ego_state shape:", ego_state.shape)                  # (B, 12)
    print("gt_ego_cmd shape:", gt_ego_cmd.shape)                # (B, 3)
    print("outputs_ego_trajs shape:", outputs_ego_trajs.shape)    # (B, ego_fut_mode * fut_steps * 2)
    print("outputs_ego_trajs_2 shape:", outputs_ego_trajs_2.shape)  # (B, fut_steps, 5)

if __name__ == "__main__":
    main()
