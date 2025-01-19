import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------
# Cross-Attention 모듈 정의
# -------------------------
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.to_query = nn.Linear(query_dim, num_heads * head_dim, bias=False)
        self.to_key = nn.Linear(key_dim, num_heads * head_dim, bias=False)
        self.to_value = nn.Linear(key_dim, num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, query_dim)

    def forward(self, query, key, value, spatial_size_query, spatial_size_key):
        """
        query: (B, T, D, H, W) -> BEV Feature
        key, value: (B, T, C, H_map, W_map) -> HD Map
        spatial_size_query: (H, W) -> Query의 공간 크기
        spatial_size_key: (H_map, W_map) -> Key의 공간 크기
        """
        B, T, D, H, W = query.shape
        _, _, C, H_map, W_map = key.shape

        # Flatten spatial dimensions
        query = rearrange(query, 'b t d h w -> b t (h w) d')
        key = rearrange(key, 'b t c h w -> b t (h w) c')
        value = rearrange(value, 'b t c h w -> b t (h w) c')

        # Project query, key, value
        Q = self.to_query(query)  # (B, T, HW, num_heads * head_dim)
        K = self.to_key(key)      # (B, T, H_map*W_map, num_heads * head_dim)
        V = self.to_value(value)  # (B, T, H_map*W_map, num_heads * head_dim)

        # Reshape for multi-head attention
        Q = rearrange(Q, 'b t n (h d) -> b t h n d', h=self.num_heads)
        K = rearrange(K, 'b t n (h d) -> b t h n d', h=self.num_heads)
        V = rearrange(V, 'b t n (h d) -> b t h n d', h=self.num_heads)

        # Compute scaled dot-product attention
        attn_scores = torch.einsum('b t h q d, b t h k d -> b t h q k', Q, K) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.einsum('b t h q k, b t h k d -> b t h q d', attn_probs, V)
        attn_output = rearrange(attn_output, 'b t h n d -> b t n (h d)')

        # Final projection
        fused_output = self.to_out(attn_output)  # (B, T, HW, D)

        # Reshape back to (B, T, D, H, W)
        fused_output = rearrange(fused_output, 'b t (h w) d -> b t d h w', h=spatial_size_query[0], w=spatial_size_query[1])
        return fused_output

# -------------------------
# BEVHDMapFusionNet 정의: BEV + HD Map + Ego 정보
# -------------------------
class BEVHDMapFusionNet(nn.Module):
    def __init__(self, bev_dim, hd_map_dim, ego_dim, fused_dim, output_dim):
        super().__init__()
        self.bev_conv = nn.Sequential(nn.Conv2d(bev_dim + ego_dim, fused_dim, kernel_size=3, padding=1), nn.ReLU())
        self.hd_map_conv = nn.Sequential(nn.Conv2d(hd_map_dim, fused_dim, kernel_size=3, padding=1), nn.ReLU())
        self.cross_attention = CrossAttention(query_dim=fused_dim, key_dim=fused_dim)
        # Cross-Attention 이후 Ego 정보 결합을 고려하여 입력 채널에 ego_dim 추가
        self.output_layer = nn.Sequential(nn.Conv2d(fused_dim + ego_dim, output_dim, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, bev, hd_map, ego_info):
        B, T, D, H, W = bev.shape
        _, _, C, H_map, W_map = hd_map.shape

        # (1) 초기 단계에서 Ego 정보를 BEV Feature와 결합
        ego_info_expanded = ego_info.unsqueeze(-1).unsqueeze(-1)  # (B, T, ego_dim) -> (B, T, ego_dim, 1, 1)
        ego_info_expanded = ego_info_expanded.expand(-1, -1, -1, H, W)  # (B, T, ego_dim, H, W)
        bev_with_ego = torch.cat([bev, ego_info_expanded], dim=2)  # (B, T, D + ego_dim, H, W)

        # (2) Backbone feature extraction
        bev_with_ego = rearrange(bev_with_ego, 'b t d h w -> (b t) d h w')
        hd_map = rearrange(hd_map, 'b t c h w -> (b t) c h w')

        bev_features = self.bev_conv(bev_with_ego)  # (B*T, fused_dim, H, W)
        hd_map_features = self.hd_map_conv(hd_map)  # (B*T, fused_dim, H_map, W_map)

        bev_features = rearrange(bev_features, '(b t) d h w -> b t d h w', b=B, t=T)
        hd_map_features = rearrange(hd_map_features, '(b t) d h w -> b t d h w', b=B, t=T)

        # (3) Cross-Attention
        fused_features = self.cross_attention(bev_features, hd_map_features, hd_map_features, spatial_size_query=(H, W), spatial_size_key=(H_map, W_map))

        # (4) Cross-Attention 이후 Ego 정보를 추가
        ego_info_expanded_after = ego_info.unsqueeze(-1).unsqueeze(-1)  # (B, T, ego_dim) -> (B, T, ego_dim, 1, 1)
        ego_info_expanded_after = ego_info_expanded_after.expand(-1, -1, -1, H, W)  # (B, T, ego_dim, H, W)
        fused_with_ego = torch.cat([fused_features, ego_info_expanded_after], dim=2)  # (B, T, fused_dim + ego_dim, H, W)

        # (5) Output Layer
        output = self.output_layer(rearrange(fused_with_ego, 'b t d h w -> (b t) d h w'))  # (B*T, output_dim, H, W)
        return rearrange(output, '(b t) d h w -> b t d h w', b=B, t=T)

