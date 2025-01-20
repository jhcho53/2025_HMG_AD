import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List

from models.backbones.efficientnet import EfficientNetExtractor

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int):
    """
    height x width 크기의 정규화(normalized) grid 생성.
    (x,y)는 [0,1] 범위.
    반환 shape: (1, 3, height, width)
    """
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)
    # PyTorch 1.10+에서 meshgrid의 indexing='xy' 사용
    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)  # (2, h, w)
    # 채널 수를 3개로 맞추기 위해 z=1 padding
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)              # (3, h, w)
    # 배치 차원(=1) 추가
    indices = indices[None]                                           # (1, 3, h, w)
    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]

class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std

class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()
        linear = nn.Conv2d(*args, **kwargs)
        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))

class BEVEmbedding(nn.Module):
    """
    BEV 공간에 놓일 learnable embedding + grid 생성 로직
    """
    def __init__(
        self,
        dim: int,
        sigma: float,
        bev_height: int,
        bev_width: int,
        h_meters: float,
        w_meters: float,
        offset: float,
        decoder_blocks: list,
    ):
        """
        dim: BEV 임베딩 채널 수
        sigma: 임베딩 초기화 스케일
        bev_height, bev_width: 최종 BEV 해상도
        h_meters, w_meters: 실제 월드에서 몇 m 범위를 볼지
        offset: ego 차량을 어느 위치에 둘지 (0.0 ~ 1.0)
        decoder_blocks: 디코더 블록들의 업샘플 단계 수
                        (ex: [2,2]이면 2번 업샘플, 따라서 실제 초기 해상도는 bev_height/4)
        """
        super().__init__()

        # decoder_blocks의 개수만큼 2배 업샘플 되므로, 초기 해상도는 / (2**len(decoder_blocks))
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # (1, 3, h, w)
        grid = generate_grid(h, w).squeeze(0)  # (3, h, w)
        # grid의 x,y를 bev_width, bev_height 스케일로 변환
        grid[0] = bev_width  * grid[0]
        grid[1] = bev_height * grid[1]

        # V: (3,3) view matrix
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()             # (3,3)
        # grid(3,h,w)을 (3, h*w)로 펴서 매트릭스 곱
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)') # (3, h*w)
        # 다시 (3, h, w)로
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)

        # 모델 추론 시 활용할 grid
        self.register_buffer('grid', grid, persistent=False)
        # 학습 가능한 BEV 임베딩 (초기값: N(0, sigma^2))
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))

    def get_prior(self):
        """
        (d, h, w) 형태의 학습된 BEV 임베딩을 반환
        """
        return self.learned_features


class CrossAttention(nn.Module):
    """
    쿼리(q)에 대해 멀티헤드 Attention을 수행하되,
    k, v는 이미지 feature 및 카메라 좌표 등으로부터 얻음.
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        # q, k, v projection
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

        # 추가적인 MLP + NORM
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b, n, d, H, W)
        k: (b, n, d, h, w)
        v: (b, n, d, h, w)
        skip: skip connection용 (b, d, H, W) 또는 None
        """
        _, _, _, H, W = q.shape

        # (b, n, H*W, d)
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        # (b, n, h*w, d)
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        # (b, n*h*w, d)
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Projection
        q = self.to_q(q)  # (b, n*(H*W), heads*dim_head)
        k = self.to_k(k)
        v = self.to_v(v)

        # 헤드 수(heads)를 batch 차원으로 통합
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Attention: q x k^T
        # q,k shape: (b*m, n*(H*W), dim_head) (단, k의 n*(h*w) 축이 다름)
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        # (b, n, Q, K) -> (b, Q, n*K)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # (b, Q, dim_head)
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        # (b, Q, m*dim_head)
        a = rearrange(a, '(b m) Q d -> b Q (m d)', m=self.heads, d=self.dim_head)

        z = self.proj(a)
        if skip is not None:
            # skip: (b, d, H, W) -> (b, H*W, d)
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        # MLP + norm
        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)

        # (b, d, H, W)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)
        return z

class CrossViewAttention(nn.Module):
    """
    이미지 평면 좌표, 카메라 중심, feature map 등을 이용해
    BEV 공간(x)에 대한 CrossAttention을 수행하는 모듈.
    """
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # (1, 3, feat_height, feat_width)
        # [0,1] 범위 grid를 (image_width, image_height) 스케일로 변경
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        # feature map -> dim
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False)
        )
        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False)
            )

        # BEV grid(2D) -> dim
        self.bev_embed = nn.Conv2d(2, dim, 1)
        # image plane(4D) -> dim
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        # camera center(4D) -> dim
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, d, H_bev, W_bev) - 이전 BEV feature
        feature: (b, n, feat_dim, h, w) - 백본 추출 이미지 feature
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)
        return: (b, d, H_bev, W_bev)
        """
        b, n, _, _, _ = feature.shape
        pixel = self.image_plane  # (1, 3, h, w)
        _, _, _, h, w = pixel.shape

        # 카메라 중심(c)
        c = E_inv[..., -1:]                       # (b, n, 4, 1)
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b*n, 4, 1, 1)
        c_embed = self.cam_embed(c_flat)          # (b*n, d, 1, 1)

        # 이미지 평면 좌표(d)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')    # (1, 3, h*w)
        cam = I_inv @ pixel_flat                                 # (b, n, 3, h*w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)       # (b, n, 4, h*w)
        d = E_inv @ cam                                          # (b, n, 4, h*w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)

        d_embed = self.img_embed(d_flat)         # (b*n, d, h, w)
        img_embed = d_embed - c_embed            # (b*n, d, h, w)
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        # BEV Grid
        world = bev.grid[:2]                     # (2, H_bev, W_bev)
        w_embed = self.bev_embed(world[None])     # (1, d, H_bev, W_bev)
        bev_embed = w_embed - c_embed             # (b*n, d, H_bev, W_bev)
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        bev_embed = rearrange(bev_embed, '(b n) d H W -> b n d H W', b=b, n=n)

        # feature map reshape
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            key_flat = img_embed
        # print(f"feature_flat shape: {feature_flat.shape}")
        # feature_flat shape: torch.Size([10, 32, 56, 120])

        val_flat = self.feature_linear(feature_flat)

        # (b, n, d, H_bev, W_bev)
        query = bev_embed + x[:, None]
        key   = rearrange(key_flat, '(b n) d h w -> b n d h w', b=b, n=n)
        val   = rearrange(val_flat, '(b n) d h w -> b n d h w', b=b, n=n)

        return self.cross_attend(query, key, val, skip=x if self.skip else None)

class SingleFrameEncoder(nn.Module):
    """
    (B, N, C, H, W) -> (B, D, H_bev, W_bev)
    기존 Encoder 역할
    """
    def __init__(
            self,
            backbone: nn.Module,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()
        self.norm = Normalize()
        self.backbone = backbone

        # scale < 1.0 이면 feature 크기 downsample
        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        # BEV Embedding
        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)

        # CrossViewAttention & residual block 세트 구성
        cross_views = []
        layers = []

        # 백본에서 뽑히는 각 스케일(feat)의 shape 정보
        # -> down() 적용 후 (b*n, feat_dim, feat_h, feat_w)를 확인
        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            # feat_shape 예: (1, c, h, w)
            dummy = self.down(torch.zeros(feat_shape))  # (1, feat_dim, feat_height, feat_width)
            _, feat_dim, feat_height, feat_width = dummy.shape

            cva = CrossViewAttention(
                feat_height,  # down-sampled h
                feat_width,   # down-sampled w
                feat_dim,
                dim,
                **cross_view
            )
            cross_views.append(cva)

            # residual layer
            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        """
        batch['image']:      (B, N, C, H, W)
        batch['intrinsics']: (B, N, 3, 3)
        batch['extrinsics']: (B, N, 4, 4)
        """
        b, n, _, _, _ = batch['image'].shape

        # (B*N, C, H, W)
        image = batch['image'].flatten(0, 1)            # 카메라별 배치 합치기
        I_inv = batch['intrinsics'].inverse()           # (B, N, 3, 3)
        E_inv = batch['extrinsics'].inverse()           # (B, N, 4, 4)

        # 백본 통과 -> 멀티 스케일 feature 리스트
        feats = [self.down(y) for y in self.backbone(self.norm(image))]

        # 초기 BEV: (d, H_bev, W_bev)
        x = self.bev_embedding.get_prior()
        # 배치 차원 b만큼 반복: (b, d, H_bev, W_bev)
        x = repeat(x, '... -> b ...', b=b)

        # scale 별로 cross-view -> residual block
        for cross_view, feat, layer in zip(self.cross_views, feats, self.layers):
            # (b, n, feat_dim, h, w)
            feat = rearrange(feat, '(b n) d h w -> b n d h w', b=b, n=n)
            # cross-view attention
            x = cross_view(x, self.bev_embedding, feat, I_inv, E_inv)
            # residual bottleneck
            x = layer(x)

        return x

class TemporalTransformer(nn.Module):
    """
    (B, T, D, H, W) -> (B, T, D, H, W)
    시간축으로 Transformer Encoder를 적용
    """
    def __init__(self, d_model=128, nhead=8, num_layers=2):
        super().__init__()
        # PyTorch 기본 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False  # 아래 rearrange 시 (T, ...)가 먼저 오므로 False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B, T, D, H, W)
        return: (B, T, D, H, W)
        """
        B, T, D, H, W = x.shape
        # (B, T, D, H, W) -> (T, B*H*W, D)
        x = rearrange(x, 'b t d h w -> t (b h w) d')
        # Transformer
        x = self.transformer(x)  # (T, B*H*W, D)
        # 복원 -> (B, T, D, H, W)
        x = rearrange(x, 't (b h w) d -> b t d h w', b=B, h=H, w=W)
        return x



class SequenceEncoder(nn.Module):
    """
    (B, T, N, C, H, W) -> (B, T, D, H_bev, W_bev)

    - 시간축 T 전체를 한 번에 입력
    - 내부적으로 SingleFrameEncoder를 호출해 프레임별 BEV 추출
    - 이후 TemporalTransformer 등으로 시간축 후처리
    """
    def __init__(self, base_encoder: nn.Module, temporal_module: nn.Module):
        super().__init__()
        self.base_encoder = base_encoder       # SingleFrameEncoder
        self.temporal_module = temporal_module # 시간축 모듈

    def forward(self, batch):
        """
        batch:
          - 'image':      (B, T, N, C, H, W)
          - 'intrinsics': (B, T, N, 3, 3)
          - 'extrinsics': (B, T, N, 4, 4)

        return:
          - (B, T, D, H_bev, W_bev)
        """
        B, T, N, C, H, W = batch['image'].shape

        # 1) time dimension flatten -> (B*T, N, C, H, W)
        images = rearrange(batch['image'],      'b t n c hh ww -> (b t) n c hh ww')
        intrin = rearrange(batch['intrinsics'], 'b t n h2 w2   -> (b t) n h2 w2')
        extrin = rearrange(batch['extrinsics'], 'b t n h2 w2   -> (b t) n h2 w2')

        # SingleFrameEncoder 입력 준비
        single_frame_batch = {
            'image': images,       # (B*T, N, C, H, W)
            'intrinsics': intrin,  # (B*T, N, 3, 3)
            'extrinsics': extrin,  # (B*T, N, 4, 4)
        }

        # 2) 프레임별 BEV 추출 -> (B*T, D, H_bev, W_bev)
        bev_flat = self.base_encoder(single_frame_batch)

        # 3) (B, T, D, H_bev, W_bev) 로 reshape
        D, H_bev, W_bev = bev_flat.shape[1:]
        bev_seq = rearrange(bev_flat, '(b t) d hh ww -> b t d hh ww', b=B, t=T)

        # 4) 시간축 Transformer -> (B, T, D, H_bev, W_bev)
        fused_bev = self.temporal_module(bev_seq)
        return fused_bev