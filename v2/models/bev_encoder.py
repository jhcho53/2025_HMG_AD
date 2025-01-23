import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from utils.attention import CrossAttention, CrossViewAttention
from utils.utils import generate_grid, get_view_matrix, Normalize, RandomCos, BEVEmbedding

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, t, n, c, h, w = batch['image'].shape  # Batch, Time, Cameras, Channels, Height, Width

        # Combine batch, time, and cameras into one dimension for backbone processing
        image = batch['image'].view(b * t * n, c, h, w)  # (b*t*n, c, h, w)
        I_inv = batch['intrinsics'].view(b * t * n, *batch['intrinsics'].shape[3:]).inverse()  # (b*t*n, 3, 3)
        E_inv = batch['extrinsics'].view(b * t * n, *batch['extrinsics'].shape[3:]).inverse()  # (b*t*n, 4, 4)

        I_inv = rearrange(I_inv, '(bt n) ... -> bt n ...', bt=b*t, n=n)
        # print(I_inv.shape) => torch.Size([4, 5, 3, 3])
        E_inv = rearrange(E_inv, '(bt n) ... -> bt n ...', bt=b*t, n=n)
        # print(E_inv.shape) => torch.Size([4, 5, 4, 4])

        # Normalize and process image features using backbone
        features = [self.down(y) for y in self.backbone(self.norm(image))]  # Backbone features
        # Initialize BEV embedding
        x = self.bev_embedding.get_prior()  # (d, H, W)
        x = repeat(x, '... -> b ...', b=b * t)  # (b*t, d, H, W)

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            # Rearrange feature to split batch, time, and cameras
            feature = rearrange(feature, '(b t n) ... -> b t n ...', b=b, t=t, n=n)  # (b, t, n, ..., h, w)
            # print(feature.shape) => torch.Size([1, 4, 5, 32, 68, 120])
            
            # Combine batch and time for cross-view attention
            feature = feature.view(b * t, n, *feature.shape[3:])  # (b*t, n, ..., h, w)
            # print(feature.shape) => torch.Size([4, 5, 32, 68, 120])
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)

            # Apply convolutional layers
            x = layer(x)

        # Separate batch and time dimensions
        x = x.view(b, t, *x.shape[1:])  # (b, t, d, H, W)
        # print(x.shape) => torch.Size([1, 4, 128, 32, 32])
        return x