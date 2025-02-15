import torch
import torch.nn as nn

class CustomTransformerLayer(nn.Module):
    """
    단일 Transformer 레이어 구현 (연산 순서: cross_attn -> norm -> ffn -> norm).
    설정(dict)을 통해 attention 및 FFN 관련 파라미터를 조정할 수 있습니다.
    """
    def __init__(self,
                 attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.1, proj_drop=0.1),
                 ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_dropout=0.1),
                 operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
                 batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        # 현재는 ('cross_attn', 'norm', 'ffn', 'norm') 순서만 지원합니다.
        if operation_order != ('cross_attn', 'norm', 'ffn', 'norm'):
            raise NotImplementedError("현재는 ('cross_attn', 'norm', 'ffn', 'norm') 순서만 지원합니다.")

        # --- Cross-Attention ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_cfg['embed_dims'],
            num_heads=attn_cfg['num_heads'],
            dropout=attn_cfg.get('attn_drop', 0.1),
            batch_first=batch_first)
        self.attn_proj_dropout = nn.Dropout(attn_cfg.get('proj_drop', 0.1))

        # --- Normalization Layers ---
        self.norm1 = nn.LayerNorm(attn_cfg['embed_dims'])
        self.norm2 = nn.LayerNorm(attn_cfg['embed_dims'])

        # --- Feed Forward Network (FFN) ---
        self.ffn = nn.Sequential(
            nn.Linear(ffn_cfg['embed_dims'], ffn_cfg['feedforward_channels']),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_cfg.get('ffn_dropout', 0.1)),
            nn.Linear(ffn_cfg['feedforward_channels'], ffn_cfg['embed_dims']),
            nn.Dropout(ffn_cfg.get('ffn_dropout', 0.1))
        )

    def forward(self, query, key, value,
                query_pos=None, key_pos=None,
                attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [B, N_query, C] 또는 [N_query, B, C] 형태의 쿼리 텐서.
            key, value: 인코더의 출력과 같은 메모리 텐서.
            query_pos, key_pos: (선택적) 위치 임베딩.
            attn_mask, key_padding_mask: (선택적) attention 마스크.
        Returns:
            query와 동일한 크기의 출력 텐서.
        """
        # 위치 임베딩이 있으면 더하기 (broadcasting 적용)
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 1. Cross-Attention (residual 연결)
        residual = query
        attn_out, _ = self.cross_attn(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        attn_out = self.attn_proj_dropout(attn_out)
        query = residual + attn_out

        # 2. Norm
        query = self.norm1(query)

        # 3. FFN (residual 연결)
        residual = query
        ffn_out = self.ffn(query)
        query = residual + ffn_out

        # 4. Norm
        query = self.norm2(query)

        return query

class CustomTransformerDecoder(nn.Module):
    """
    Custom Transformer Decoder 모듈.
    여러 Transformer 레이어를 순차적으로 쌓으며, 설정에 따라 중간 결과를 반환할 수 있습니다.
    """
    def __init__(self,
                 num_layers=1,
                 return_intermediate=False,
                 transformerlayers=dict(
                     attn_cfg=dict(embed_dims=256, num_heads=8, attn_drop=0.1, proj_drop=0.1),
                     ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_dropout=0.1),
                     operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
                     batch_first=True)):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = CustomTransformerLayer(**transformerlayers)
            self.layers.append(layer)

    def forward(self, tgt, memory,
                query_pos=None, key_pos=None,
                attn_mask=None, key_padding_mask=None):
        """
        Args:
            tgt: decoder 입력 (query), shape: [B, N_query, C] (batch_first=True인 경우)
            memory: encoder 출력 (key, value), shape: [B, N_memory, C]
            query_pos, key_pos: (선택적) 위치 임베딩.
            attn_mask, key_padding_mask: (선택적) attention 마스크.
        Returns:
            최종 출력 텐서 또는 (return_intermediate=True인 경우) 각 레이어의 출력 스택.
        """
        output = tgt
        intermediate_outputs = []
        for layer in self.layers:
            output = layer(
                query=output,
                key=memory,
                value=memory,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask)
            if self.return_intermediate:
                intermediate_outputs.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate_outputs)  # [num_layers, B, N_query, C]
        return output
class MLN(nn.Module):
    ''' 
    MLN 모듈
    Args:
        c_dim (int): latent code c의 차원
        f_dim (int): feature dimension (기본값: 256)
        use_ln (bool): LayerNorm 사용 여부
    '''
    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out

class EgoFutDecoder(nn.Module):
    """
    Ego 미래 trajectory 예측 디코더.
    Args:
        embed_dims (int): 임베딩 차원 (기본값: 256)
        num_reg_fcs (int): 회귀를 위한 fully-connected layer 개수 (기본값: 2)
        fut_steps (int): 미래 스텝 수 (기본값: 2)
        ego_fut_mode (int): 예측 모드 수 (기본값: 3)
    """
    def __init__(self, embed_dims=256, num_reg_fcs=2, fut_steps=2):
        super().__init__()
        layers = []
        in_dim = embed_dims + 12  # 추가적인 입력 정보 (예: ego 정보)

        for i in range(num_reg_fcs):
            if i == 0:
                layers.append(nn.Linear(in_dim, embed_dims))
            else:
                layers.append(nn.Linear(embed_dims, embed_dims))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(embed_dims, fut_steps * 5))
        self.ego_fut_decoder = nn.Sequential(*layers)

    def forward(self, ego_query):
        return self.ego_fut_decoder(ego_query)
