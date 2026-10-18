import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from torch.nn.functional import scaled_dot_product_attention

from omniegocap.diffusion.nn import timestep_embedding


# the adaLN-Zero conditioning (modulate, adaLN_modulation with zero init) follows DiT (Peebles & Xie, 2023)
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq: int = 256):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq = freq


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.freq))     # (B, hidden_size)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super(TransformerEncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        assert self.hidden_size % self.num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.rotary_emb = RotaryEmbedding(self.hidden_size // self.num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mlp_dropout = nn.Dropout(dropout_p)


    def self_attn(self, x: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv_proj(x),
            'b t (qkv h d) -> qkv b h t d',
            qkv=3,
            h=self.num_heads,
        )
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = flex_attention(q, k, v, block_mask=attn_mask)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.out_proj(x)
        x = self.attn_dropout(x)
        return x


    def forward(self, x: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        x = self.self_attn(self.norm1(x), attn_mask) + x
        x = self.mlp_dropout(self.mlp(self.norm2(x))) + x
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, window_size: int, dropout_p: float = 0.0):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, dropout_p)
            for _ in range(num_layers)
        ])

        self.post_norm = nn.LayerNorm(hidden_size)


    def create_attn_mask(self, max_length: int, seq_length: torch.Tensor) -> BlockMask:
        def mask_mod(b, h, q_idx, kv_idx):
            return ((q_idx - kv_idx) <= self.window_size) & \
                ((q_idx - kv_idx) >= -self.window_size) & \
                (q_idx < seq_length[b]) & \
                (kv_idx < seq_length[b])

        block_mask = create_block_mask(
            mask_mod,
            seq_length.size(0),
            self.num_heads,
            max_length,
            max_length,
            device=seq_length.device,
        )
        return block_mask


    def forward(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        attn_mask = self.create_attn_mask(x.size(1), length)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.post_norm(x)
        return x


class DiTSelfBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super(DiTSelfBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        assert self.hidden_size % self.num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.rotary_emb = RotaryEmbedding(self.hidden_size // self.num_heads, theta=10000)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.self_out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_dropout = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mlp_dropout = nn.Dropout(dropout_p)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)


    def self_attn(self, x: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        q, k, v = rearrange(
            self.self_qkv_proj(x),
            'b t (qkv h d) -> qkv b h t d',
            qkv=3,
            h=self.num_heads,
        )
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = flex_attention(q, k, v, block_mask=attn_mask)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.self_out_proj(x)
        x = self.self_attn_dropout(x)
        return x


    def forward(self, x: torch.Tensor, global_cond: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(global_cond).chunk(6, dim=-1)
        x = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask) * gate_msa.unsqueeze(1) + x
        x = self.mlp_dropout(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))) * gate_mlp.unsqueeze(1) + x
        return x


class DiTSelf(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, window_size: int, dropout_p: float = 0.0):
        super(DiTSelf, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size

        self.layers = nn.ModuleList([
            DiTSelfBlock(hidden_size, num_heads, dropout_p)
            for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(hidden_size)


    def create_attn_mask(self, max_length: int, seq_length: torch.Tensor) -> BlockMask:
        def mask_mod(b, h, q_idx, kv_idx):
            return ((q_idx - kv_idx) <= self.window_size) & \
                ((q_idx - kv_idx) >= -self.window_size) & \
                (q_idx < seq_length[b]) & \
                (kv_idx < seq_length[b])

        block_mask = create_block_mask(
            mask_mod,
            seq_length.size(0),
            self.num_heads,
            max_length,
            max_length,
            device=seq_length.device,
        )
        return block_mask


    def forward(self, x: torch.Tensor, global_cond: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        attn_mask = self.create_attn_mask(x.size(1), length)
        for layer in self.layers:
            x = layer(x, global_cond, attn_mask)
        x = self.post_norm(x)
        return x


class DiTCrossBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super(DiTCrossBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        assert self.hidden_size % self.num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.rotary_emb = RotaryEmbedding(self.hidden_size // self.num_heads, theta=10000)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.self_qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.self_out_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn_dropout = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.cross_q_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_dropout = nn.Dropout(dropout_p)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mlp_dropout = nn.Dropout(dropout_p)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 9),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)


    def self_attn(self, x: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        q, k, v = rearrange(
            self.self_qkv_proj(x),
            'b t (qkv h d) -> qkv b h t d',
            qkv=3,
            h=self.num_heads,
        )
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = flex_attention(q, k, v, block_mask=attn_mask)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.self_out_proj(x)
        x = self.self_attn_dropout(x)
        return x


    def cross_attn(self, x: torch.Tensor, y: torch.Tensor, attn_mask: BlockMask) -> torch.Tensor:
        q = rearrange(
            self.cross_q_proj(x),
            'b t (h d) -> b h t d',
            h=self.num_heads,
        )
        k, v = rearrange(
            self.cross_kv_proj(y),
            'b t (kv h d) -> kv b h t d',
            kv=2,
            h=self.num_heads,
        )
        q = self.rotary_emb.rotate_queries_or_keys(q, seq_dim=-2)
        k = self.rotary_emb.rotate_queries_or_keys(k, seq_dim=-2)
        x = flex_attention(q, k, v, block_mask=attn_mask)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.cross_out_proj(x)
        x = self.cross_attn_dropout(x)
        return x


    def forward(
        self,
        x: torch.Tensor,
        temporal_cond: torch.Tensor,
        global_cond: torch.Tensor,
        attn_mask: BlockMask,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(global_cond).chunk(9, dim=-1)
        x = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask) * gate_msa.unsqueeze(1) + x
        x = self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), temporal_cond, attn_mask) \
            * gate_mca.unsqueeze(1) + x
        x = self.mlp_dropout(self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))) * gate_mlp.unsqueeze(1) + x
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, x_dim: int):
        super(FinalLayer, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_size, x_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)


    def forward(self, x: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(global_cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.out_proj(x)
        return x


class DiTCross(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        x_dim: int,
        dropout_p: float = 0.0,
    ):
        super(DiTCross, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.x_dim = x_dim

        self.layers = nn.ModuleList([
            DiTCrossBlock(hidden_size, num_heads, dropout_p)
            for _ in range(num_layers)
        ])

        self.in_proj = nn.Linear(x_dim, hidden_size)
        self.out_proj = FinalLayer(hidden_size, x_dim)


    def create_attn_mask(self, max_length: int, seq_length: torch.Tensor) -> BlockMask:
        def mask_mod(b, h, q_idx, kv_idx):
            return ((q_idx - kv_idx) <= self.window_size) & \
                ((q_idx - kv_idx) >= -self.window_size) & \
                (q_idx < seq_length[b]) & \
                (kv_idx < seq_length[b])

        block_mask = create_block_mask(
            mask_mod,
            seq_length.size(0),
            self.num_heads,
            max_length,
            max_length,
            device=seq_length.device,
        )
        return block_mask


    def forward(
        self,
        x: torch.Tensor,
        temporal_cond: torch.Tensor,
        global_cond: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(x)

        attn_mask = self.create_attn_mask(x.size(1), length)
        for layer in self.layers:
            x = layer(x, temporal_cond, global_cond, attn_mask)

        x = self.out_proj(x, global_cond)
        return x


class AttentionCrossBlock(nn.Module):
    def __init__(self, num_queries: int, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super(AttentionCrossBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)


    def forward(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        q = rearrange(
            self.queries.expand(x.size(0), -1, -1),
            'b t (h d) -> b h t d',
            h=self.num_heads,
        )
        k, v = rearrange(
            self.kv_proj(x),
            'b t (kv h d) -> kv b h t d',
            kv=2,
            h=self.num_heads,
        )

        mask = torch.arange(x.size(1), device=x.device) < length.unsqueeze(1)   # (B, L)
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])

        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.out_proj(x)                # (B, num_queries, hidden_size)
        x = self.attn_dropout(x)            # (B, num_queries, hidden_size)

        return x                            # (B, num_queries, hidden_size)


class AttentionSelfBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_p: float = 0.0):
        super(AttentionSelfBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.mlp_dropout = nn.Dropout(dropout_p)


    def self_attn(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(
            self.qkv_proj(x),
            'b t (qkv h d) -> qkv b h t d',
            qkv=3,
            h=self.num_heads,
        )
        x = scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.out_proj(x)
        x = self.attn_dropout(x)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(self.norm1(x)) + x
        x = self.mlp_dropout(self.mlp(self.norm2(x))) + x
        return x


class AttentionPool(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_layers: int,
        num_queries: int,
        hidden_size: int,
        num_heads: int,
        dropout_p: float = 0.0,
    ):
        super(AttentionPool, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.cross = AttentionCrossBlock(num_queries, hidden_size, num_heads, dropout_p)
        self.self = nn.ModuleList([
            AttentionSelfBlock(hidden_size, num_heads, dropout_p) for _ in range(num_layers)
        ])

        self.post_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 2, out_dim),
        )


    def forward(self, x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        x = self.cross(x, length)
        for layer in self.self:
            x = layer(x)
        x = self.post_norm(x)
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x
