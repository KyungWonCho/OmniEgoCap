from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from omniegocap.core.modules import AttentionPool, DiTCross, DiTSelf, TimestepEmbedder, TransformerEncoder
from omniegocap.utils.transforms import inv_R_tensor


class ConditionEmbedder(nn.Module):
    def __init__(self, output_dim: int):
        super(ConditionEmbedder, self).__init__()
        self.head_embedder = nn.Sequential(
            nn.Linear(17, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.left_embedder = nn.Sequential(
            nn.Linear(9, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.right_embedder = nn.Sequential(
            nn.Linear(9, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.left_null_embed = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.right_null_embed = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

        self.proj = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
        )


    def forward(
        self,
        head: torch.Tensor,
        head_delta: torch.Tensor,
        canonical: torch.Tensor,
        yaw_delta: torch.Tensor,
        wrist: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed the conditioning information into a feature vector.

        Args:
            head (batch, seq_length, 4, 4):
                Head transformation in world coordinate
            head_delta (batch, seq_length, 4, 4):
                Relative transformation of head compared to the previous frame
            canonical (batch, seq_length, 4, 4):
                Canonical transformation in world coordinate
            yaw_delta (batch, seq_length):
                Relative yaw angle of canonical compared to the previous frame
            wrist (batch, seq_length, 2, 4, 4):
                Wrists transformation in head coordinate
            visible (batch, seq_length, 2):
                Visibility mask for each wrist

        Returns:
            cond (batch, seq_length, output_dim):
                Embedded conditioning feature
        """

        B, L = head.shape[:2]

        canonical_head = inv_R_tensor(canonical[:, :, :3, :3]) @ head[:, :, :3, :3]

        head_cond = torch.cat([
            head_delta[:, :, :2, :3].reshape(B, L, -1),
            head_delta[:, :, :3, 3],
            canonical_head[:, :, :2, :3].reshape(B, L, -1),
            yaw_delta.unsqueeze(-1),
            head[:, :, 2, 3].unsqueeze(-1),
        ], dim=-1)      # (B, L, 6 + 3 + 6 + 1 + 1 = 17)
        head_emb = self.head_embedder(
            head_cond
        )

        left_cond = torch.cat([
            wrist[:, :, 0, :2, :3].reshape(B, L, -1),
            wrist[:, :, 0, :3, 3],
        ], dim=-1)      # (B, L, 6 + 3 = 9)
        left_emb = torch.where(
            visible[:, :, 0:1],
            self.left_embedder(left_cond),
            self.left_null_embed,
        )
        right_cond = torch.cat([
            wrist[:, :, 1, :2, :3].reshape(B, L, -1),
            wrist[:, :, 1, :3, 3],
        ], dim=-1)      # (B, L, 6 + 3 = 9)
        right_emb = torch.where(
            visible[:, :, 1:2],
            self.right_embedder(right_cond),
            self.right_null_embed,
        )

        cond = torch.cat([head_emb, left_emb, right_emb], dim=-1)
        cond = self.proj(cond)

        return cond


class OmniEgoCapEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(OmniEgoCapEncoder, self).__init__()

        self.use_shape = cfg.use_shape
        self.use_global_AdaLN = cfg.use_global_AdaLN
        self.use_global_concat = cfg.use_global_concat
        assert not (self.use_global_AdaLN and self.use_global_concat), \
            'Cannot use both AdaLN and concat for global condition.'

        if self.use_shape:
            self.shape_embedder = ConditionEmbedder(
                output_dim=cfg.shape_encoder.hidden_size,
            )
            self.shape_encoder = TransformerEncoder(
                num_layers=cfg.shape_encoder.num_layers,
                hidden_size=cfg.shape_encoder.hidden_size,
                num_heads=cfg.shape_encoder.num_heads,
                window_size=cfg.shape_encoder.window_size,
                dropout_p=cfg.shape_encoder.dropout_p,
            )
            self.shape_pool = AttentionPool(
                out_dim=16,                                     # SMPL-H betas (16 dim)
                num_queries=cfg.shape_pool.num_queries,
                num_layers=cfg.shape_pool.num_layers,
                hidden_size=cfg.shape_pool.hidden_size,
                num_heads=cfg.shape_pool.num_heads,
                dropout_p=cfg.shape_pool.dropout_p,
            )

        if self.use_global_AdaLN or self.use_global_concat:
            self.global_embedder = ConditionEmbedder(
                output_dim=cfg.global_encoder.hidden_size,
            )
            self.global_encoder = TransformerEncoder(
                num_layers=cfg.global_encoder.num_layers,
                hidden_size=cfg.global_encoder.hidden_size,
                num_heads=cfg.global_encoder.num_heads,
                window_size=cfg.global_encoder.window_size,
                dropout_p=cfg.global_encoder.dropout_p,
            )
            self.global_pool = AttentionPool(
                out_dim=cfg.local_encoder.hidden_size,
                num_queries=cfg.global_pool.num_queries,
                num_layers=cfg.global_pool.num_layers,
                hidden_size=cfg.global_pool.hidden_size,
                num_heads=cfg.global_pool.num_heads,
                dropout_p=cfg.global_pool.dropout_p,
            )

        self.local_embedder = ConditionEmbedder(
            output_dim=cfg.local_encoder.hidden_size,
        )
        if self.use_global_AdaLN:
            self.local_encoder = DiTSelf(
                num_layers=cfg.local_encoder.num_layers,
                hidden_size=cfg.local_encoder.hidden_size,
                num_heads=cfg.local_encoder.num_heads,
                window_size=cfg.local_encoder.window_size,
                dropout_p=cfg.local_encoder.dropout_p,
            )
        elif self.use_global_concat:
            self.local_proj = nn.Linear(
                cfg.local_encoder.hidden_size + cfg.local_encoder.hidden_size,
                cfg.local_encoder.hidden_size,
            )
            self.local_encoder = TransformerEncoder(
                num_layers=cfg.local_encoder.num_layers,
                hidden_size=cfg.local_encoder.hidden_size,
                num_heads=cfg.local_encoder.num_heads,
                window_size=cfg.local_encoder.window_size,
                dropout_p=cfg.local_encoder.dropout_p,
            )
        else:
            self.local_encoder = TransformerEncoder(
                num_layers=cfg.local_encoder.num_layers,
                hidden_size=cfg.local_encoder.hidden_size,
                num_heads=cfg.local_encoder.num_heads,
                window_size=cfg.local_encoder.window_size,
                dropout_p=cfg.local_encoder.dropout_p,
            )


    def forward(
        self,
        head: torch.Tensor,
        head_delta: torch.Tensor,
        canonical: torch.Tensor,
        yaw_delta: torch.Tensor,
        wrist: torch.Tensor,
        visible: torch.Tensor,
        length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the conditioning information into a summary embedding and shape parameters.

        Args:
            head (batch, seq_length, 4, 4):
                Head transformation in world coordinate
            head_delta (batch, seq_length, 4, 4):
                Relative transformation of head
            canonical (batch, seq_length, 4, 4):
                Canonical transformation in world coordinate
            yaw_delta (batch, seq_length):
                Relative yaw angle of canonical
            wrist (batch, seq_length, 2, 4, 4):
                Wrists transformation in head coordinate
            visible (batch, seq_length, 2):
                Visibility mask for each wrist
            length (batch,):
                Valid length of each sequence in the batch.

        Returns:
            summary_emb (batch, seq_length, hidden_size):
                Summary embedding for each frame
            betas (batch, 16):
                Predicted shape parameters (SMPL-H betas)
        """

        B, L = head.shape[:2]

        # shape
        if self.use_shape:
            shape_cond = self.shape_embedder(head, head_delta, canonical, yaw_delta, wrist, visible)
            shape_emb = self.shape_encoder(shape_cond, length)
            betas = self.shape_pool(shape_emb, length)
        else:
            betas = torch.zeros(B, 16, device=head.device, requires_grad=False)    # mean shape

        # global
        if self.use_global_AdaLN or self.use_global_concat:
            global_cond = self.global_embedder(head, head_delta, canonical, yaw_delta, wrist, visible)
            global_emb = self.global_encoder(global_cond, length)
            global_emb = self.global_pool(global_emb, length)

        # local
        local_cond = self.local_embedder(head, head_delta, canonical, yaw_delta, wrist, visible)
        if self.use_global_AdaLN:
            local_emb = self.local_encoder(local_cond, global_emb, length)
        elif self.use_global_concat:
            global_emb = global_emb.unsqueeze(1).expand(-1, L, -1)          # (B, L, H_g)
            local_cond = torch.cat([local_cond, global_emb], dim=-1)        # (B, L, H_l + H_g)
            local_cond_proj = self.local_proj(local_cond)                   # (B, L, H_l)
            local_emb = self.local_encoder(local_cond_proj, length)
        else:
            local_emb = self.local_encoder(local_cond, length)

        return local_emb, betas


class OmniEgoCapDecoder(nn.Module):
    def __init__(self, cfg: DictConfig, x_dim: int):
        super(OmniEgoCapDecoder, self).__init__()
        self.use_shape_condition = cfg.use_shape_condition

        self.timestep_embedder = TimestepEmbedder(
            hidden_size=cfg.decoder.hidden_size,
        )
        self.decoder = DiTCross(
            num_layers=cfg.decoder.num_layers,
            hidden_size=cfg.decoder.hidden_size,
            num_heads=cfg.decoder.num_heads,
            window_size=cfg.decoder.window_size,
            x_dim=x_dim,
            dropout_p=cfg.decoder.dropout_p,
        )

        if self.use_shape_condition:
            self.betas_proj = nn.Sequential(
                nn.Linear(16, cfg.decoder.hidden_size),
                nn.SiLU(),
                nn.Linear(cfg.decoder.hidden_size, cfg.decoder.hidden_size),
            )


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        summary_emb: torch.Tensor,
        betas: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise the noisy canonicalized pose `x`.

        Args:
            x (batch, x_dim, seq_length):
                Noisy canonicalized pose
            t (batch,):
                Diffusion timestep
            summary_emb (batch, seq_length, hidden_size):
                Summary embedding from the encoder
            betas (batch, 16):
                Shape parameters from the encoder
            length (batch,):
                Valid length of each sequence

        Returns:
            x (batch, x_dim, seq_length):
                Denoised canonicalized pose
        """

        x = x.permute(0, 2, 1)

        t_emb = self.timestep_embedder(t)
        if self.use_shape_condition:
            betas_emb = self.betas_proj(betas)
            global_emb = betas_emb + t_emb
        else:
            global_emb = t_emb

        x = self.decoder(x, summary_emb, global_emb, length)

        x = x.permute(0, 2, 1)

        return x


class OmniEgoCapModel(nn.Module):
    def __init__(self, cfg: DictConfig, x_dim: int):
        super(OmniEgoCapModel, self).__init__()
        assert cfg.local_encoder.hidden_size == cfg.decoder.hidden_size
        self.encoder = OmniEgoCapEncoder(cfg)
        self.decoder = OmniEgoCapDecoder(cfg, x_dim)


    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        head: torch.Tensor,
        head_delta: torch.Tensor,
        canonical: torch.Tensor,
        yaw_delta: torch.Tensor,
        wrist: torch.Tensor,
        visible: torch.Tensor,
        length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the noisy canonicalized pose `x` using the conditioning information.

        Args:
            x (batch, x_dim, seq_length):
                Noisy canonicalized pose
            t (batch,):
                Diffusion timestep
            head (batch, seq_length, 4, 4):
                Head transformation in world coordinate
            head_delta (batch, seq_length, 4, 4):
                Relative transformation of head
            canonical (batch, seq_length, 4, 4):
                Canonical transformation in world coordinate
            yaw_delta (batch, seq_length):
                Relative yaw angle of canonical
            wrist (batch, seq_length, 2, 4, 4):
                Wrists transformation in head coordinate
            visible (batch, seq_length, 2):
                Visibility mask for each wrist
            length (batch,):
                Valid length of each sequence

        Returns:
            x (batch, x_dim, seq_length):
                Denoised canonicalized pose
            betas (batch, 16):
                Predicted shape parameters (SMPL-H betas)
        """

        summary_emb, betas = self.encoder(head, head_delta, canonical, yaw_delta, wrist, visible, length)

        x = self.decoder(x, t, summary_emb, betas, length)

        return x, betas
