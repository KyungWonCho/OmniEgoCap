"""
Usage
    from omniegocap.core.inference import OmniEgoCap
    from omniegocap.io.sequence import EgoSequence

    model = OmniEgoCap('checkpoints/omniegocap.pt')
    result = model.predict(EgoSequence.load('seq_input.npz'))
    result.save('seq_smplh.npz')
"""

import os
import time
from typing import Optional, Tuple

import torch
from human_body_prior.body_model.body_model import BodyModel

from omniegocap.core.alignment import get_x_dim, make_x, x_to_world
from omniegocap.core.features import make_features
from omniegocap.core.optimization import optimize_inner
from omniegocap.diffusion.gaussian_diffusion import _extract_into_tensor
from omniegocap.diffusion.script_util import create_diffusion
from omniegocap.io.checkpoint import build_model
from omniegocap.io.sequence import EgoSequence
from omniegocap.io.smplh import SMPLHSequence
from omniegocap.utils.paths import smplh_path

GUIDANCE_SCALE = 30.0
SEED = 42


class OmniEgoCap:
    def __init__(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        smplh_dir: Optional[str] = None,
    ):
        """
        Pretrained OmniEgoCap model with its diffusion sampler and the SMPL-H body model.

        Args:
            checkpoint (str):
                Path to a checkpoint saved by omniegocap.io.checkpoint.
            device (str or None):
                Device to run on. If None, use CUDA when available.
            smplh_dir (str or None):
                Directory of the SMPL-H model files (see omniegocap.utils.paths).
        """

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint = checkpoint
        self.model, self.config, self.meta = build_model(checkpoint, device=self.device)
        self.x_dim = get_x_dim(self.config.representation)
        self.diffusion = create_diffusion(
            steps=self.config.diffusion.steps,
            noise_schedule=self.config.diffusion.noise_schedule,
            predict_xstart=self.config.diffusion.predict_xstart,
            timestep_respacing=self.config.diffusion.timestep_respacing,
        )
        self.body_model = BodyModel(bm_fname=smplh_path('neutral', smplh_dir), num_betas=16).to(self.device)


    @torch.no_grad()
    def sample(
        self,
        head: torch.Tensor,
        head_delta: torch.Tensor,
        canonical: torch.Tensor,
        yaw_delta: torch.Tensor,
        wrist: torch.Tensor,
        visible: torch.Tensor,
        length: torch.Tensor,
        optimization: bool = True,
        guidance_scale: float = GUIDANCE_SCALE,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DDIM sampling (eta = 0) with timestep-adaptive guidance optimization, as described in Sec 3.5 (Inference) in
        the paper.

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
                Valid length of each sequence
            optimization (bool):
                Whether to run the guidance optimization.
            guidance_scale (float):
                Guidance scale for optimization. Higher means more weight on observation.
            generator (torch.Generator or None):
                Generator for the initial noise. If None, use the global random state.

        Returns:
            local_T (batch, seq_length, 22, 4, 4):
                Root transformation in world (joint 0) and local joint transformations (joints 1 .. 21).
            betas (batch, 16):
                Predicted shape parameters (SMPL-H betas)
        """

        B, L = head.shape[:2]

        summary_emb, pred_betas = self.model.encoder(head, head_delta, canonical, yaw_delta, wrist, visible, length)

        x = torch.randn((B, self.x_dim, L), device=self.device, generator=generator)
        for i in list(range(self.diffusion.num_timesteps))[::-1]:
            t = torch.tensor([i] * B, device=self.device)

            out = self.diffusion.p_mean_variance(
                self.model.decoder,
                x,
                t,
                model_kwargs={
                    'summary_emb': summary_emb,
                    'betas': pred_betas,
                    'length': length,
                },
            )
            pred_xstart = out['pred_xstart']

            if optimization:
                noise_level = 1 - self.diffusion.alphas_cumprod[i]
                global_T, local_T = x_to_world(
                    x=pred_xstart,
                    betas=pred_betas,
                    canonical=canonical,
                    head=head,
                    body_model=self.body_model,
                    config=self.config.representation,
                )
                wrist_world = head[:, :, None] @ wrist
                local_T = optimize_inner(
                    global_T=global_T,
                    local_T=local_T,
                    observ=wrist_world,
                    visible=visible,
                    noise_level=noise_level,
                    guidance_scale=guidance_scale,
                )
                pred_xstart = make_x(
                    root=local_T[:, :, 0],
                    joint_orient=local_T[:, :, 1:, :3, :3],
                    head=head,
                    canonical=canonical,
                    config=self.config.representation,
                )

            # eta = 0
            eps = self.diffusion._predict_eps_from_xstart(x, t, pred_xstart)
            alpha_bar_prev = _extract_into_tensor(self.diffusion.alphas_cumprod_prev, t, x.shape)
            x = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps

        _, local_T = x_to_world(
            x=x,
            betas=pred_betas,
            canonical=canonical,
            head=head,
            body_model=self.body_model,
            config=self.config.representation,
        )                                                   # (B, L, 22, 4, 4)

        return local_T, pred_betas


    def predict(
        self,
        seq: EgoSequence,
        optimization: bool = True,
        guidance_scale: float = GUIDANCE_SCALE,
        seed: int = SEED,
    ) -> SMPLHSequence:
        """
        Reconstruct full-body SMPL-H motion from one input sequence.

        Args:
            seq (EgoSequence):
                Head and wrist poses in world coordinates.
            optimization (bool):
                Whether to run the guidance optimization.
            guidance_scale (float):
                Guidance scale for optimization.
            seed (int):
                Seed of the initial noise, so the same input always gives the same output.

        Returns:
            result (SMPLHSequence):
                SMPL-H motion with the same number of frames as the input (frame 0 marked invalid).
        """

        features = make_features(seq.head, seq.wrist, seq.visible)
        inputs = {key: torch.from_numpy(value)[None].to(self.device) for key, value in features.items()}
        length = torch.tensor([len(seq) - 1], dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        start_time = time.time()
        local_T, betas = self.sample(
            **inputs,
            length=length,
            optimization=optimization,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        local_T, betas = local_T[0].cpu().numpy(), betas[0].cpu().numpy()
        runtime = time.time() - start_time

        meta = {
            'checkpoint': os.path.basename(self.checkpoint),
            'checkpoint_meta': self.meta,
            'seed': seed,
            'optimization': optimization,
            'guidance_scale': guidance_scale,
            'runtime_s': runtime,
        }

        return SMPLHSequence.from_local_T(local_T, betas, self.body_model, meta=meta)
