"""
Usage
mpiexec -n 8 python scripts/train.py
"""

import os

import hydra
import torch
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import DictConfig

from omniegocap.core.alignment import get_x_dim
from omniegocap.core.model import OmniEgoCapModel
from omniegocap.diffusion import dist_util, logger
from omniegocap.diffusion.resample import create_named_schedule_sampler
from omniegocap.diffusion.script_util import create_diffusion
from omniegocap.training.dataset import OmniEgoCapDataset
from omniegocap.training.train_util import TrainLoop
from omniegocap.utils.paths import register_resolvers
from omniegocap.utils.transforms import set_seed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
register_resolvers()


@hydra.main(config_path='../conf', config_name='config', version_base='1.2')
def main(cfg: DictConfig):
    set_seed(42)

    dist_util.setup_dist()

    logger.configure(dir=os.path.join(cfg.train.save_dir, 'logs'))

    train_dataset = OmniEgoCapDataset(
        file=os.path.join(cfg.paths.preprocess, 'train.h5'),
        max_length=cfg.train.max_length,
    )

    model = OmniEgoCapModel(cfg.model, get_x_dim(cfg.representation))
    model.to(dist_util.dev())

    diffusion = create_diffusion(
        steps=cfg.diffusion.steps,
        noise_schedule=cfg.diffusion.noise_schedule,
        predict_xstart=cfg.diffusion.predict_xstart,
    )

    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    body_model = BodyModel(bm_fname=cfg.paths.smplh.neutral, num_betas=16)
    body_model.to(dist_util.dev())

    TrainLoop(
        model=model,
        config=cfg,
        body_model=body_model,
        representation_cfg=cfg.representation,
        augment_cfg=cfg.augment,
        diffusion=diffusion,
        train_dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        ema_rate=cfg.train.ema_rate,
        log_interval=cfg.train.log_interval,
        save_interval=cfg.train.save_interval,
        max_steps=cfg.train.max_steps,
        save_dir=cfg.train.save_dir,
        resume_checkpoint=cfg.train.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.train.weight_decay,
        lambda_joint=cfg.train.lambda_joint,
        lambda_fk=cfg.train.lambda_fk,
        lambda_foot=cfg.train.lambda_foot,
        max_length=cfg.train.max_length,
    ).run_loop()


if __name__ == '__main__':
    main()
