# Adapted from https://github.com/openai/guided-diffusion (MIT License, see THIRD_PARTY_NOTICES.md)
import functools
from typing import Optional

import blobfile as bf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as data
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import DictConfig
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from omniegocap.core.alignment import make_x, x_to_world
from omniegocap.diffusion import dist_util, logger
from omniegocap.diffusion.nn import update_ema
from omniegocap.diffusion.resample import LossAwareSampler, ScheduleSampler, UniformSampler
from omniegocap.diffusion.respace import SpacedDiffusion
from omniegocap.io.checkpoint import load_checkpoint, save_checkpoint
from omniegocap.training.augment import augment
from omniegocap.training.dataset import collate_omniegocap


class TrainLoop:
    def __init__(
        self,
        model: torch.nn.Module,
        config: DictConfig,
        body_model: BodyModel,
        representation_cfg: DictConfig,
        augment_cfg: DictConfig,
        diffusion: SpacedDiffusion,
        train_dataset: data.Dataset,
        batch_size: int,
        lr: float,
        ema_rate: float,
        log_interval: int,
        save_interval: int,
        max_steps: int,
        save_dir: str,
        resume_checkpoint: Optional[str] = None,
        schedule_sampler: Optional[ScheduleSampler] = None,
        weight_decay: float = 0.0,
        lambda_joint: float = 0.0,
        lambda_fk: float = 0.0,
        lambda_foot: float = 0.0,
        max_length: int = 512,
    ):
        self.model = model
        self.config = config
        self.diffusion = diffusion
        self.representation_cfg = representation_cfg
        self.augment_cfg = augment_cfg
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(',')]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.save_dir = save_dir
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.body_model = body_model
        self.lambda_joint = lambda_joint
        self.lambda_fk = lambda_fk
        self.lambda_foot = lambda_foot
        self.max_length = max_length

        self.train_sampler = data.DistributedSampler(train_dataset)

        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            pin_memory=True,
            num_workers=3,
            persistent_workers=True,
            collate_fn=collate_omniegocap,
        )

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = torch.cuda.is_available()
        assert self.sync_cuda, 'CPU training is not supported'

        self._load_and_sync_parameters()
        print('Model parameters loaded and synchronized.')

        self.opt = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True,
        )

        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [[p.detach().clone().float() for p in self.model.parameters()] for _ in self.ema_rate]

        self.ddp_model = DDP(
            self.model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
            static_graph=True,
        )

        v_template = self.body_model.init_v_template[0]
        J_regressor = self.body_model.J_regressor
        shapedirs = self.body_model.shapedirs
        self.J_template = torch.einsum('mk,jm->jk', v_template, J_regressor)
        self.J_betas = torch.einsum('mkl,jm->jkl', shapedirs, J_regressor)

        print('TrainLoop initialized.')


    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                state_dict, _, _ = load_checkpoint(resume_checkpoint, map_location=dist_util.dev())
                self.model.load_state_dict(state_dict)

        dist_util.sync_params(self.model.parameters())


    def _load_ema_parameters(self, rate):
        ema_params = [p.detach().clone().float() for p in self.model.parameters()]

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f'loading EMA from checkpoint: {ema_checkpoint}...')
                state_dict, _, _ = load_checkpoint(ema_checkpoint, map_location=dist_util.dev())
                ema_params = [state_dict[name].detach().clone().float() for name, _ in self.model.named_parameters()]

        dist_util.sync_params(ema_params)
        return ema_params


    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f'opt{self.resume_step:06}.pt'
        )
        if bf.exists(opt_checkpoint):
            logger.log(f'loading optimizer state from checkpoint: {opt_checkpoint}')
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev(), weights_only=True
            )
            self.opt.load_state_dict(state_dict)


    def run_loop(self):
        epoch = 0
        while True:
            self.train_sampler.set_epoch(epoch)

            self.ddp_model.train()
            for batch in self.train_loader:
                self.run_step(batch)
                if (self.step + self.resume_step) % self.log_interval == 0:
                    logger.dumpkvs()
                if (self.step + self.resume_step) % self.save_interval == 0:
                    self.save()
                if self.max_steps > 0 and (self.step + self.resume_step) >= self.max_steps:
                    return
                self.step += 1
            epoch += 1


    def run_step(self, batch):
        self.forward_backward(batch)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self._update_ema()
        self.log_step()


    def forward_backward(self, batch):
        head, head_delta, canonical = batch['head'], batch['head_delta'], batch['canonical']
        yaw_delta, wrist, length = batch['yaw_delta'], batch['wrist'], batch['length']
        root, joint_orient, betas = batch['root'], batch['joint_orient'], batch['betas']
        joint_pos, contact = batch['joint_pos'], batch['contact']
        batch_size = head.shape[0]

        head = head.to(dist_util.dev())
        head_delta = head_delta.to(dist_util.dev())
        canonical = canonical.to(dist_util.dev())
        yaw_delta = yaw_delta.to(dist_util.dev())
        wrist = wrist.to(dist_util.dev())
        length = length.to(dist_util.dev())
        betas = betas.to(dist_util.dev())
        root = root.to(dist_util.dev())
        joint_orient = joint_orient.to(dist_util.dev())

        xstart = make_x(root, joint_orient, head, canonical, self.representation_cfg)

        wrist, visible = augment(wrist, length, self.augment_cfg)

        t, weights = self.schedule_sampler.sample(batch_size, dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            xstart,
            t,
            model_kwargs={
                'head': head,
                'head_delta': head_delta,
                'canonical': canonical,
                'yaw_delta': yaw_delta,
                'wrist': wrist,
                'visible': visible,
                'length': length,
            },
        )

        losses, pred_betas, pred_xstart = compute_losses()

        # loss about betas
        joint_output = self.J_template[None, :, :] + torch.einsum('bl,jkl->bjk', pred_betas, self.J_betas)
        joint_target = self.J_template[None, :, :] + torch.einsum('bl,jkl->bjk', betas, self.J_betas)

        losses['joint'] = F.mse_loss(joint_output, joint_target, reduction='none').mean(dim=(-2, -1))
        losses['loss'] += self.lambda_joint * losses['joint']

        if self.lambda_fk > 0.0 or self.lambda_foot > 0.0:
            # loss about forward kinematics
            pred_T, _ = x_to_world(pred_xstart, pred_betas, canonical, head, self.body_model, self.representation_cfg)
            joint_pos = joint_pos.to(dist_util.dev())
            fk_loss = F.mse_loss(pred_T[:, :, :, :3, 3], joint_pos, reduction='none').mean(dim=(-2, -1))
            fk_loss *= (torch.arange(fk_loss.shape[1], device=dist_util.dev())[None, :] < length[:, None]).float()
            losses['fk'] = fk_loss.sum(dim=-1) / length.float()

            # loss about foot skating
            contact = contact.to(dist_util.dev())
            foot_vel = pred_T[:, 1:, 10:12, :3, 3] - pred_T[:, :-1, 10:12, :3, 3]
            mask = torch.arange(foot_vel.shape[1], device=dist_util.dev())[None, :] < (length[:, None] - 1)
            mask = (contact[:, 1:, :] & mask[:, :, None]).unsqueeze(-1)
            foot_vel *= mask.float()
            losses['foot'] = (foot_vel ** 2).sum(dim=(1, 2, 3)) / length.float()

            signal_level = torch.from_numpy(self.diffusion.alphas_cumprod).to(dist_util.dev())[t].to(torch.float32)
            losses['loss'] += (self.lambda_fk * losses['fk'] + self.lambda_foot * losses['foot']) * signal_level


        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses['loss'].detach().float()
            )

        loss = (losses['loss'] * weights).mean().float()
        log_loss_dict(
            self.diffusion, t, {k: (v * weights).float() for k, v in losses.items()}
        )
        loss.backward()

    @torch.no_grad()
    def _update_ema(self):
        src_cpu = [p.detach().float() for p in self.model.parameters()]
        for rate, ema_list in zip(self.ema_rate, self.ema_params):
            update_ema(ema_list, src_cpu, rate)


    def log_step(self):
        logger.logkv('step', self.step + self.resume_step)
        logger.logkv('samples', (self.step + self.resume_step + 1) * self.global_batch)


    def save(self):
        def save_weights(rate, ema_list):
            if dist.get_rank() > 0:
                return
            logger.log(f'saving model {rate}...')
            step = self.step + self.resume_step
            state_dict = self.model.state_dict()
            if not rate:
                filename = f'model{step:06d}.pt'
            else:
                for (name, _), ema in zip(self.model.named_parameters(), ema_list):
                    state_dict[name] = ema.to(state_dict[name].dtype)
                filename = f'ema_{rate}_{step:06d}.pt'
            # weights together with the configuration, so inference needs no config files
            save_checkpoint(
                bf.join(self.save_dir, filename),
                state_dict,
                self.config,
                meta={'step': step, 'ema_rate': rate if rate else None},
            )

        save_weights(0, None)
        for rate, ema_list in zip(self.ema_rate, self.ema_params):
            save_weights(rate, ema_list)

        if dist.get_rank() == 0:
            with bf.BlobFile(bf.join(self.save_dir, f'opt{(self.step + self.resume_step):06d}.pt'), 'wb') as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split('model')
    if len(split) < 2:
        return 0
    split1 = split[-1].split('.')[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f'ema_{rate}_{(step):06d}.pt'
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f'{key}_q{quartile}', sub_loss)
