from typing import Tuple

import torch
from omegaconf import DictConfig

from omniegocap.utils.transforms import axis_angle_to_matrix


def determine_geometry(
    wrist_pos: torch.Tensor,
    yaw_offset: torch.Tensor,
    pitch_offset: torch.Tensor,
    yaw_half: torch.Tensor,
    pitch_half: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    r"""
    Determine the visibility of wrist joints based on their positions and geometry parameters, using a generalized
    ellipse equation, as described in Sec 3.4 (Geometry-Aware Visibility Augmentation) in the paper.

    Args:
        wrist_pos (batch, seq_length, 2, 3):
            Wrist positions in head coordinates.
        yaw_offset (batch,):
            Horizontal component of center offset (`\phi_x`)
        pitch_offset (batch,):
            Vertical component of center offset (`\phi_y`)
        yaw_half (batch,):
            Horizontal half angle of the visibility boundary (`\gamma_x`)
        pitch_half (batch,):
            Vertical half angle of the visibility boundary (`\gamma_y`)
        p (batch,):
            Power of the generalized ellipse equation (`p`)

    Returns:
        mask (batch, seq_length, 2):
            Visibility mask for each wrist joint (1 if visible, 0 if not).
    """

    B = wrist_pos.shape[0]

    dirs = wrist_pos / torch.norm(wrist_pos, dim=-1, keepdim=True).clamp(min=1e-8)

    yaw_offset = yaw_offset.view(B, 1, 1)
    pitch_offset = pitch_offset.view(B, 1, 1)

    cy, sy = torch.cos(-yaw_offset), torch.sin(-yaw_offset)
    cp, sp = torch.cos(-pitch_offset), torch.sin(-pitch_offset)

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

    x_y = cy * x + sy * z
    y_y = y
    z_y = -sy * x + cy * z

    x_p = x_y
    y_p = cp * y_y - sp * z_y
    z_p = sp * y_y + cp * z_y

    yaw = torch.atan2(x_p, z_p)
    pitch = torch.atan2(-y_p, torch.sqrt(x_p ** 2 + z_p ** 2).clamp(min=1e-8))

    yaw_half = yaw_half.view(B, 1, 1)
    pitch_half = pitch_half.view(B, 1, 1)
    p = p.view(B, 1, 1)

    u, v = yaw / yaw_half, pitch / pitch_half

    val = torch.abs(u) ** p + torch.abs(v) ** p
    mask = val <= 1.0

    return mask


def create_params(
    batch: int,
    yaw_half_range: Tuple[float, float] = (0.35, 2.15),
    pitch_half_range: Tuple[float, float] = (0.35, 1.35),
    yaw_offset_range: Tuple[float, float] = (-0.15, 0.15),
    pitch_offset_min: float = 0.0,
    pitch_offset_max_at_min: float = 1.5,
    pitch_offset_max_at_max: float = 0.4,
    ratio_min: float = 0.4,
    ratio_max: float = 1.1,
    p_range: Tuple[float, float] = (2.0, 10.0),
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Sample geometry parameters for visibility augmentation, as described in Appendix B (Geometry-Aware Augmentation)
    in the paper.

    Args:
        batch (int):
            Batch size.
        yaw_half_range (tuple):
            Range of horizontal half angle of the visibility boundary (`\gamma_x`).
        pitch_half_range (tuple):
            Range of vertical half angle of the visibility boundary (`\gamma_y`).
        yaw_offset_range (tuple):
            Range of horizontal component of center offset (`\phi_x`).
        pitch_offset_min (float):
            Minimum vertical component of center offset (`\phi_y`).
        pitch_offset_max_at_min (float):
            Maximum vertical component of center offset (`\phi_y`) when the half angles are at their minimum.
        pitch_offset_max_at_max (float):
            Maximum vertical component of center offset (`\phi_y`) when the half angles are at their maximum.
        ratio_min (float):
            Minimum ratio between vertical and horizontal half angles (`\gamma_y / \gamma_x`).
        ratio_max (float):
            Maximum ratio between vertical and horizontal half angles (`\gamma_y / \gamma_x`).
        p_range (tuple):
            Range of power of the generalized ellipse equation (`p`).
        device (torch.device):
            Device to create the tensors.

    Returns:
        yaw_offset (batch,):
            Horizontal component of center offset (`\phi_x`).
        pitch_offset (batch,):
            Vertical component of center offset (`\phi_y`).
        yaw_half (batch,):
            Horizontal half angle of the visibility boundary (`\gamma_x`).
        pitch_half (batch,):
            Vertical half angle of the visibility boundary (`\gamma_y`).
        p (batch,):
            Power of the generalized ellipse equation (`p`).
    """

    p = torch.empty(batch, device=device).uniform_(p_range[0], p_range[1])
    yaw_offset = torch.empty(batch, device=device).uniform_(yaw_offset_range[0], yaw_offset_range[1])

    yaw_half = torch.empty(batch, device=device).uniform_(yaw_half_range[0], yaw_half_range[1])
    pitch_half_lower = torch.maximum(torch.full((batch,), pitch_half_range[0], device=device), yaw_half * ratio_min)
    pitch_half_upper = torch.minimum(torch.full((batch,), pitch_half_range[1], device=device), yaw_half * ratio_max)
    pitch_half = pitch_half_lower + torch.rand(batch, device=device) * (pitch_half_upper - pitch_half_lower)

    pitch_offset_max = pitch_offset_max_at_min + (pitch_offset_max_at_max - pitch_offset_max_at_min) \
        * (pitch_half - pitch_half_range[0]) / (pitch_half_range[1] - pitch_half_range[0])
    pitch_offset = pitch_offset_min + torch.rand(batch, device=device) * (pitch_offset_max - pitch_offset_min)

    return yaw_offset, pitch_offset, yaw_half, pitch_half, p


def create_drop(
    batch: int,
    seq_length: int,
    rate: float,
    mean: float,
    std: float,
    min_len: int,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Create a visibility mask for temporal signal drops, as described in Appendix B (Temporal Drops and Pose
    Perturbation) in the paper.

    Args:
        batch (int):
            Batch size.
        seq_length (int):
            Sequence length.
        rate (float):
            Average drop rate (between 0 and 1).
        mean (float):
            Mean of the log-normal distribution for drop lengths.
        std (float):
            Standard deviation of the log-normal distribution for drop lengths.
        min_len (int):
            Minimum length of each drop (in frames).
        device (torch.device):
            Device to create the tensors.

    Returns:
        drop_mask (batch, seq_length, 2):
            Visibility mask for each wrist joint (1 if visible, 0 if not) only considering temporal signal drops.
    """

    drop_rate = torch.rand((batch, 2), device=device) * rate * 2

    avg_num_drop = drop_rate * seq_length / mean
    num_drop = torch.poisson(avg_num_drop).long()
    num_drop_max = num_drop.max().item()

    if num_drop_max == 0:
        return torch.ones((batch, seq_length, 2), device=device, dtype=torch.bool)

    starts = torch.floor(torch.rand((batch, 2, num_drop_max), device=device) * seq_length).long()

    # log-normal distribution
    mean, std = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    var = std ** 2
    mu = 2 * torch.log(mean) - 0.5 * torch.log(var + mean ** 2)
    sigma = torch.sqrt(torch.log(var / (mean ** 2) + 1.0))
    mu = mu.view(1, 1, 1).expand(batch, 2, num_drop_max)
    sigma = sigma.view(1, 1, 1).expand(batch, 2, num_drop_max)
    m = torch.distributions.log_normal.LogNormal(mu, sigma)
    length = m.sample()
    length = length.long().clamp(min=min_len)

    valid = torch.arange(num_drop_max, device=device).view(1, 1, -1) < num_drop.view(batch, 2, 1)
    frame_idx = torch.arange(seq_length, device=device).view(1, 1, 1, -1)
    starts = starts.unsqueeze(-1)
    length = length.unsqueeze(-1)
    drop_intervals = (frame_idx >= starts) & (frame_idx < (starts + length))
    drop_intervals = drop_intervals & valid.unsqueeze(-1)
    drop_mask = drop_intervals.any(dim=2)
    drop_mask = drop_mask.permute(0, 2, 1)
    drop_mask = ~drop_mask.contiguous()
    return drop_mask


def determine_drop(
    batch: int,
    seq_length: int,
    config: DictConfig,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Determine the visibility of wrist joints based on temporal signal drops.

    Args:
        batch (int):
            Batch size.
        seq_length (int):
            Sequence length.
        config (DictConfig):
            Configuration for temporal signal drops.
        device (torch.device):
            Device to create the tensors.

    Returns:
        mask (batch, seq_length, 2):
            Visibility mask for each wrist joint (1 if visible, 0 if not).
    """

    # short drop / long drop
    # 1. short
    short_mask = create_drop(
        batch=batch,
        seq_length=seq_length,
        rate=config.short.rate,
        mean=config.short.mean,
        std=config.short.std,
        min_len=config.short.min_len,
        device=device,
    )
    # 2. long
    long_mask = create_drop(
        batch=batch,
        seq_length=seq_length,
        rate=config.long.rate,
        mean=config.long.mean,
        std=config.long.std,
        min_len=config.long.min_len,
        device=device,
    )
    mask = short_mask & long_mask
    return mask


def perturbation(
    wrist_T: torch.Tensor,
    rotation: float,
    translation: float,
) -> torch.Tensor:
    """
    Apply random perturbation to the wrist transformations, as described in Appendix B (Temporal Drops and Pose
    Perturbation) in the paper.

    Args:
        wrist_T (batch, seq_length, 2, 4, 4):
            Wrist transformations in head coordinates.
        rotation (float):
            Standard deviation of the random rotation (in radians).
        translation (float):
            Standard deviation of the random translation (in meters).

    Returns:
        wrist_T_aug (batch, seq_length, 2, 4, 4):
            Augmented wrist transformations.
    """

    B = wrist_T.shape[0]

    angles = torch.randn((B, 2, 3), device=wrist_T.device) * rotation
    R_aug = axis_angle_to_matrix(angles.view(-1, 3)).view(B, 1, 2, 3, 3)

    t_aug = torch.randn((B, 1, 2, 3), device=wrist_T.device) * translation

    wrist_T_aug = wrist_T.clone()
    wrist_T_aug[..., :3, :3] = torch.matmul(R_aug, wrist_T[..., :3, :3])
    wrist_T_aug[..., :3, 3] = torch.matmul(R_aug, wrist_T[..., :3, 3].unsqueeze(-1)).squeeze(-1) + t_aug

    return wrist_T_aug


def augment_full(
    wrist: torch.Tensor,
    length: torch.Tensor,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the wrist input and visibility for training, including geometry-aware augmentation, temporal signal drops
    and pose perturbation.

    Args:
        wrist (batch, seq_length, 2, 4, 4):
            Wrist transformation in head coordinates.
        length (batch,):
            Valid length of each sequence in the batch.
        config (DictConfig):
            Augmentation configuration.

    Returns:
        wrist (batch, seq_length, 2, 4, 4):
            Augmented wrist transformation.
        visible (batch, seq_length, 2):
            Visibility mask.
    """

    B, L = wrist.shape[:2]
    device = wrist.device

    yaw_offset, pitch_offset, yaw_half, pitch_half, p = create_params(B, **config.geometry, device=device)
    geometry_vis = determine_geometry(wrist[..., :3, 3], yaw_offset, pitch_offset, yaw_half, pitch_half, p)

    drop_vis = determine_drop(B, L, config.drop, device=device)

    length_mask = torch.arange(L, device=device).view(1, -1) < length.view(-1, 1)
    visible = geometry_vis & drop_vis & length_mask.unsqueeze(-1)

    wrist = perturbation(wrist, config.perturbation.rotation, config.perturbation.translation)

    return wrist, visible


def augment_geometry(
    wrist: torch.Tensor,
    length: torch.Tensor,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the wrist visibility for training, only considering geometry-aware augmentation.

    Args:
        wrist (batch, seq_length, 2, 4, 4):
            Wrist transformation in head coordinates.
        length (batch,):
            Valid length of each sequence in the batch.
        config (DictConfig):
            Augmentation configuration.

    Returns:
        wrist (batch, seq_length, 2, 4, 4):
            Augmented wrist transformation.
        visible (batch, seq_length, 2):
            Visibility mask.
    """

    B, L = wrist.shape[:2]
    device = wrist.device

    yaw_offset, pitch_offset, yaw_half, pitch_half, p = create_params(B, **config.geometry, device=device)
    geometry_vis = determine_geometry(wrist[..., :3, 3], yaw_offset, pitch_offset, yaw_half, pitch_half, p)

    length_mask = torch.arange(L, device=device).view(1, -1) < length.view(-1, 1)
    visible = geometry_vis & length_mask.unsqueeze(-1)

    return wrist, visible


def augment_fixed(
    wrist: torch.Tensor,
    length: torch.Tensor,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the wrist visibility for training with a fixed field of view for ablation.

    Args:
        wrist (batch, seq_length, 2, 4, 4):
            Wrist transformation in head coordinates.
        length (batch,):
            Valid length of each sequence in the batch.
        config (DictConfig):
            Augmentation configuration, including fixed FoV.

    Returns:
        wrist (batch, seq_length, 2, 4, 4):
            Augmented wrist transformation.
        visible (batch, seq_length, 2):
            Visibility mask.
    """

    _, L = wrist.shape[:2]
    device = wrist.device
    pos = wrist[..., :3, 3]    # (B, L, 2, 3)
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]

    half = 0.5 * config.fixed.fov * torch.pi / 180
    yaw = torch.atan2(x, z)
    pitch = torch.atan2(y, z)

    vis = (z > 0.0) & (yaw.abs() <= half) & (pitch.abs() <= half)

    length_mask = torch.arange(L, device=device).view(1, -1) < length.view(-1, 1)
    vis = vis & length_mask.unsqueeze(-1)

    return wrist, vis


def augment_random(
    wrist: torch.Tensor,
    length: torch.Tensor,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the wrist visibility for training by randomly masking out (bernoulli) for ablation.

    Args:
        wrist (batch, seq_length, 2, 4, 4):
            Wrist transformation in head coordinates.
        length (batch,):
            Valid length of each sequence in the batch.
        config (DictConfig):
            Augmentation configuration, including drop rate.

    Returns:
        wrist (batch, seq_length, 2, 4, 4):
            Augmented wrist transformation.
        visible (batch, seq_length, 2):
            Visibility mask.
    """

    B, L = wrist.shape[:2]
    device = wrist.device

    # i.i.d. Bernoulli dropout, per-frame and per-wrist (geometry-free)
    keep_prob = 1.0 - config.random.p
    visible = (torch.rand((B, L, 2), device=device) < keep_prob)

    length_mask = torch.arange(L, device=device).view(1, -1) < length.view(-1, 1)
    visible = visible & length_mask.unsqueeze(-1)

    return wrist, visible


def augment(
    wrist: torch.Tensor,
    length: torch.Tensor,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment the wrist input and visibility for training.

    Args:
        wrist (batch, seq_length, 2, 4, 4):
            Wrist transformation in head coordinates.
        length (batch,):
            Valid length of each sequence in the batch.
        config (DictConfig):
            Augmentation configuration.

    Returns:
        wrist (batch, seq_length, 2, 4, 4):
            Augmented wrist transformation.
        visible (batch, seq_length, 2):
            Visibility mask.
    """

    if config.mode == 'full':
        return augment_full(wrist, length, config)
    elif config.mode == 'geometry':
        return augment_geometry(wrist, length, config)
    elif config.mode == 'fixed':
        return augment_fixed(wrist, length, config)
    elif config.mode == 'random':
        return augment_random(wrist, length, config)
    raise NotImplementedError
