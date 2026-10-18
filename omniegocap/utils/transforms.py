import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inv_T_tensor(T: torch.Tensor) -> torch.Tensor:
    # T: (..., 4, 4)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = T[..., :3, :3].transpose(-1, -2)
    T_inv[..., :3, 3] = -(T_inv[..., :3, :3] @ T[..., :3, 3:4]).squeeze(-1)
    T_inv[..., 3, 3] = 1.0
    return T_inv


def inv_R_tensor(R: torch.Tensor) -> torch.Tensor:
    # R: (..., 3, 3)
    return R.transpose(-1, -2)


def inv_T_numpy(T: np.ndarray) -> np.ndarray:
    # T: (..., 4, 4)
    T_inv = np.zeros_like(T)
    T_inv[..., :3, :3] = T[..., :3, :3].swapaxes(-1, -2)
    T_inv[..., :3, 3] = -(T_inv[..., :3, :3] @ T[..., :3, 3:4]).squeeze(-1)
    T_inv[..., 3, 3] = 1.0
    return T_inv


def make_canonical(
    T: np.ndarray,  # z-axis forward, y-axis up
) -> Tuple[np.ndarray, np.ndarray]:
    L = T.shape[0]

    fwd = T[:, :3, 2].copy()                    # (L, 3)
    fwd[:, 2] = 0.0                             # project to xy plane

    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])

    y_axis = np.stack([np.cos(yaw), np.sin(yaw), np.zeros((L,))], axis=-1)
    z_axis = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (L, 1))
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)

    T_canonical = np.zeros_like(T)
    T_canonical[:, :3, 0] = x_axis
    T_canonical[:, :3, 1] = y_axis
    T_canonical[:, :3, 2] = z_axis
    T_canonical[:, :2, 3] = T[:, :2, 3]
    T_canonical[:, 3, 3] = 1.0

    yaw_delta = yaw[1:] - yaw[:-1]

    return T_canonical[1:], yaw_delta



def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    # R: (..., 3, 3) -> (..., 6), the first two rows
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    # d6: (..., 6) -> (..., 3, 3), rows from Gram-Schmidt on the two 3-vectors
    r0 = F.normalize(d6[..., :3], dim=-1)
    r1 = d6[..., 3:]
    r1 = F.normalize(r1 - (r0 * r1).sum(-1, keepdim=True) * r0, dim=-1)
    r2 = torch.cross(r0, r1, dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def skew(v: torch.Tensor) -> torch.Tensor:
    # v: (..., 3) -> (..., 3, 3), skew(v) @ u = v x u
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(v.shape[:-1] + (3, 3))


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    # aa: (..., 3) -> (..., 3, 3), Rodrigues: R = I + sin(t) / t K + (1 - cos(t)) / t^2 K^2 with K = skew(aa)
    t = aa.norm(dim=-1)[..., None, None] / torch.pi
    a = torch.sinc(t)                                       # sin(t) / t
    b = 0.5 * torch.sinc(t / 2) ** 2                        # (1 - cos(t)) / t^2, without cancellation
    K = skew(aa)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + a * K + b * (K @ K)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    # R: (..., 3, 3) -> (..., 3), angle in [0, pi]
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sin_t = 0.5 * w.norm(dim=-1)                                                # (...)
    cos_t = 0.5 * (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)                    # (...)
    t = torch.atan2(sin_t, cos_t)

    # up to 90 degrees: aa = t / (2 sin(t)) * w
    small = sin_t < 1e-6
    sin_safe = torch.where(small, torch.ones_like(sin_t), sin_t)
    scale = torch.where(small, 0.5 + t * t / 12, t / (2 * sin_safe))
    aa = scale[..., None] * w

    # beyond 90 degrees: axis from the symmetric part, n n^T = (S - cos(t) I) / (1 - cos(t))
    S = 0.5 * (R + R.transpose(-1, -2))
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    nnT = (S - cos_t[..., None, None] * eye) / (1 - cos_t).clamp(min=1e-6)[..., None, None]
    k = nnT.diagonal(dim1=-2, dim2=-1).argmax(-1)                               # (...)
    col = torch.gather(nnT, -1, k[..., None, None].expand(R.shape[:-1] + (1,))).squeeze(-1)
    n = col / torch.gather(col, -1, k[..., None]).clamp(min=1e-12).sqrt()
    n = torch.where((n * w).sum(-1, keepdim=True) < 0, -n, n)                   # same side as w
    return torch.where((cos_t < 0)[..., None], t[..., None] * n, aa)
