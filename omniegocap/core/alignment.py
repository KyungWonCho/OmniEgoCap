from typing import Optional, Tuple

import torch
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import DictConfig

from omniegocap.utils import constants
from omniegocap.utils.transforms import inv_R_tensor, inv_T_tensor, matrix_to_rotation_6d, rotation_6d_to_matrix


def forward_kinematics(
    joint_orient: torch.Tensor,
    betas: torch.Tensor,
    body_model: BodyModel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform forward kinematics to get global joint transformation in root (pelvis) coordinates.

    Args:
        joint_orient (batch, seq_length, 21, 3, 3):
            Joint orientations in local coordinates (parent to child).
        betas (batch, 16):
            Body shape parameters.
        body_model (BodyModel):
            SMPL-H body model.

    Returns:
        root_T (batch, seq_length, 22, 4, 4):
            Global joint transformation in root (pelvis) coordinates.
        skel_offset (batch, 21, 3):
            Joint offset from parent joint to child joint in T-pose.
    """

    device = joint_orient.device

    B, L = joint_orient.shape[:2]

    v_template = body_model.init_v_template.expand(B, -1, -1).to(device)
    J_regressor = body_model.J_regressor.to(device)
    shapedirs = body_model.shapedirs.to(device)
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', betas, shapedirs)
    J = torch.einsum('bik,ji->bjk', v_shaped, J_regressor[:22, :])
    skel_parents = body_model.kintree_table[0, :22].long()
    skel_offset = J[:, 1:] - J[:, skel_parents[1:]]
    skel_offset = skel_offset.reshape(B, 21, 1, 3, 1)

    root_T = [torch.eye(4, device=device).repeat(B, L, 1, 1)]
    for i in range(1, 22):
        joint_T = torch.zeros((B, L, 4, 4), device=device)
        joint_T[:, :, :3, :3] = root_T[skel_parents[i]][:, :, :3, :3] @ joint_orient[:, :, i - 1]
        joint_T[:, :, :3, 3] = (root_T[skel_parents[i]][:, :, :3, :3] @ skel_offset[:, i - 1]).squeeze(-1) \
            + root_T[skel_parents[i]][:, :, :3, 3]
        joint_T[:, :, 3, 3] = 1.0
        root_T.append(joint_T)
    root_T = torch.stack(root_T, dim=2)

    return root_T, skel_offset.squeeze(-1).squeeze(-2)


def make_x(
    root: torch.Tensor,
    joint_orient: torch.Tensor,
    head: torch.Tensor,
    canonical: torch.Tensor,
    config: DictConfig,
) -> torch.Tensor:
    """
    Make denoising target `x` (canonicalized pose representation).

    Args:
        root (batch, seq_length, 4, 4):
            Root transformation in world coordinates.
        joint_orient (batch, seq_length, 21, 3, 3):
            Joint orientations in local coordinates.
        head (batch, seq_length, 4, 4):
            Head transformation in world coordinates.
        canonical (batch, seq_length, 4, 4):
            Canonical transformation in world coordinates.
        config (DictConfig):
            Representation configuration.

    Returns:
        xstart (batch, x_dim, seq_length):
            Denoising target `x` (canonicalized pose representation).
    """

    if config.reference == 'head':
        root = inv_T_tensor(head) @ root
    elif config.reference == 'canonical':
        root = inv_T_tensor(canonical) @ root
    elif config.reference != 'none':
        raise NotImplementedError

    if config.root == 'none':
        xstart = matrix_to_rotation_6d(joint_orient).flatten(2)
    elif config.root == 'orientation':
        xstart = torch.cat([
            matrix_to_rotation_6d(root[:, :, :3, :3]),
            matrix_to_rotation_6d(joint_orient).flatten(2)
        ], dim=-1)
    else:
        raise NotImplementedError

    xstart = xstart.permute(0, 2, 1).contiguous()

    return xstart


def get_x_dim(config: DictConfig) -> int:
    """
    Get the dimension of the denoising target `x` (canonicalized pose representation).

    Args:
        config (DictConfig):
            Representation configuration.

    Returns:
        x_dim (int):
            The dimension of the denoising target `x`.
    """

    assert config.root in ['none', 'orientation'], f'Unsupported root representation: {config.root}'
    assert config.reference in ['none', 'head', 'canonical'], f'Unsupported reference: {config.reference}'
    assert not (config.root == 'orientation' and config.reference == 'none'), \
        'If root orientation is used, reference cannot be none.'
    if config.root == 'none':
        x_dim = 21 * 6
    elif config.root == 'orientation':
        x_dim = 6 + 21 * 6
    else:
        raise NotImplementedError
    return x_dim


def destruct_x(
    x: torch.Tensor,
    head: torch.Tensor,
    canonical: torch.Tensor,
    config: DictConfig,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Destruct the model output `x` (denoising target, canonicalized pose representation) into root orientation
    and joint orientation. Transform root orientation to world coordinates if necessary.

    Args:
        x (batch, x_dim, seq_length):
            Model output `x` (denoising target, canonicalized pose representation).
        head (batch, seq_length, 4, 4):
            Head transformation in world coordinates.
        canonical (batch, seq_length, 4, 4):
            Canonical transformation in world coordinates.
        config (DictConfig):
            Representation configuration.

    Returns:
        root_orient (batch, seq_length, 3, 3) or None:
            Root orientation in world coordinates or None if model does not predict root orientation.
        joint_orient (batch, seq_length, 21, 3, 3):
            Joint orientations in local coordinates.
    """

    x = x.permute(0, 2, 1)
    B, L = x.shape[:2]
    if config.root == 'orientation':
        root_orient = rotation_6d_to_matrix(x[:, :, :6])
        joint_orient = rotation_6d_to_matrix(x[:, :, 6:].reshape(B, L, 21, 6))
    elif config.root == 'none':
        root_orient = None
        joint_orient = rotation_6d_to_matrix(x[:, :, :].reshape(B, L, 21, 6))
    else:
        raise NotImplementedError

    if config.reference == 'head':
        if root_orient is not None:
            root_orient = head[:, :, :3, :3] @ root_orient
    elif config.reference == 'canonical':
        if root_orient is not None:
            root_orient = canonical[:, :, :3, :3] @ root_orient
    elif config.reference != 'none':
        raise NotImplementedError

    return root_orient, joint_orient


def move_to_world(
    root_T: torch.Tensor,
    root_orient: Optional[torch.Tensor],
    head: torch.Tensor,
) -> torch.Tensor:
    """
    Transform the skeleton from root (pelvis) coordinates to world coordinates.
    If model does not predict root orientation, calculate root orientation by aligning the head joint with the input
    head pose.

    Args:
        root_T (batch, seq_length, 22, 4, 4):
            Global joint transformation in root (pelvis) coordinates.
        root_orient (batch, seq_length, 3, 3) or None:
            Root orientation in world coordinates or None if model does not predict root orientation.
        head (batch, seq_length, 4, 4):
            Head transformation in world coordinates.

    Returns:
        global_T (batch, seq_length, 22, 4, 4):
            Global joint transformation in world coordinates.
    """

    B, L = root_T.shape[:2]

    if root_orient is None:
        root_orient = head[:, :, :3, :3] @ inv_R_tensor(root_T[:, :, constants.HEAD, :3, :3])
    root_transl = head[:, :, :3, 3] - (root_orient @ root_T[:, :, constants.HEAD, :3, 3].unsqueeze(-1)).squeeze(-1)

    root_transform = torch.zeros((B, L, 4, 4), device=root_T.device)    # T_world_root
    root_transform[:, :, :3, :3] = root_orient
    root_transform[:, :, :3, 3] = root_transl
    root_transform[:, :, 3, 3] = 1.0
    global_T = root_transform[:, :, None, :, :] @ root_T

    return global_T


def x_to_world(
    x: torch.Tensor,
    betas: torch.Tensor,
    canonical: torch.Tensor,
    head: torch.Tensor,
    body_model: BodyModel,
    config: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align the model output `x` (denoising target, canonicalized pose representation) into skeleton in world
    coordinates, as described in Sec 3.2 (Global Alignment) in the paper.

    Args:
        x (batch, x_dim, seq_length):
            Model output `x` (denoising target, canonicalized pose representation).
        betas (batch, 16):
            Body shape parameters. (predicted)
        canonical (batch, seq_length, 4, 4):
            Canonical transformation in world coordinates.
        head (batch, seq_length, 4, 4):
            Head transformation in world coordinates.
        body_model (BodyModel):
            SMPL-H body model.
        config (DictConfig):
            Representation configuration.

    Returns:
        global_T (batch, seq_length, 22, 4, 4):
            Global joint transformation in world coordinates.
        local_T (batch, seq_length, 22, 4, 4):
            Local joint transformation in world coordinates (joint orientation in upper 3x3 and joint offset in
            upper 3x1).
    """

    root_orient, joint_orient = destruct_x(x, head, canonical, config)
    root_T, skel_offset = forward_kinematics(joint_orient, betas, body_model)
    global_T = move_to_world(root_T, root_orient, head)
    local_T = torch.zeros_like(global_T)
    local_T[:, :, 0] = global_T[:, :, 0]
    local_T[:, :, 1:, :3, :3] = joint_orient
    local_T[:, :, 1:, :3, 3] = skel_offset[:, None, :, :]
    local_T[:, :, 1:, 3, 3] = 1.0
    return global_T, local_T
