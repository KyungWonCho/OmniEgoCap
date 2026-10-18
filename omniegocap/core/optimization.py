import math

import theseus as th
import torch

from omniegocap.utils import constants


def optimize_inner_batch(
    data: torch.Tensor,
    noise_level: float,
    guidance_scale: float = 30.0,
) -> torch.Tensor:
    """
    Optimize per-frame shoulder, elbow, wrist joint rotation to satisfy wrist observations using timestep-adaptive
    optimization. Highly parallelizable due to individual frame optimization.
    Use theseus library for Levenberg-Marquardt optimization.

    Args:
        data (N, 6, 4, 4):
            Packed input data for optimization
                data[:, 0]: collar global transform
                data[:, 1]: wrist global transform
                data[:, 2]: shoulder local transform
                data[:, 3]: elbow local transform
                data[:, 4]: wrist local transform
                data[:, 5]: wrist observation
        noise_level (float):
            Noise level for diffusion model, between 0 and 1. Higher means noisier.
        guidance_scale (float):
            Guidance scale for optimization. Higher means more weight on observation.

    Returns:
        answer (N, 3, 3, 3):
            Optimized local joint rotations for shoulder, elbow, wrist.
    """

    device = data.device
    N = data.shape[0]

    # prepare variables
    shoulder_th = th.SO3(tensor=data[:, 2, :3, :3], name='shoulder')
    elbow_th = th.SO3(tensor=data[:, 3, :3, :3], name='elbow')
    wrist_th = th.SO3(tensor=data[:, 4, :3, :3], name='wrist')

    collar_th = th.SE3(tensor=data[:, 0, :3, :4], name='collar')
    predict_th = th.SE3(tensor=data[:, 1, :3, :4], name='predict')
    observ_th = th.SE3(tensor=data[:, 5, :3, :4], name='observ')
    offset_th = th.Variable(tensor=data[:, 2:5, :3, 3], name='offset')

    # build objective
    objective = th.Objective().to(device)

    def fk_se3_fn(optim_vars, aux_vars):
        shoulder, elbow, wrist = optim_vars
        collar, predict, observ, offset = aux_vars

        shoulder_orient = collar.tensor[:, :3, :3] @ shoulder.tensor
        shoulder_transl = (collar.tensor[:, :3, :3] @ offset[:, 0, :].unsqueeze(-1)).squeeze(-1) \
            + collar.tensor[:, :3, 3]
        elbow_orient = shoulder_orient @ elbow.tensor
        elbow_transl = (shoulder_orient @ offset[:, 1, :].unsqueeze(-1)).squeeze(-1) + shoulder_transl
        wrist_transform = torch.zeros_like(predict.tensor)
        wrist_transform[:, :3, :3] = elbow_orient @ wrist.tensor
        wrist_transform[:, :3, 3] = (elbow_orient @ offset[:, 2, :].unsqueeze(-1)).squeeze(-1) + elbow_transl
        wrist_transform = th.SE3(tensor=wrist_transform)

        # weights (original loss)
        # noise_level = 1 - alpha_bar  -> alpha_bar = 1 - noise_level
        alpha_bar = 1.0 - float(noise_level)
        w_prior = max(alpha_bar, 0.0) ** 0.25
        w_obs = max(float(noise_level), 0.0) ** 0.25 * math.sqrt(float(guidance_scale))

        # residuals (SE3 local -> 6D)
        residual_predict = wrist_transform.local(predict) * w_prior
        residual_observ = wrist_transform.local(observ) * w_obs

        residual = torch.cat([residual_predict, residual_observ], dim=1)    # (B, 12)

        return residual

    cost_function = th.AutoDiffCostFunction(
        [shoulder_th, elbow_th, wrist_th],
        fk_se3_fn,
        dim=12,
        cost_weight=th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float32, device=device)),
        aux_vars=[collar_th, predict_th, observ_th, offset_th],
        name='wrist_observ_cost'
    )
    objective.add(cost_function)

    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=5,
        vectorize=True
    )
    inputs = {
        'shoulder': shoulder_th,
        'elbow': elbow_th,
        'wrist': wrist_th,
        'collar': collar_th,
        'predict': predict_th,
        'observ': observ_th,
        'offset': offset_th,
    }
    optimizer.objective.update(inputs)
    optimizer.optimize()

    answer = torch.zeros((N, 3, 3, 3), device=device)
    answer[:, 0] = shoulder_th.tensor
    answer[:, 1] = elbow_th.tensor
    answer[:, 2] = wrist_th.tensor

    return answer


def optimize_inner(
    global_T: torch.Tensor,
    local_T: torch.Tensor,
    observ: torch.Tensor,
    visible: torch.Tensor,
    noise_level: float,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Optimize local_T's shoulder, elbow, wrist joint rotation to satisfy wrist observations using timestep-adaptive
    optimization, as described in Sec 3.5 (Inference) and Appendix C (Timestep-Adaptive Guidance Optimization) in the
    paper.

    Args:
        global_T (batch, seq_length, 22, 4, 4):
            Global joint transformations in the world coordinate.
        local_T (batch, seq_length, 22, 4, 4):
            Local joint transformations in the world coordinate.
        observ (batch, seq_length, 2, 4, 4):
            Observed wrist transformations in the world coordinate.
        visible (batch, seq_length, 2):
            Visibility of left and right wrists
        noise_level (float):
            Noise level for diffusion model, between 0 and 1. Higher means noisier.
        guidance_scale (float):
            Guidance scale for optimization. Higher means more weight on observation.

    Returns:
        local_T (batch, seq_length, 22, 4, 4):
            Optimized local joint transformations. Only shoulder, elbow, wrist joints are optimized.
    """

    B, L = local_T.shape[:2]
    device = local_T.device

    # pack inputs
    left_pack = torch.stack([
        global_T[:, :, constants.L_COLLAR],
        global_T[:, :, constants.L_WRIST],
        local_T[:, :, constants.L_SHOULDER],
        local_T[:, :, constants.L_ELBOW],
        local_T[:, :, constants.L_WRIST],
        observ[:, :, 0],
    ], dim=2)   # (B, L, 6, 4, 4)
    right_pack = torch.stack([
        global_T[:, :, constants.R_COLLAR],
        global_T[:, :, constants.R_WRIST],
        local_T[:, :, constants.R_SHOULDER],
        local_T[:, :, constants.R_ELBOW],
        local_T[:, :, constants.R_WRIST],
        observ[:, :, 1],
    ], dim=2)   # (B, L, 6, 4, 4)

    full_pack = torch.stack([left_pack, right_pack], dim=2)     # (B, L, 2, 6, 4, 4)
    mask = visible.reshape(-1)                                  # (B * L * 2,)
    full_pack = full_pack.reshape(-1, 6, 4, 4)                  # (B * L * 2, 6, 4, 4)
    full_pack = full_pack[mask]                                 # (N, 6, 4, 4)

    if full_pack.shape[0] == 0:
        return local_T

    # optimize
    result = optimize_inner_batch(full_pack, noise_level, guidance_scale)       # (N, 3, 3, 3)

    # restore
    result_full = torch.zeros((B * L * 2, 3, 3, 3), device=device)
    result_full[mask] = result
    result_full = result_full.reshape(B, L, 2, 3, 3, 3)

    local_T[:, :, constants.L_SHOULDER, :3, :3] = torch.where(
        visible[:, :, 0, None, None],
        result_full[:, :, 0, 0],
        local_T[:, :, constants.L_SHOULDER, :3, :3]
    )
    local_T[:, :, constants.L_ELBOW, :3, :3] = torch.where(
        visible[:, :, 0, None, None],
        result_full[:, :, 0, 1],
        local_T[:, :, constants.L_ELBOW, :3, :3]
    )
    local_T[:, :, constants.L_WRIST, :3, :3] = torch.where(
        visible[:, :, 0, None, None],
        result_full[:, :, 0, 2],
        local_T[:, :, constants.L_WRIST, :3, :3]
    )
    local_T[:, :, constants.R_SHOULDER, :3, :3] = torch.where(
        visible[:, :, 1, None, None],
        result_full[:, :, 1, 0],
        local_T[:, :, constants.R_SHOULDER, :3, :3]
    )
    local_T[:, :, constants.R_ELBOW, :3, :3] = torch.where(
        visible[:, :, 1, None, None],
        result_full[:, :, 1, 1],
        local_T[:, :, constants.R_ELBOW, :3, :3]
    )
    local_T[:, :, constants.R_WRIST, :3, :3] = torch.where(
        visible[:, :, 1, None, None],
        result_full[:, :, 1, 2],
        local_T[:, :, constants.R_WRIST, :3, :3]
    )

    return local_T
