"""
Per-frame error terms shared by the AMASS and OmniEgoDB benchmarks. All inputs are positions in meters.
"""

import numpy as np


def position_error(
    pred: np.ndarray,
    gt: np.ndarray,
) -> np.ndarray:
    """
    Mean joint position error of each frame.

    Args:
        pred (L, J, 3):
            Predicted joint positions.
        gt (L, J, 3):
            Ground truth joint positions.

    Returns:
        error (L,):
            Mean Euclidean distance over joints (meters).
    """

    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def velocity_error(
    pred: np.ndarray,
    gt: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Mean joint velocity error between consecutive frames.

    Args:
        pred (L, J, 3):
            Predicted joint positions.
        gt (L, J, 3):
            Ground truth joint positions.
        fps (float):
            Frame rate.

    Returns:
        error (L - 1,):
            Mean velocity error over joints (meters per second).
    """

    pred_vel = (pred[1:] - pred[:-1]) * fps                         # (L - 1, J, 3)
    gt_vel = (gt[1:] - gt[:-1]) * fps                               # (L - 1, J, 3)

    return np.linalg.norm(pred_vel - gt_vel, axis=-1).mean(axis=-1)


def jerk(
    pred: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Mean jerk magnitude (third finite difference of position).

    Args:
        pred (L, J, 3):
            Joint positions.
        fps (float):
            Frame rate.

    Returns:
        jerk (L - 3,):
            Mean jerk over joints (meters per second cubed).
    """

    jerk = pred[3:] - 3 * pred[2:-1] + 3 * pred[1:-2] - pred[:-3]   # (L - 3, J, 3)

    return np.linalg.norm(jerk * fps ** 3, axis=-1).mean(axis=-1)
