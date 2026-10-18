"""
Geometry helpers shared by the adapters. Everything works on whole sequences at once.
"""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from omniegocap.utils.constants import FPS


def make_T(
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Stack rotations and translations into homogeneous transformations.

    Args:
        R (..., 3, 3):
            Rotations.
        t (..., 3):
            Translations.

    Returns:
        T (..., 4, 4):
            Transformations.
    """

    T = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    return T


def nearest_index(
    t_src: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """
    Index of the nearest source time for every query time (ties go to the earlier sample).

    Args:
        t_src (N,):
            Sorted source times.
        t_query (M,):
            Query times.

    Returns:
        index (M,):
            Index into t_src.
    """

    upper = np.clip(np.searchsorted(t_src, t_query, side='left'), 1, len(t_src) - 1)
    lower = upper - 1
    index = np.where(np.abs(t_src[lower] - t_query) <= np.abs(t_src[upper] - t_query), lower, upper)
    index[t_query <= t_src[0]] = 0
    index[t_query >= t_src[-1]] = len(t_src) - 1

    return index


def interpolate_T(
    T: np.ndarray,
    t_src: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate rigid transformations in time: rotations by slerp, translations linearly. Query times outside the
    source range are clamped to the ends.

    Args:
        T (N, 4, 4):
            Transformations at the source times.
        t_src (N,):
            Sorted source times.
        t_query (M,):
            Query times.

    Returns:
        T_query (M, 4, 4):
            Interpolated transformations.
    """

    t_query = np.clip(t_query, t_src[0], t_src[-1])
    R = Slerp(t_src, Rotation.from_matrix(T[:, :3, :3]))(t_query).as_matrix()
    t = np.stack([np.interp(t_query, t_src, T[:, i, 3]) for i in range(3)], axis=-1)

    return make_T(R, t)


def wrist_frame_from_palm(
    wrist_pos: np.ndarray,
    palm_pos: np.ndarray,
    palm_normal: np.ndarray,
    side: str,
) -> np.ndarray:
    """
    SMPL-H wrist frame from the wrist and palm points and the palm normal: y = -(palm normal), x along the hand
    (wrist -> palm for the left hand, palm -> wrist for the right hand) made orthogonal to y, z = x cross y.
    In T-pose these axes match the body axes (x left, y up, z forward).

    Args:
        wrist_pos (N, 3):
            Wrist positions.
        palm_pos (N, 3):
            Palm center positions.
        palm_normal (N, 3):
            Palm normals.
        side (str):
            'left' or 'right'.

    Returns:
        wrist_T (N, 4, 4):
            Wrist transformations.
    """

    y = -palm_normal / np.linalg.norm(palm_normal, axis=-1, keepdims=True)
    if side == 'left':
        x = palm_pos - wrist_pos
    elif side == 'right':
        x = wrist_pos - palm_pos
    else:
        raise ValueError(f'Unknown side: {side}')
    x = x - y * (x * y).sum(axis=-1, keepdims=True)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    return make_T(np.stack([x, y, z], axis=-1), wrist_pos)


def find_floor_z(
    points: np.ndarray,
    band: float = 0.02,
    step: float = 0.01,
) -> float:
    """
    Floor height from a point cloud: the most populated height (within +- band) among the lowest 10% of points.

    Args:
        points (N, 3):
            Point cloud in a z-up world.
        band (float):
            Half width of the inlier band (meters).
        step (float):
            Spacing of the candidate heights (meters).

    Returns:
        floor_z (float):
            Mean height of the inliers of the best candidate.
    """

    z = np.sort(points[:, 2])
    z = z[:int(0.1 * len(z))]

    candidates = np.arange(np.min(z), np.max(z), step)
    count = np.searchsorted(z, candidates + band, side='left') - np.searchsorted(z, candidates - band, side='right')
    best_z = candidates[np.argmax(count)] if len(candidates) else 0.0

    return float(np.mean(z[np.abs(z - best_z) < band]))


def check_fps(
    t: np.ndarray,
    tolerance: float = 0.05,
) -> Optional[str]:
    """
    Whether timestamps run at 30 fps.

    Args:
        t (N,):
            Timestamps in seconds.
        tolerance (float):
            Allowed relative deviation of the median frame interval.

    Returns:
        message (str or None):
            None if the rate matches, else a description of the measured rate.
    """

    measured = 1.0 / np.median(np.diff(t))
    if abs(measured - FPS) > tolerance * FPS:
        return f'{measured:.2f} Hz'

    return None
