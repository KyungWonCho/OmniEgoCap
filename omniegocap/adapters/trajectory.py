"""
Any head-mounted device: a camera trajectory plus wrist poses from a hand estimator -> model input.

Usage
python scripts/prepare_input.py trajectory --trajectory traj.txt --wrist wrist.npz --cam_to_head cam_to_head.txt \
    --output seq_input.npz [--wrist_frame camera] [--floor_z 0.0] [--points points.npy] [--device gopro]

Inputs
    --trajectory   TUM text (`time tx ty tz qx qy qz qw` per line, camera-to-world, seconds) or npz with
                   `T_world_cam` (N, 4, 4) and `time_s` (N,). The world must be metric with z up (gravity);
                   --floor_z is subtracted so that the floor ends up at z = 0. Monocular SLAM is up to scale and
                   has an arbitrary up direction: fix both before using it here. The viewer finds video frames
                   by these timestamps, so use the video's clock (seconds from its first frame) to see the video.
    --wrist        npz with `wrist` (M, 2, 4, 4) and `visible` (M, 2) for the left and right wrist, and `time_s` (M,)
                   (optional; without it the rows must match the trajectory rows one to one).
                   The wrist axes follow the SMPL-H wrist joint (see omniegocap.adapters.geometry.
                   wrist_frame_from_palm). The OmniEgoDB GoPro inputs used HaWoR's MANO `global_orient` and `transl`
                   as the camera-frame wrist transformation as they are.
    --wrist_frame  'camera' (default; composed with the trajectory at the wrist's own time) or 'world'.
    --cam_to_head  4 x 4 transformation from the head frame to the camera frame (T_cam_head: x_cam = T @ x_head),
                   as 16 numbers in a text file or a .npy file. The head frame is the SMPL-H head joint
                   (x left, y up, z forward).

Output frames run at 30 fps over the trajectory: its own timestamps if it already runs at 30 fps, otherwise a 30 Hz
grid with interpolated camera poses. Each frame takes the nearest wrist sample within half a frame (1/60 s).
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from omniegocap.adapters.base import Adapter, register
from omniegocap.adapters.geometry import check_fps, interpolate_T, make_T, nearest_index
from omniegocap.io.sequence import EgoSequence
from omniegocap.utils.constants import FPS


def read_trajectory(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Camera trajectory from a TUM text file or an npz.

    Args:
        path (str):
            Trajectory file.

    Returns:
        T_world_cam (N, 4, 4):
            Camera-to-world transformations.
        time_s (N,):
            Timestamps in seconds.
    """

    if path.endswith('.npz'):
        data = np.load(path)
        return data['T_world_cam'].astype(np.float64), data['time_s'].astype(np.float64)

    rows = np.loadtxt(path, comments='#', ndmin=2)                                             # (N, 8)
    T_world_cam = make_T(Rotation.from_quat(rows[:, 4:8]).as_matrix(), rows[:, 1:4])

    return T_world_cam, rows[:, 0]


def read_matrix(path: str) -> np.ndarray:
    """
    A 4 x 4 transformation from a .npy file or a text file with 16 numbers.

    Args:
        path (str):
            File path.

    Returns:
        T (4, 4):
            Transformation.
    """

    T = np.load(path) if path.endswith('.npy') else np.loadtxt(path)

    return np.asarray(T, dtype=np.float64).reshape(4, 4)


def load_trajectory(
    trajectory_path: str,
    wrist_path: str,
    cam_to_head_path: str,
    wrist_frame: str = 'camera',
    floor_z: float = 0.0,
    points_path: Optional[str] = None,
    device: str = '',
) -> EgoSequence:
    """
    Build the model input from a camera trajectory and wrist poses.

    Args:
        trajectory_path (str):
            Trajectory file (see the module docstring).
        wrist_path (str):
            Wrist npz.
        cam_to_head_path (str):
            T_cam_head file.
        wrist_frame (str):
            'camera' or 'world'.
        floor_z (float):
            Floor height in the trajectory's world, subtracted from every position.
        points_path (str or None):
            Optional (P, 3) .npy point cloud in the trajectory's world, for the viewer.
        device (str):
            Device name; the viewer draws the camera mesh of 'aria' and 'gopro'.

    Returns:
        seq (EgoSequence):
            Model input; raw_time_s is the trajectory time of every frame.
    """

    T_world_cam, traj_t = read_trajectory(trajectory_path)
    T_cam_head = read_matrix(cam_to_head_path)
    wrist_data = np.load(wrist_path)
    wrist_src = wrist_data['wrist'].astype(np.float64)                                         # (M, 2, 4, 4)
    visible_src = wrist_data['visible'].astype(np.bool_)                                       # (M, 2)
    if 'time_s' in wrist_data:
        wrist_t = wrist_data['time_s'].astype(np.float64)
    elif len(wrist_src) == len(traj_t):
        wrist_t = traj_t
    else:
        raise ValueError(f'{wrist_path} has no time_s and {len(wrist_src)} rows for {len(traj_t)} trajectory rows')

    # 30 fps frames
    if check_fps(traj_t) is None:
        frame_t = traj_t
        T_frame = T_world_cam
    else:
        frame_t = traj_t[0] + np.arange(int(np.floor((traj_t[-1] - traj_t[0]) * FPS)) + 1) / FPS
        T_frame = interpolate_T(T_world_cam, traj_t, frame_t)                                  # (L, 4, 4)

    # wrists in world, at their own time
    if wrist_frame == 'camera':
        wrist_world = interpolate_T(T_world_cam, traj_t, wrist_t)[:, None] @ wrist_src         # (M, 2, 4, 4)
    elif wrist_frame == 'world':
        wrist_world = wrist_src
    else:
        raise ValueError(f'Unknown wrist frame: {wrist_frame}')
    j = nearest_index(wrist_t, frame_t)                                                        # (L,)
    visible = visible_src[j] & (np.abs(wrist_t[j] - frame_t) <= 0.5 / FPS)[:, None]           # (L, 2)
    wrist = wrist_world[j]                                                                     # (L, 2, 4, 4)
    wrist[~visible] = np.eye(4)

    # floor at z = 0
    T_frame = T_frame.copy()
    T_frame[:, 2, 3] -= floor_z
    wrist[visible, 2, 3] -= floor_z
    points = None
    if points_path is not None:
        points = np.load(points_path).astype(np.float32)
        points[:, 2] -= floor_z

    return EgoSequence(
        head=T_frame @ T_cam_head,
        wrist=wrist,
        visible=visible,
        device_pose=T_frame,
        points=points,
        raw_time_s=frame_t,
        device=device,
        meta={
            'adapter': 'trajectory',
            'trajectory': os.path.basename(trajectory_path),
            'wrist': os.path.basename(wrist_path),
            'wrist_frame': wrist_frame,
            'floor_z': floor_z,
        },
    )


@register
class Trajectory(Adapter):
    name = 'trajectory'


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--trajectory', type=str, required=True, help='TUM text or npz (T_world_cam, time_s).')
        parser.add_argument('--wrist', type=str, required=True, help='npz with wrist, visible (and time_s).')
        parser.add_argument('--cam_to_head', type=str, required=True, help='T_cam_head: 16 numbers or .npy.')
        parser.add_argument('--wrist_frame', type=str, default='camera', choices=['camera', 'world'])
        parser.add_argument('--floor_z', type=float, default=0.0, help='Floor height in the trajectory world.')
        parser.add_argument('--points', type=str, default=None, help='Optional (P, 3) .npy point cloud.')
        parser.add_argument('--device', type=str, default='', help='Device name (aria, gopro: mesh in the viewer).')


    def load(self, args: argparse.Namespace) -> EgoSequence:
        return load_trajectory(
            trajectory_path=args.trajectory,
            wrist_path=args.wrist,
            cam_to_head_path=args.cam_to_head,
            wrist_frame=args.wrist_frame,
            floor_z=args.floor_z,
            points_path=args.points,
            device=args.device,
        )
