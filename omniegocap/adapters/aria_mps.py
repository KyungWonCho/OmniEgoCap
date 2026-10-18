"""
Aria (Gen 1) recording with Machine Perception Services (MPS) output -> model input.

Usage
python scripts/prepare_input.py aria_mps --vrs recording.vrs --mps_dir mps/ --output seq_input.npz

Frames follow the RGB camera (30 fps) between the first and the last pose of the closed-loop trajectory.
    head    CPF (central pupil frame) pose, moved to the SMPL-H head joint with constants.T_cpf_head
    wrist   MPS hand tracking, used when its confidence is >= 0.9 and it is within 1/60 s of the frame
    points  semidense point cloud (confidence-filtered); its floor defines z = 0
"""

import argparse
import os

import numpy as np

from omniegocap.adapters.base import Adapter, register
from omniegocap.adapters.geometry import check_fps, find_floor_z, nearest_index, wrist_frame_from_palm
from omniegocap.io.sequence import EgoSequence
from omniegocap.utils import constants

HAND_CONFIDENCE = 0.9
HAND_MAX_OFFSET_NS = 1e9 / 60


def read_hands(
    hand_tracking: list,
    side: str,
) -> dict:
    """
    One hand of the MPS hand tracking results as arrays (NaN / -1 where the hand is missing).

    Args:
        hand_tracking (list):
            Results of mps.hand_tracking.read_hand_tracking_results.
        side (str):
            'left' or 'right'.

    Returns:
        Dictionary containing the following items:
            confidence (N,):
                tracking confidence, -1 if missing
            wrist (N, 3), palm (N, 3), normal (N, 3):
                wrist and palm positions and palm normal in device coordinates
    """

    hands = [getattr(result, f'{side}_hand') for result in hand_tracking]
    missing = np.full(3, np.nan)

    return {
        'confidence': np.array([-1.0 if hand is None else hand.confidence for hand in hands]),
        'wrist': np.array([missing if hand is None else hand.get_wrist_position_device() for hand in hands]),
        'palm': np.array([missing if hand is None else hand.get_palm_position_device() for hand in hands]),
        'normal': np.array([
            missing if hand is None else hand.wrist_and_palm_normal_device.palm_normal_device for hand in hands
        ]),
    }


def load_aria_mps(
    vrs_path: str,
    mps_dir: str,
) -> EgoSequence:
    """
    Read an Aria recording and its MPS output.

    Args:
        vrs_path (str):
            Path to the VRS file.
        mps_dir (str):
            MPS output folder (slam/ and hand_tracking/).

    Returns:
        seq (EgoSequence):
            Model input; raw_time_s is the device time of each RGB frame.
    """

    # projectaria_tools is only needed for Aria recordings
    from projectaria_tools.core import mps
    from projectaria_tools.core.data_provider import create_vrs_data_provider
    from projectaria_tools.core.mps import utils as mps_utils
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    provider = create_vrs_data_provider(vrs_path)
    mps_paths = mps.MpsDataPathsProvider(mps_dir).get_data_paths()
    trajectory = mps.read_closed_loop_trajectory(mps_paths.slam.closed_loop_trajectory)
    points = mps_utils.filter_points_from_confidence(mps.read_global_point_cloud(mps_paths.slam.semidense_points))
    points = np.array([p.position_world for p in points], dtype=np.float32)                  # (P, 3)
    hand_tracking = mps.hand_tracking.read_hand_tracking_results(mps_paths.hand_tracking.hand_tracking_results)

    # trajectory and hand tracking as arrays
    traj_t = np.array([pose.tracking_timestamp.total_seconds() * 1e9 for pose in trajectory])  # (N,)
    traj_T = np.stack([pose.transform_world_device.to_matrix() for pose in trajectory])        # (N, 4, 4)
    hand_t = np.array([result.tracking_timestamp.total_seconds() * 1e9 for result in hand_tracking])

    # RGB frames inside the trajectory
    rgb_stream = provider.get_stream_id_from_label('camera-rgb')
    start_idx = provider.get_index_by_time_ns(
        rgb_stream,
        int(traj_t[0]),
        TimeDomain.DEVICE_TIME,
        TimeQueryOptions.AFTER,
    )
    end_idx = provider.get_index_by_time_ns(
        rgb_stream,
        int(traj_t[-1]),
        TimeDomain.DEVICE_TIME,
        TimeQueryOptions.BEFORE,
    )
    if start_idx + 1 > end_idx:
        raise ValueError(f'The trajectory covers fewer than 2 RGB frames ({start_idx} .. {end_idx})')
    rgb_index = np.arange(start_idx, end_idx + 1)                                              # (L,)
    frame_t = np.array(provider.get_timestamps_ns(rgb_stream, TimeDomain.DEVICE_TIME))[rgb_index].astype(np.float64)
    rate = check_fps(frame_t / 1e9)
    if rate is not None:
        raise ValueError(f'The RGB stream runs at {rate}; record at 30 fps')

    # head
    T_device_cpf = provider.get_device_calibration().get_transform_device_cpf().to_matrix()
    T_world_device = traj_T[nearest_index(traj_t, frame_t)]                                    # (L, 4, 4)
    cpf = T_world_device @ T_device_cpf                                                        # (L, 4, 4)

    # wrists
    hand_idx = nearest_index(hand_t, frame_t)                                                  # (L,)
    close = np.abs(hand_t[hand_idx] - frame_t) <= HAND_MAX_OFFSET_NS                            # (L,)
    R, t = T_world_device[:, :3, :3], T_world_device[:, :3, 3]
    wrist, visible = [], []
    for side in ('left', 'right'):
        hand = {key: value[hand_idx] for key, value in read_hands(hand_tracking, side).items()}
        side_visible = close & (hand['confidence'] >= HAND_CONFIDENCE)                         # (L,)
        side_wrist = wrist_frame_from_palm(
            wrist_pos=np.einsum('lij,lj->li', R, hand['wrist']) + t,
            palm_pos=np.einsum('lij,lj->li', R, hand['palm']) + t,
            palm_normal=np.einsum('lij,lj->li', R, hand['normal']),
            side=side,
        )                                                                                      # (L, 4, 4)
        side_wrist[~side_visible] = np.eye(4)
        wrist.append(side_wrist)
        visible.append(side_visible)
    wrist = np.stack(wrist, axis=1)                                                            # (L, 2, 4, 4)
    visible = np.stack(visible, axis=1)                                                        # (L, 2)

    # floor at z = 0
    floor_z = find_floor_z(points)
    cpf[:, 2, 3] -= floor_z
    wrist[visible, 2, 3] -= floor_z
    points[:, 2] -= floor_z

    head = cpf @ np.array(constants.T_cpf_head)

    return EgoSequence(
        head=head,
        wrist=wrist,
        visible=visible,
        device_pose=cpf,
        points=points,
        raw_time_s=frame_t / 1e9,
        device='aria',
        meta={
            'adapter': 'aria_mps',
            'vrs': os.path.basename(vrs_path),
            'floor_z': floor_z,
        },
    )


@register
class AriaMPS(Adapter):
    name = 'aria_mps'


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--vrs', type=str, required=True, help='Path to the VRS file.')
        parser.add_argument('--mps_dir', type=str, required=True, help='MPS output folder (slam/, hand_tracking/).')


    def load(self, args: argparse.Namespace) -> EgoSequence:
        return load_aria_mps(args.vrs, args.mps_dir)
