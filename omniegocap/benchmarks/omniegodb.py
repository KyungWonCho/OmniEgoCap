"""
OmniEgoDB benchmark (Tab. 3 in the paper): real captures with Xsens motion capture as ground truth.

Layout of a capture folder `<root>/<device>/<capture>/`:
    input.npz       model input (omniegocap.io.sequence), wrists from the device's own hand tracking
    input_rgb.npz   Aria only: same, wrists estimated from the RGB video
    gt.npz          Xsens segments in the same world and on the same frames as the input
                        segments (L, 23, 4, 4), segment_names (23,), valid (L,), meta
    raw recording   recording.vrs + mps/ (Aria), video.MP4 (GoPro), poselog.csv (Quest)

Tracks (device folder, input file): aria (aria, input.npz), aria_rgb (aria, input_rgb.npz), gopro, quest.

Scoring compares 15 SMPL-H joints with the matching Xsens segment origins (no alignment) on the frames where both
the ground truth and the prediction are valid; captures are averaged weighted by their number of scored frames.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm

from omniegocap.core.inference import OmniEgoCap
from omniegocap.io.sequence import EgoSequence
from omniegocap.io.smplh import SMPLHSequence
from omniegocap.utils import constants
from omniegocap.utils.metrics import jerk, position_error, velocity_error

TRACKS = {
    'aria': ('aria', 'input.npz'),
    'aria_rgb': ('aria', 'input_rgb.npz'),
    'gopro': ('gopro', 'input.npz'),
    'quest': ('quest', 'input.npz'),
}

# (SMPL-H joint, Xsens segment); the last two are the hands
JOINT_PAIRS = [
    (0, 'Pelvis'),
    (1, 'LeftUpperLeg'),
    (2, 'RightUpperLeg'),
    (4, 'LeftLowerLeg'),
    (5, 'RightLowerLeg'),
    (7, 'LeftFoot'),
    (8, 'RightFoot'),
    (12, 'Neck'),
    (15, 'Head'),
    (16, 'LeftUpperArm'),
    (17, 'RightUpperArm'),
    (18, 'LeftForeArm'),
    (19, 'RightForeArm'),
    (20, 'LeftHand'),
    (21, 'RightHand'),
]

# Xsens skeleton, child -> parent
SEGMENT_PARENT = {
    'L5': 'Pelvis',
    'L3': 'L5',
    'T12': 'L3',
    'T8': 'T12',
    'Neck': 'T8',
    'Head': 'Neck',
    'RightShoulder': 'T8',
    'RightUpperArm': 'RightShoulder',
    'RightForeArm': 'RightUpperArm',
    'RightHand': 'RightForeArm',
    'LeftShoulder': 'T8',
    'LeftUpperArm': 'LeftShoulder',
    'LeftForeArm': 'LeftUpperArm',
    'LeftHand': 'LeftForeArm',
    'RightUpperLeg': 'Pelvis',
    'RightLowerLeg': 'RightUpperLeg',
    'RightFoot': 'RightLowerLeg',
    'RightToe': 'RightFoot',
    'LeftUpperLeg': 'Pelvis',
    'LeftLowerLeg': 'LeftUpperLeg',
    'LeftFoot': 'LeftLowerLeg',
    'LeftToe': 'LeftFoot',
}

METRICS = {
    'MPJPE': 'mm',
    'MPJVE': 'cm/s',
    'Jerk': 'km/s^3',
    'Hand PE': 'mm',
    'Vis Hand PE': 'mm',
}


class GroundTruth:
    def __init__(
        self,
        segments: np.ndarray,
        segment_names: List[str],
        valid: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Xsens ground truth of one capture.

        Args:
            segments (L, 23, 4, 4):
                Segment transformations in world coordinates, on the input frames.
            segment_names (List[str]):
                Xsens segment names.
            valid (L,):
                Frames to score.
            meta (Dict[str, Any] or None):
                Synchronization with the raw mocap file and other information.
        """

        self.segments = np.asarray(segments, dtype=np.float32)
        self.segment_names = list(segment_names)
        self.valid = np.asarray(valid, dtype=np.bool_)
        self.meta = meta or {}


    def __len__(self) -> int:
        return self.segments.shape[0]


    @classmethod
    def load(cls, path: str) -> 'GroundTruth':
        data = np.load(path, allow_pickle=False)

        return cls(
            segments=data['segments'],
            segment_names=[str(name) for name in data['segment_names']],
            valid=data['valid'],
            meta=json.loads(str(data['meta'])) if 'meta' in data else {},
        )


    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            segments=self.segments,
            segment_names=np.array(self.segment_names),
            valid=self.valid,
            meta=np.str_(json.dumps(self.meta)),
        )


    def bones(self) -> List[Tuple[int, int]]:
        """
        Skeleton edges as (parent index, child index).
        """

        index = {name: i for i, name in enumerate(self.segment_names)}

        return [(index[parent], index[child]) for child, parent in SEGMENT_PARENT.items()
                if child in index and parent in index]


def list_captures(
    root: str,
    track: str,
) -> List[Tuple[str, str]]:
    """
    Captures of one track.

    Args:
        root (str):
            OmniEgoDB root.
        track (str):
            One of TRACKS.

    Returns:
        captures (List[Tuple[str, str]]):
            (capture name, capture folder) pairs that have the track's input and gt.npz.
    """

    device, input_name = TRACKS[track]
    folders = sorted(glob.glob(os.path.join(root, device, '*', input_name)))

    return [(os.path.basename(os.path.dirname(path)), os.path.dirname(path)) for path in folders
            if os.path.exists(os.path.join(os.path.dirname(path), 'gt.npz'))]


def score(
    pred: SMPLHSequence,
    gt: GroundTruth,
    visible: np.ndarray,
    body_model: BodyModel,
) -> Dict[str, float]:
    """
    Metrics of one capture.

    Args:
        pred (SMPLHSequence):
            Prediction on the input frames.
        gt (GroundTruth):
            Ground truth on the input frames.
        visible (L, 2):
            Wrist visibility of the input the prediction was made from.
        body_model (BodyModel):
            Neutral SMPL-H body model.

    Returns:
        metrics (Dict[str, float]):
            METRICS plus `num_frames` (scored frames) and `Runtime` (seconds, if recorded).
    """

    if len(pred) != len(gt):
        raise ValueError(f'prediction has {len(pred)} frames, ground truth {len(gt)}')
    scored = np.flatnonzero(gt.valid & pred.frame_valid)
    if len(scored) < 4 or np.any(np.diff(scored) != 1):
        raise ValueError('scored frames must be one contiguous block of at least 4 frames')

    index = {name: i for i, name in enumerate(gt.segment_names)}
    smplh_idx = [joint for joint, _ in JOINT_PAIRS]
    xsens_idx = [index[name] for _, name in JOINT_PAIRS]
    pred_pos = pred.forward(body_model)[0][scored][:, smplh_idx]                    # (F, 15, 3)
    gt_pos = gt.segments[scored][:, xsens_idx][..., :3, 3]                         # (F, 15, 3)
    hand_dist = np.linalg.norm(pred_pos[:, -2:] - gt_pos[:, -2:], axis=-1)          # (F, 2)
    hand_visible = visible[scored].astype(np.bool_)                                # (F, 2)

    return {
        'num_frames': len(scored),
        'MPJPE': float(position_error(pred_pos, gt_pos).mean() * 1000),
        'MPJVE': float(velocity_error(pred_pos, gt_pos, constants.FPS).mean() * 100),
        'Jerk': float(jerk(pred_pos, constants.FPS).mean() / 1000),
        'Hand PE': float(hand_dist.mean() * 1000),
        'Vis Hand PE': float(hand_dist[hand_visible].mean() * 1000) if hand_visible.any() else float('nan'),
        'Runtime': float(pred.meta.get('runtime_s', float('nan'))),
    }


def evaluate(
    root: str,
    track: str,
    model: Optional[OmniEgoCap] = None,
    body_model: Optional[BodyModel] = None,
    pred_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    optimization: bool = True,
    guidance_scale: float = 30.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Score a model, or saved predictions `<pred_dir>/<capture>.npz`, on one track.

    Args:
        root (str):
            OmniEgoDB root.
        track (str):
            One of TRACKS.
        model (OmniEgoCap or None):
            Model to run. Ignored if pred_dir is given.
        body_model (BodyModel or None):
            Neutral SMPL-H body model. If None, the model's.
        pred_dir (str or None):
            Folder of saved predictions to score instead of running the model.
        save_dir (str or None):
            Folder to save the model's predictions to.
        optimization (bool):
            Whether to run the guidance optimization.
        guidance_scale (float):
            Guidance scale for optimization.
        seed (int):
            Seed of the initial noise of every capture.

    Returns:
        result (Dict[str, Any]):
            `captures`: per-capture metrics; `mean`: frame-weighted mean of every metric.
    """

    assert model is not None or pred_dir is not None, 'Either a model or a prediction folder is needed.'
    body_model = body_model or model.body_model
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    input_name = TRACKS[track][1]

    captures = {}
    for name, folder in tqdm(list_captures(root, track)):
        seq = EgoSequence.load(os.path.join(folder, input_name))
        if pred_dir is not None:
            pred = SMPLHSequence.load(os.path.join(pred_dir, f'{name}.npz'))
        else:
            pred = model.predict(seq, optimization=optimization, guidance_scale=guidance_scale, seed=seed)
            if save_dir is not None:
                pred.save(os.path.join(save_dir, f'{name}.npz'))
        captures[name] = score(pred, GroundTruth.load(os.path.join(folder, 'gt.npz')), seq.visible, body_model)

    # frame-weighted mean over the captures where the metric is defined
    mean = {}
    for key in list(METRICS) + ['Runtime']:
        values = np.array([metrics[key] for metrics in captures.values()], dtype=np.float64)
        frames = np.array([metrics['num_frames'] for metrics in captures.values()], dtype=np.float64)
        defined = np.isfinite(values)
        mean[key] = float((values * frames)[defined].sum() / frames[defined].sum()) if defined.any() else float('nan')

    return {'captures': captures, 'mean': mean}
