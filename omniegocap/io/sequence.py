"""
Model input: one egocentric sequence as head and wrist poses in a common world frame.

File `<name>.npz` (numpy only):
    head         (L, 4, 4)   float32   required   head frame H in world (SMPL-H head joint; x left, y up, z forward)
    wrist        (L, 2, 4, 4) float32  required   left / right wrist frame in world (SMPL-H wrist joints)
    visible      (L, 2)      bool      required   whether each wrist is observed; invisible wrists are ignored
    device_pose  (L, 4, 4)   float32   optional   device (glasses, camera) pose in world, for the viewer
    points       (N, 3)      float32   optional   scene point cloud in world, for the viewer
    raw_time_s   (L,)        float64   optional   time of each frame on the raw recording's clock (seconds), for the
                                                  viewer's video panel (Aria: device time of the RGB frame)
    device       ()          str       optional   'aria', 'gopro', 'quest', ...
    meta         ()          str       optional   JSON

Frames run at 30 fps (resample before saving).
World: meters, right-handed, z up (against gravity), floor at z = 0. Horizontal origin and heading are arbitrary.
"""

import json
import warnings
from typing import Any, Dict, Optional

import numpy as np

OPTIONAL = ('device_pose', 'points', 'raw_time_s')


class EgoSequence:
    def __init__(
        self,
        head: np.ndarray,
        wrist: np.ndarray,
        visible: np.ndarray,
        device_pose: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        raw_time_s: Optional[np.ndarray] = None,
        device: str = '',
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Input sequence for OmniEgoCap.

        Args:
            head (L, 4, 4):
                Head transformation in world coordinates.
            wrist (L, 2, 4, 4):
                Left and right wrist transformation in world coordinates.
            visible (L, 2):
                Visibility of left and right wrists.
            device_pose (L, 4, 4) or None:
                Device pose in world coordinates (viewer only).
            points (N, 3) or None:
                Scene point cloud in world coordinates (viewer only).
            raw_time_s (L,) or None:
                Time of each frame on the raw recording's clock in seconds (viewer only).
            device (str):
                Capture device name.
            meta (Dict[str, Any] or None):
                Free-form information (adapter, source files, ...).
        """

        self.head = np.asarray(head, dtype=np.float32)
        self.wrist = np.asarray(wrist, dtype=np.float32)
        self.visible = np.asarray(visible, dtype=np.bool_)
        self.device_pose = None if device_pose is None else np.asarray(device_pose, dtype=np.float32)
        self.points = None if points is None else np.asarray(points, dtype=np.float32)
        self.raw_time_s = None if raw_time_s is None else np.asarray(raw_time_s, dtype=np.float64)
        self.device = device
        self.meta = meta or {}


    def __len__(self) -> int:
        return self.head.shape[0]


    @classmethod
    def load(cls, path: str) -> 'EgoSequence':
        """
        Load an input npz file and validate it.

        Args:
            path (str):
                Path to the npz file.

        Returns:
            seq (EgoSequence):
                Loaded sequence.
        """

        data = np.load(path, allow_pickle=False)
        for key in ('head', 'wrist', 'visible'):
            if key not in data:
                raise ValueError(f'{path}: missing required key "{key}"')

        seq = cls(
            head=data['head'],
            wrist=data['wrist'],
            visible=data['visible'],
            device=str(data['device']) if 'device' in data else '',
            meta=json.loads(str(data['meta'])) if 'meta' in data else {},
            **{key: data[key] for key in OPTIONAL if key in data},
        )
        seq.validate(name=path)

        return seq


    def save(self, path: str) -> None:
        """
        Validate and save as an input npz file.

        Args:
            path (str):
                Output path.
        """

        self.validate(name=path)
        data = {
            'head': self.head,
            'wrist': self.wrist,
            'visible': self.visible,
            'device': np.str_(self.device),
            'meta': np.str_(json.dumps(self.meta)),
        }
        for key in OPTIONAL:
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        np.savez_compressed(path, **data)


    def validate(self, name: str = 'sequence') -> None:
        """
        Check the input contract. Raise ValueError on violations, warn on suspicious values.

        Args:
            name (str):
                Name used in the messages.
        """

        L = self.head.shape[0]
        if self.head.shape != (L, 4, 4):
            raise ValueError(f'{name}: head must be (L, 4, 4), got {self.head.shape}')
        if self.wrist.shape != (L, 2, 4, 4):
            raise ValueError(f'{name}: wrist must be (L, 2, 4, 4), got {self.wrist.shape}')
        if self.visible.shape != (L, 2):
            raise ValueError(f'{name}: visible must be (L, 2), got {self.visible.shape}')
        if L < 2:
            raise ValueError(f'{name}: at least 2 frames are needed, got {L}')

        poses = {
            'head': self.head,
            'wrist': self.wrist[self.visible],
        }
        if self.device_pose is not None:
            if self.device_pose.shape != (L, 4, 4):
                raise ValueError(f'{name}: device_pose must be (L, 4, 4), got {self.device_pose.shape}')
            poses['device_pose'] = self.device_pose
        for key, T_pose in poses.items():
            if not np.isfinite(T_pose).all():
                raise ValueError(f'{name}: {key} has non-finite values')
            if np.abs(T_pose[..., 3, :] - np.array([0.0, 0.0, 0.0, 1.0])).max(initial=0.0) > 1e-5:
                raise ValueError(f'{name}: {key} last row must be [0, 0, 0, 1]')
            R = T_pose[..., :3, :3].astype(np.float64)
            ortho = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max(initial=0.0)
            if ortho > 1e-3 or (np.linalg.det(R) <= 0).any():
                raise ValueError(f'{name}: {key} rotation is not orthonormal (max |R^T R - I| = {ortho:.2e})')

        if self.raw_time_s is not None and self.raw_time_s.shape != (L,):
            raise ValueError(f'{name}: raw_time_s must be (L,), got {self.raw_time_s.shape}')
        if self.points is not None and (self.points.ndim != 2 or self.points.shape[1] != 3):
            raise ValueError(f'{name}: points must be (N, 3), got {self.points.shape}')

        height = np.median(self.head[:, 2, 3])
        if not 0.8 <= height <= 2.3:
            warnings.warn(f'{name}: median head height is {height:.2f} m; is the floor at z = 0?')
        if not self.visible.any():
            warnings.warn(f'{name}: no wrist is visible in any frame')
