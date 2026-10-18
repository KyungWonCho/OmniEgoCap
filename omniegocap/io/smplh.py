"""
Model output: SMPL-H parameters in the AMASS layout.

File `<name>.npz`:
    poses               (L, 156)    float32   axis-angle: pelvis rotation in world (3), 21 body joints (63, local),
                                              hands (90, zeros: fingers are not predicted)
    trans               (L, 3)      float32   SMPL-H `trans` = pelvis position in world - J0(betas)
    betas               (16,)       float32   neutral SMPL-H shape
    gender              ()          str       'neutral'
    mocap_framerate     ()          float     30
    frame_valid         (L,)        bool      False for frame 0, which is a copy of frame 1
    meta                ()          str       JSON (checkpoint, seed, guidance, runtime, ...)

For the SMPL-H body model: root_orient = poses[:, :3], pose_body = poses[:, 3:66].

The model predicts frames 1 .. L - 1 of an L-frame input (frame 0 only provides the previous head pose), so frame 0
is filled with frame 1 and marked invalid. Output length always equals input length.
"""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel

from omniegocap.utils.constants import FPS
from omniegocap.utils.transforms import matrix_to_axis_angle


class SMPLHSequence:
    def __init__(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        betas: np.ndarray,
        frame_valid: np.ndarray,
        fps: float = FPS,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        SMPL-H motion of one sequence.

        Args:
            trans (L, 3):
                SMPL-H translation (pelvis position in world minus the rest pelvis position J0).
            root_orient (L, 3):
                Pelvis rotation in world, axis-angle.
            pose_body (L, 63):
                Body joint rotations (local), axis-angle.
            betas (16,):
                Neutral SMPL-H shape parameters.
            frame_valid (L,):
                Whether each frame is a model prediction.
            fps (float):
                Frame rate.
            meta (Dict[str, Any] or None):
                Free-form information.
        """

        self.trans = np.asarray(trans, dtype=np.float32)
        self.root_orient = np.asarray(root_orient, dtype=np.float32)
        self.pose_body = np.asarray(pose_body, dtype=np.float32)
        self.betas = np.asarray(betas, dtype=np.float32)
        self.frame_valid = np.asarray(frame_valid, dtype=np.bool_)
        self.fps = float(fps)
        self.meta = meta or {}


    def __len__(self) -> int:
        return self.trans.shape[0]


    @classmethod
    def from_transforms(
        cls,
        root_orient: np.ndarray,
        root_transl: np.ndarray,
        joint_orient: np.ndarray,
        betas: np.ndarray,
        body_model: BodyModel,
        frame_valid: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> 'SMPLHSequence':
        """
        Convert rotation matrices and the pelvis position to SMPL-H parameters.

        Args:
            root_orient (L, 3, 3):
                Pelvis rotation in world.
            root_transl (L, 3):
                Pelvis position in world.
            joint_orient (L, 21, 3, 3):
                Local body joint rotations.
            betas (16,):
                Neutral SMPL-H shape parameters.
            body_model (BodyModel):
                Neutral SMPL-H body model, used for the rest pelvis position J0(betas).
            frame_valid (L,) or None:
                Whether each frame is valid. If None, all frames are.
            meta (Dict[str, Any] or None):
                Free-form information.

        Returns:
            result (SMPLHSequence):
                Converted sequence.
        """

        device = body_model.shapedirs.device
        betas_t = torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(device)[None]
        with torch.no_grad():
            J0 = body_model(betas=betas_t).Jtr[0, 0].cpu().numpy()                     # (3,)

        rotations = torch.from_numpy(np.ascontiguousarray(np.concatenate([root_orient[:, None], joint_orient], axis=1)))
        axis_angle = matrix_to_axis_angle(rotations).numpy()                          # (L, 22, 3)
        L = axis_angle.shape[0]

        return cls(
            trans=root_transl - J0[None],
            root_orient=axis_angle[:, 0],
            pose_body=axis_angle[:, 1:].reshape(L, 63),
            betas=betas,
            frame_valid=np.ones(L, dtype=np.bool_) if frame_valid is None else frame_valid,
            meta=meta,
        )


    @classmethod
    def from_local_T(
        cls,
        local_T: np.ndarray,
        betas: np.ndarray,
        body_model: BodyModel,
        meta: Optional[Dict[str, Any]] = None,
    ) -> 'SMPLHSequence':
        """
        Convert the model output (frames 1 .. L - 1) to SMPL-H parameters for all L frames.

        Args:
            local_T (L - 1, 22, 4, 4):
                Root transformation in world (joint 0) and local joint rotations (joints 1 .. 21).
            betas (16,):
                Predicted shape parameters.
            body_model (BodyModel):
                Neutral SMPL-H body model, used for the rest pelvis position J0(betas).
            meta (Dict[str, Any] or None):
                Free-form information.

        Returns:
            result (SMPLHSequence):
                Converted sequence, frame 0 copied from frame 1 and marked invalid.
        """

        # frame 0 is not predicted: repeat frame 1
        local_T = np.concatenate([local_T[:1], local_T], axis=0)                     # (L, 22, 4, 4)
        frame_valid = np.ones(local_T.shape[0], dtype=np.bool_)
        frame_valid[0] = False

        return cls.from_transforms(
            root_orient=local_T[:, 0, :3, :3],
            root_transl=local_T[:, 0, :3, 3],
            joint_orient=local_T[:, 1:, :3, :3],
            betas=betas,
            body_model=body_model,
            frame_valid=frame_valid,
            meta=meta,
        )


    @classmethod
    def load(cls, path: str) -> 'SMPLHSequence':
        """
        Load an SMPL-H npz file (this format, or an AMASS SMPL+H file).

        Args:
            path (str):
                Path to the npz file.

        Returns:
            result (SMPLHSequence):
                Loaded sequence.
        """

        data = np.load(path, allow_pickle=False)
        L = data['poses'].shape[0]

        return cls(
            trans=data['trans'],
            root_orient=data['poses'][:, :3],
            pose_body=data['poses'][:, 3:66],
            betas=data['betas'][:16],
            frame_valid=data['frame_valid'] if 'frame_valid' in data else np.ones(L, dtype=np.bool_),
            fps=float(data['mocap_framerate']) if 'mocap_framerate' in data else FPS,
            meta=json.loads(str(data['meta'])) if 'meta' in data else {},
        )


    def save(self, path: str) -> None:
        """
        Save as an SMPL-H npz file.

        Args:
            path (str):
                Output path.
        """

        pose_hand = np.zeros((len(self), 90), dtype=np.float32)
        np.savez_compressed(
            path,
            poses=np.concatenate([self.root_orient, self.pose_body, pose_hand], axis=1),
            trans=self.trans,
            betas=self.betas,
            gender=np.str_('neutral'),
            mocap_framerate=np.float32(self.fps),
            frame_valid=self.frame_valid,
            meta=np.str_(json.dumps(self.meta)),
        )


    def forward(self, body_model: BodyModel) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the SMPL-H body model.

        Args:
            body_model (BodyModel):
                Neutral SMPL-H body model.

        Returns:
            joints (L, 22, 3):
                Body joint positions in world coordinates.
            vertices (L, V, 3):
                Mesh vertices in world coordinates.
        """

        device = body_model.shapedirs.device
        with torch.no_grad():
            body = body_model(
                root_orient=torch.from_numpy(self.root_orient).to(device),
                pose_body=torch.from_numpy(self.pose_body).to(device),
                trans=torch.from_numpy(self.trans).to(device),
                betas=torch.from_numpy(self.betas).to(device)[None],
            )

        return body.Jtr[:, :22].cpu().numpy(), body.v.cpu().numpy()
