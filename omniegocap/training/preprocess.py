"""
AMASS -> h5 for training and evaluation. Run through scripts/preprocess_amass.py.

Every set is one h5 file (one group per sequence), named after its setting:
    train.h5                                 training split, no visibility (made by augmentation during training)
    test_diverse.h5                          Tab. 1: geometry-aware augmentation + temporal drops + perturbation
    test_fov{h}x{v}[_tilt{t}].h5             Tab. 2 and custom cameras: rectangular field of view in head coordinates

Head related
    - head         : (L, 4, 4)
    - head_delta   : (L, 4, 4)
    - canonical    : (L, 4, 4)
    - yaw_delta    : (L,)
    - head_full    : (L + 1, 4, 4)

Wrist related
    - wrist        : (L, 2, 4, 4)
    - visible      : (L, 2)

Body related
    - betas        : (16,)
    - root_transl  : (L, 3)        pelvis position (floor at z = 0)
    - root_orient  : (L, 3, 3)
    - joint_orient : (L, 21, 3, 3)

Loss related
    - contact      : (L, 2)
    - joint_pos    : (L, 22, 3)
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from omniegocap.core.features import make_features
from omniegocap.training.augment import augment
from omniegocap.utils import constants
from omniegocap.utils.paths import repo_path, smplh_path
from omniegocap.utils.transforms import axis_angle_to_matrix

# floor detection and the AMASS fixes in load_sequence() follow HuMoR (THIRD_PARTY_NOTICES.md)
FOOT_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
SEED = 42

# Tab. 1 test set. Fixed here (a copy of conf/augment/defaults.yaml at submission) so that changing the training
# augmentation does not silently change the evaluation set.
DIVERSE_AUGMENT = {
    'mode': 'full',
    'geometry': {
        'yaw_half_range': [0.35, 2.15],
        'pitch_half_range': [0.35, 1.35],
        'yaw_offset_range': [-0.15, 0.15],
        'pitch_offset_min': 0.0,
        'pitch_offset_max_at_min': 1.5,
        'pitch_offset_max_at_max': 0.4,
        'ratio_min': 0.4,
        'ratio_max': 1.1,
        'p_range': [2.0, 10.0],
    },
    'drop': {
        'short': {'rate': 0.05, 'mean': 2.0, 'std': 1.0, 'min_len': 1},
        'long': {'rate': 0.1, 'mean': 28.0, 'std': 25.0, 'min_len': 5},
    },
    'perturbation': {'rotation': 0.017453292519943, 'translation': 0.005},
}


def camera_setting(
    fov_h: float,
    fov_v: float,
    tilt: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluation setting of a camera with a rectangular field of view.

    Args:
        fov_h (float):
            Horizontal field of view in degree.
        fov_v (float):
            Vertical field of view in degree.
        tilt (float):
            Downward tilt in degree (negative = upward).

    Returns:
        setting (Dict[str, Any]):
            Setting dictionary.
    """

    return {'split': 'test', 'visibility': 'fov', 'fov_h': float(fov_h), 'fov_v': float(fov_v), 'tilt': float(tilt)}


PAPER_SETS = {
    'train': {'split': 'train', 'visibility': 'none'},
    'diverse': {'split': 'test', 'visibility': 'augment', 'augment': DIVERSE_AUGMENT, 'seed': SEED},
    'fov90': camera_setting(90, 90),
    'fov120': camera_setting(120, 120),
    'fov180': camera_setting(180, 180),
    'fov120x60': camera_setting(120, 60),
    'tilt30': camera_setting(120, 120, 30),
}


def setting_filename(setting: Dict[str, Any]) -> str:
    """
    File name of a setting: train.h5, test_diverse.h5, or test_fov{h}x{v}[_tilt{t}].h5 (decimal point -> 'p').

    Args:
        setting (Dict[str, Any]):
            Setting dictionary.

    Returns:
        name (str):
            File name.
    """

    def number(value: float) -> str:
        return f'{value:g}'.replace('.', 'p')

    if setting['split'] == 'train':
        return 'train.h5'
    if setting['visibility'] == 'augment':
        return 'test_diverse.h5'
    name = f'test_fov{number(setting["fov_h"])}x{number(setting["fov_v"])}'
    if setting['tilt'] != 0:
        name += f'_tilt{number(setting["tilt"])}'

    return name + '.h5'


def in_fov(
    points: np.ndarray,
    fov_h: float,
    fov_v: float,
    tilt_angle: float = 0.0,
) -> np.ndarray:
    """
    Check points are in the field of view of the camera with given horizontal and vertical FOV and tilt angle.
    Positive tilt angle means looking downwards.

    Args:
        points (N, 3):
            Points in head coordinate.
        fov_h (float):
            Horizontal FOV in degree.
        fov_v (float):
            Vertical FOV in degree.
        tilt_angle (float):
            Tilt angle in degree.

    Returns:
        in_fov (N,):
            Boolean array indicating whether each point is visible.
    """

    if tilt_angle != 0:
        theta = np.deg2rad(-tilt_angle)
        c, s = np.cos(theta), np.sin(theta)
        y, z = points[:, 1], points[:, 2]
        y_new, z_new = c * y - s * z, s * y + c * z
        x, y, z = points[:, 0], y_new, z_new
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

    in_horizontor = np.arccos(z / np.sqrt(np.square(x) + np.square(z))) <= np.deg2rad(fov_h / 2)
    in_vertical = np.arccos(z / np.sqrt(np.square(y) + np.square(z))) <= np.deg2rad(fov_v / 2)
    in_fov = in_horizontor & in_vertical
    return in_fov


class PreprocessOmniEgoCap:
    def __init__(
        self,
        amass_dir: str,
        setting: Dict[str, Any],
        out_dir: str = repo_path('data', 'preprocess'),
        smplh_dir: Optional[str] = None,
    ):
        """
        Preprocess one set of AMASS sequences into one h5 file.

        Args:
            amass_dir (str):
                AMASS root (SMPL+H G); the split lists in data/amass are relative to it.
            setting (Dict[str, Any]):
                A value of PAPER_SETS or camera_setting(...).
            out_dir (str):
                Output folder; the file name comes from setting_filename.
            smplh_dir (str or None):
                Folder of the SMPL-H model files (see omniegocap.utils.paths).
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(out_dir, exist_ok=True)

        self.setting = setting
        with open(repo_path('data', 'amass', f'{setting["split"]}.txt'), 'r') as f:
            self.files = [os.path.join(amass_dir, line.strip()) for line in f.readlines()]
        self.output = os.path.join(out_dir, setting_filename(setting))

        self.male_weight, self.male_bias = self.precompute_affine_transform(
            gender_model=smplh_path('male', smplh_dir),
            neutral_model=smplh_path('neutral', smplh_dir),
        )
        self.female_weight, self.female_bias = self.precompute_affine_transform(
            gender_model=smplh_path('female', smplh_dir),
            neutral_model=smplh_path('neutral', smplh_dir),
        )
        self.body_model = BodyModel(bm_fname=smplh_path('neutral', smplh_dir), num_betas=16).to(self.device)


    @staticmethod
    def precompute_affine_transform(
        gender_model: str,
        neutral_model: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute the affine transform for converting betas between gender-specific model and neutral model.

        Args:
            gender_model (str):
                Path to the gender-specific SMPL-H model npz file.
            neutral_model (str):
                Path to the neutral SMPL-H model npz file.

        Returns:
            weight (16, 16):
                Weight matrix for the affine transform.
            bias (16,):
                Bias vector for the affine transform.
        """

        gender_model = np.load(gender_model, allow_pickle=True)
        neutral_model = np.load(neutral_model, allow_pickle=True)

        neutral_shapedirs = neutral_model['shapedirs'].reshape(-1, 16)
        v_template_delta = (gender_model['v_template'] - neutral_model['v_template']).reshape(-1)
        gender_shapedirs = gender_model['shapedirs'].reshape(-1, 16)
        inv = np.linalg.pinv(neutral_shapedirs)

        return inv @ gender_shapedirs, inv @ v_template_delta


    def compute_neutral_betas(
        self,
        gender: str,
        gender_betas: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the neutral betas from the gender-specific betas using the precomputed affine transform.

        Args:
            gender (str):
                Gender of the subject, either 'male' or 'female'.
            gender_betas (16,):
                Betas for the gender-specific model.

        Returns:
            betas_neutral (16,):
                Betas for the neutral model.
        """

        if gender == 'male':
            betas_neutral = self.male_weight @ gender_betas + self.male_bias
        elif gender == 'female':
            betas_neutral = self.female_weight @ gender_betas + self.female_bias
        else:
            raise ValueError(f'Unknown gender: {gender}')

        return betas_neutral


    def forward_kinematics(
        self,
        trans: np.ndarray,
        poses: np.ndarray,
        betas: np.ndarray,
    ) -> np.ndarray:
        """
        Perform forward kinematics to get global joint transformation in world coordinates.

        Args:
            trans (L, 3):
                Root joint translation.
            poses (L, 66):
                Joint rotations in axis-angle representation.
            betas (16,):
                Body shape parameters.

        Returns:
            global_T (L, 22, 4, 4):
                Global joint transformation in world coordinates.
        """

        num_frames = trans.shape[0]

        trans = torch.from_numpy(trans).float().to(self.device)
        poses = torch.from_numpy(poses).float().to(self.device)
        betas = torch.from_numpy(betas).float().unsqueeze(0).to(self.device)
        tpose = self.body_model(betas=betas)

        offset = [None]
        parent = self.body_model.kintree_table[0].cpu().numpy()
        for joint in range(1, 22):
            offset.append(tpose.Jtr[0, joint] - tpose.Jtr[0, parent[joint]])
        global_T = [torch.zeros((num_frames, 4, 4), device=self.device)]
        local_R = axis_angle_to_matrix(poses.reshape(-1, 22, 3))
        global_T[0][:, :3, 3] = trans
        global_T[0][:, :3, :3] = local_R[:, 0]
        global_T[0][:, 3, 3] = 1.0
        for joint in range(1, 22):
            cur_T = torch.zeros((num_frames, 4, 4), device=self.device)
            cur_T[:, :3, :3] = global_T[parent[joint]][:, :3, :3] @ local_R[:, joint]
            cur_T[:, :3, 3] = global_T[parent[joint]][:, :3, :3] @ offset[joint] + global_T[parent[joint]][:, :3, 3]
            cur_T[:, 3, 3] = 1.0
            global_T.append(cur_T)
        global_T = torch.stack(global_T, dim=1)   # (num_frames, 22, 4, 4)

        return global_T.cpu().numpy()


    def adjust_floor(
        self,
        global_T: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Adjust the floor height to be z=0 and annotate foot contact based on foot velocity.

        Args:
            global_T (L, 22, 4, 4):
                Global joint transformation in world coordinates.

        Returns:
            global_T (L, 22, 4, 4):
                Global joint transformation in world coordinates after floor adjustment.
            contact (L, 2):
                Foot contact label for left and right foot.
            offset_floor_height (float):
                The height that has been subtracted from the original global_T to adjust the floor.
        """

        left_foot_seq = global_T[:, constants.L_FOOT, :3, 3]
        right_foot_seq = global_T[:, constants.R_FOOT, :3, 3]
        left_foot_vel = np.linalg.norm(left_foot_seq[1:] - left_foot_seq[:-1], axis=1)
        left_foot_vel = np.append(left_foot_vel, left_foot_vel[-1])
        right_foot_vel = np.linalg.norm(right_foot_seq[1:] - right_foot_seq[:-1], axis=1)
        right_foot_vel = np.append(right_foot_vel, right_foot_vel[-1])
        left_foot_height = left_foot_seq[:, 2]
        right_foot_height = right_foot_seq[:, 2]

        left_static_foot_heights = left_foot_height[left_foot_vel < FOOT_VEL_THRESH]
        right_static_foot_heights = right_foot_height[right_foot_vel < FOOT_VEL_THRESH]
        all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)

        if all_static_foot_heights.shape[0] > 0:
            clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
            all_labels = np.unique(clustering.labels_)
            min_median = float('inf')
            for cur_label in all_labels:
                cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
                cur_median = np.median(cur_clust)

                if cur_median < min_median:
                    min_median = cur_median
            floor_height = min_median
            offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET
        else:
            floor_height = offset_floor_height = 0.0

        global_T[:, :, 2, 3] -= offset_floor_height
        contact = np.stack([left_foot_vel < FOOT_VEL_THRESH, right_foot_vel < FOOT_VEL_THRESH], axis=1)

        return global_T, contact, offset_floor_height


    def load_sequence(
        self,
        file: str,
    ) -> np.ndarray:
        """
        Load and preprocess a single sequence from the given file.

        Args:
            file (str):
                Path to the npz file containing the sequence data.
        Returns:
            data (dict):
                A dictionary containing the preprocessed data for the sequence.
        """

        data = np.load(file, allow_pickle=True)

        fps = int(data['mocap_framerate'])
        trans, poses, gender, betas = data['trans'][:], data['poses'][:, :66], data['gender'].item(), data['betas'][:16]
        if isinstance(gender, bytes):
            gender = gender.decode()

        # correct mislabled data
        if file.find('BMLhandball') >= 0:
            fps = 240
        if file.find('20160930_50032') >= 0 or file.find('20161014_50033') >= 0:
            fps = 59

        # cleanup data
        if file.find('BioMotionLab_NTroje') >= 0 and (file.find('treadmill') >= 0 or file.find('normal_') >= 0):
            return None
        if file.find('MPI_HDM05') >= 0 and file.find('dg/HDM_dg_07-01') >= 0:
            return None

        # only keep middle 80% of sequences to avoid redundant static poses
        num_frames = trans.shape[0]
        sidx, eidx = int(num_frames * 0.1), int(num_frames * 0.9)
        trans, poses = trans[sidx:eidx], poses[sidx:eidx]
        num_frames = eidx - sidx

        fps_ratio = 30 / fps
        new_num_frames = int(fps_ratio * num_frames)
        indices = np.linspace(0, num_frames - 1, new_num_frames, dtype=int)
        trans, poses = trans[indices], poses[indices]
        num_frames = new_num_frames

        if num_frames <= 128:
            return None

        betas = self.compute_neutral_betas(gender, betas)
        global_T = self.forward_kinematics(trans, poses, betas)
        global_T, contact, offset_floor_height = self.adjust_floor(global_T)

        head_full_T = global_T[:, constants.HEAD]                       # (num_frames, 4, 4)
        wrist_full_T = global_T[:, [constants.L_WRIST, constants.R_WRIST]]  # (num_frames, 2, 4, 4)

        # model input (same features as inference, frames 1 .. num_frames - 1)
        features = make_features(head_full_T, wrist_full_T, np.ones((num_frames, 2), dtype=np.bool_))
        wrist_T = features['wrist']                                     # (num_frames - 1, 2, 4, 4), head coordinate

        # model output
        root_transl = trans[1:]                                         # (num_frames - 1, 3)
        root_transl[:, 2] -= offset_floor_height
        root_orient = axis_angle_to_matrix(                             # (num_frames - 1, 3, 3)
            torch.from_numpy(poses[1:, :3]).float()
        ).numpy()
        joint_orient = axis_angle_to_matrix(                            # (num_frames - 1, 21, 3, 3)
            torch.from_numpy(poses[1:, 3:66].reshape(-1, 21, 3)).float()
        ).numpy()

        # additional information
        contact = contact[1:]                                           # (num_frames - 1, 2)
        joint_pos = global_T[1:, :, :3, 3]                              # (num_frames - 1, 22, 3)

        data = {
            'head': features['head'],
            'head_delta': features['head_delta'],
            'canonical': features['canonical'],
            'yaw_delta': features['yaw_delta'],
            'wrist': wrist_T,
            'betas': betas.astype(np.float32),
            'root_transl': root_transl.astype(np.float32),
            'root_orient': root_orient.astype(np.float32),
            'joint_orient': joint_orient.astype(np.float32),
            'contact': contact.astype(np.bool_),
            'joint_pos': joint_pos.astype(np.float32),
        }

        if self.setting['split'] != 'train':
            data['head_full'] = head_full_T.astype(np.float32)

            if self.setting['visibility'] == 'augment':
                wrist_T, visible = augment(
                    wrist=torch.from_numpy(wrist_T).float().unsqueeze(0),       # (1, num_frames - 1, 2, 4, 4)
                    length=torch.tensor([num_frames - 1], dtype=torch.long),    # (1,)
                    config=OmegaConf.create(self.setting['augment']),
                )
                data['wrist'] = wrist_T.numpy().astype(np.float32)[0]
                data['visible'] = visible.numpy().astype(np.bool_)[0]
            elif self.setting['visibility'] == 'fov':
                visible = in_fov(
                    wrist_T[:, :, :3, 3].reshape(-1, 3),
                    fov_h=self.setting['fov_h'],
                    fov_v=self.setting['fov_v'],
                    tilt_angle=self.setting['tilt'],
                ).reshape(-1, 2)                                                # (num_frames - 1, 2)
                data['visible'] = visible.astype(np.bool_)
            else:
                raise NotImplementedError

        return data


    @staticmethod
    def get_name_from_path(file: str) -> str:
        split = file.split('/')
        return split[-3] + '_' + split[-2] + '_' + split[-1].replace('.npz', '')

    def run(self) -> None:
        with h5py.File(self.output, 'w') as f:
            f.attrs['eval_setting'] = json.dumps(self.setting)
            for file in tqdm(self.files):
                name = self.get_name_from_path(file)
                data = self.load_sequence(file)
                if data is None:
                    continue

                grp = f.create_group(name)
                for key, item in data.items():
                    grp.create_dataset(key, data=item)
