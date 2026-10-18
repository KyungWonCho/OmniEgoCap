"""
AMASS benchmark (Tab. 1 and 2 in the paper): evaluation h5 made by scripts/preprocess_amass.py from the AMASS test
split, with the wrist visibility of each setting baked in.

Each sequence group stores frames 1 .. N of an (N + 1)-frame clip and `head_full` adds frame 0, so a sequence is
fed to the model exactly like a user input of N + 1 frames, and the prediction of frames 1 .. N is scored.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm

from omniegocap.core.inference import OmniEgoCap
from omniegocap.io.sequence import EgoSequence
from omniegocap.io.smplh import SMPLHSequence
from omniegocap.utils import constants
from omniegocap.utils.metrics import jerk, position_error, velocity_error

HANDS = [constants.L_WRIST, constants.R_WRIST]
METRICS = {
    'MPJPE': 'mm',
    'MPJVE': 'cm/s',
    'Jerk': 'km/s^3',
    'Hand PE': 'mm',
    'Vis Hand PE': 'mm',
    'Height': 'cm',
    'Span': 'cm',
}


def load_input(group: h5py.Group) -> EgoSequence:
    """
    Model input of one evaluation sequence, in world coordinates.

    Args:
        group (h5py.Group):
            Sequence group of an evaluation h5.

    Returns:
        seq (EgoSequence):
            N + 1 frames; frame 0 has no wrist observation.
    """

    head = group['head_full'][()]                                                   # (N + 1, 4, 4)
    wrist = head[1:, None] @ group['wrist'][()]                                     # (N, 2, 4, 4)
    wrist = np.concatenate([np.tile(np.eye(4, dtype=np.float32), (1, 2, 1, 1)), wrist], axis=0)
    visible = np.concatenate([np.zeros((1, 2), dtype=np.bool_), group['visible'][()]], axis=0)

    return EgoSequence(head=head, wrist=wrist, visible=visible)


def load_gt(
    group: h5py.Group,
    body_model: BodyModel,
) -> SMPLHSequence:
    """
    Ground truth motion of frames 1 .. N.

    Args:
        group (h5py.Group):
            Sequence group of an evaluation h5.
        body_model (BodyModel):
            Neutral SMPL-H body model.

    Returns:
        gt (SMPLHSequence):
            Ground truth, N frames.
    """

    return SMPLHSequence.from_transforms(
        root_orient=group['root_orient'][()],
        root_transl=group['root_transl'][()],
        joint_orient=group['joint_orient'][()],
        betas=group['betas'][()],
        body_model=body_model,
    )


def body_size(
    betas: np.ndarray,
    body_model: BodyModel,
) -> Tuple[float, float]:
    """
    Height and arm span of the rest-pose mesh.

    Args:
        betas (16,):
            Shape parameters.
        body_model (BodyModel):
            Neutral SMPL-H body model.

    Returns:
        height (float):
            Vertical extent (meters).
        span (float):
            Left-right extent (meters).
    """

    device = body_model.shapedirs.device
    with torch.no_grad():
        v = body_model(betas=torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(device)[None]).v[0]

    return (v[:, 1].max() - v[:, 1].min()).item(), (v[:, 0].max() - v[:, 0].min()).item()


def score(
    pred: SMPLHSequence,
    gt: SMPLHSequence,
    visible: np.ndarray,
    body_model: BodyModel,
) -> Dict[str, Tuple[float, int]]:
    """
    Metrics of one sequence as (sum, count), so they can be accumulated over a set.

    Args:
        pred (SMPLHSequence):
            Prediction for the N + 1 input frames (frame 0 is not scored).
        gt (SMPLHSequence):
            Ground truth of frames 1 .. N.
        visible (N, 2):
            Wrist visibility of frames 1 .. N.
        body_model (BodyModel):
            Neutral SMPL-H body model.

    Returns:
        metrics (Dict[str, Tuple[float, int]]):
            Metric name -> (sum in the unit of METRICS, count).
    """

    N = len(gt)
    if len(pred) != N + 1:
        raise ValueError(f'prediction has {len(pred)} frames, expected {N + 1} (the input length)')

    pred_pos = pred.forward(body_model)[0][1:]                                      # (N, 22, 3)
    gt_pos = gt.forward(body_model)[0]                                              # (N, 22, 3)
    hand_dist = np.linalg.norm(pred_pos[:, HANDS] - gt_pos[:, HANDS], axis=-1)      # (N, 2)

    pred_height, pred_span = body_size(pred.betas, body_model)
    gt_height, gt_span = body_size(gt.betas, body_model)

    return {
        'MPJPE': (position_error(pred_pos, gt_pos).sum() * 1000, N),
        'MPJVE': (velocity_error(pred_pos, gt_pos, constants.FPS).sum() * 100, N - 1),
        'Jerk': (jerk(pred_pos, constants.FPS).sum() / 1000, N - 3),
        'Hand PE': (hand_dist.mean(axis=-1).sum() * 1000, N),
        'Vis Hand PE': (hand_dist[visible].sum() * 1000, int(visible.sum())),
        'Height': (abs(pred_height - gt_height) * 100, 1),
        'Span': (abs(pred_span - gt_span) * 100, 1),
    }


def evaluate(
    h5_path: str,
    model: Optional[OmniEgoCap] = None,
    body_model: Optional[BodyModel] = None,
    pred_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    optimization: bool = True,
    guidance_scale: float = 30.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Score a model, or saved SMPL-H npz predictions named `<sequence>.npz`, on one evaluation h5.

    Args:
        h5_path (str):
            Evaluation h5.
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
            Seed of the initial noise of every sequence.

    Returns:
        result (Dict[str, Any]):
            setting (the h5's eval_setting), num_sequences, and the mean of every metric and of the runtime.
    """

    assert model is not None or pred_dir is not None, 'Either a model or a prediction folder is needed.'
    body_model = body_model or model.body_model
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    totals = {name: [0.0, 0] for name in METRICS}
    runtimes = []
    with h5py.File(h5_path, 'r') as h5:
        setting = json.loads(h5.attrs['eval_setting']) if 'eval_setting' in h5.attrs else {}
        names = list(h5.keys())
        for name in tqdm(names):
            group = h5[name]
            if pred_dir is not None:
                pred = SMPLHSequence.load(os.path.join(pred_dir, f'{name}.npz'))
            else:
                pred = model.predict(
                    load_input(group),
                    optimization=optimization,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                if save_dir is not None:
                    pred.save(os.path.join(save_dir, f'{name}.npz'))
            if 'runtime_s' in pred.meta:
                runtimes.append(pred.meta['runtime_s'])

            metrics = score(pred, load_gt(group, body_model), group['visible'][()], body_model)
            for key, (value, count) in metrics.items():
                totals[key][0] += value
                totals[key][1] += count

    result = {'setting': setting, 'num_sequences': len(names)}
    result.update({key: float(value / max(count, 1)) for key, (value, count) in totals.items()})
    result['Runtime'] = float(np.mean(runtimes)) if runtimes else float('nan')

    return result


def export_inputs(
    h5_path: str,
    out_dir: str,
) -> int:
    """
    Write the model input of every sequence as an input npz, for running other methods on the same data.

    Args:
        h5_path (str):
            Evaluation h5.
        out_dir (str):
            Output folder; files are named `<sequence>.npz`.

    Returns:
        count (int):
            Number of files written.
    """

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, 'r') as h5:
        for name in tqdm(list(h5.keys())):
            load_input(h5[name]).save(os.path.join(out_dir, f'{name}.npz'))

        return len(h5.keys())
