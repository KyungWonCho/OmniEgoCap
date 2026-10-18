"""
Usage
Any sequence (input, optional predictions and raw recording):
python scripts/visualize.py --input seq_input.npz [--pred seq_smplh.npz other=other_smplh.npz] [--video recording.vrs]

OmniEgoDB captures of one track, switched with a dropdown (input, ground truth, raw recording and metrics are found
in each capture folder):
python scripts/visualize.py --omniegodb OmniEgoDB --track aria [--pred_dir ours=preds/aria/ ...]

    [--port 8080] [--device cuda] [--smplh_dir data/smplh]

Predictions are given as `label=path` or just `path` (label = file or folder name).
"""

import argparse
import os
from typing import Dict, List

from human_body_prior.body_model.body_model import BodyModel

from omniegocap.benchmarks.omniegodb import TRACKS, list_captures
from omniegocap.utils.paths import smplh_path
from omniegocap.vis.viewer import Viewer, capture_loaders, load_scene


def labeled(items: List[str]) -> Dict[str, str]:
    paths = {}
    for item in items:
        label, path = item.split('=', 1) if '=' in item else (None, item)
        paths[label or os.path.splitext(os.path.basename(os.path.normpath(path)))[0]] = path

    return paths


def main(args: argparse.Namespace):
    pred_dirs = labeled(args.pred_dir)
    input_name = TRACKS[args.track][1]
    if args.omniegodb:
        loaders = capture_loaders(list_captures(args.omniegodb, args.track), input_name, pred_dirs)
        if not loaders:
            raise ValueError(f'No {args.track} captures in {args.omniegodb}')
    elif args.input:
        name = os.path.splitext(os.path.basename(args.input))[0]
        preds = labeled(args.pred)
        loaders = {name: lambda: load_scene(name, args.input, preds, video_path=args.video)}
    else:
        raise ValueError('Give --input or --omniegodb')

    device = args.device or 'cuda'
    body_model = BodyModel(bm_fname=smplh_path('neutral', args.smplh_dir), num_betas=16).to(device)
    viewer = Viewer(loaders, body_model, port=args.port)
    viewer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Input npz.')
    parser.add_argument('--pred', type=str, nargs='*', default=[], help='SMPL-H npz files (label=path).')
    parser.add_argument('--video', type=str, default=None, help='Raw recording (VRS or video) for --input.')
    parser.add_argument('--omniegodb', type=str, default=None, help='OmniEgoDB root, for a dropdown of captures.')
    parser.add_argument('--track', type=str, default='aria', choices=list(TRACKS))
    parser.add_argument('--pred_dir', type=str, nargs='*', default=[], help='Prediction folders (label=folder).')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--smplh_dir', type=str, default=None, help='Folder of the SMPL-H model files.')
    args = parser.parse_args()

    main(args)
