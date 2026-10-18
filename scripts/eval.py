"""
Usage
AMASS (Tab. 1 and 2), evaluation h5 made by scripts/preprocess_amass.py:
python scripts/eval.py amass --h5 data/preprocess/test_diverse.h5 [--checkpoint checkpoints/omniegocap.pt]
    [--no_optimization] [--guidance_scale 30] [--seed 42] [--save_pred preds/]
python scripts/eval.py amass --h5 data/preprocess/test_diverse.h5 --pred_dir other_method/   (score saved SMPL-H npz)
python scripts/eval.py amass --h5 data/preprocess/test_diverse.h5 --export_inputs inputs/    (inputs for other methods)

OmniEgoDB (Tab. 3), tracks aria / aria_rgb / gopro / quest:
python scripts/eval.py omniegodb --root /path/to/OmniEgoDB --track aria [--checkpoint ...] [--save_pred preds/aria/]
python scripts/eval.py omniegodb --root /path/to/OmniEgoDB --track aria --pred_dir other_method/aria/
    [--per_capture]
"""

import argparse
import json

from human_body_prior.body_model.body_model import BodyModel

from omniegocap.benchmarks import amass, omniegodb
from omniegocap.core.inference import GUIDANCE_SCALE, SEED, OmniEgoCap
from omniegocap.utils.paths import repo_path, smplh_path


def run_amass(args: argparse.Namespace):
    if args.export_inputs:
        count = amass.export_inputs(args.h5, args.export_inputs)
        print(f'Wrote {count} input files to {args.export_inputs}')
        return

    model, body_model = load_models(args)

    result = amass.evaluate(
        args.h5,
        model=model,
        body_model=body_model,
        pred_dir=args.pred_dir,
        save_dir=args.save_pred,
        optimization=not args.no_optimization,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    print(f'Setting: {json.dumps(result["setting"]) if result["setting"] else "(not recorded in the h5)"}')
    print(f'Sequences: {result["num_sequences"]}')
    for name, unit in amass.METRICS.items():
        print(f'{name}({unit}): {result[name]:.3f}' if name == 'Jerk' else f'{name}({unit}): {result[name]:.2f}')
    print(f'Runtime(s): {result["Runtime"]:.2f}')


def load_models(args: argparse.Namespace):
    if args.pred_dir:
        body_model = BodyModel(bm_fname=smplh_path('neutral', args.smplh_dir), num_betas=16)
        if args.device != 'cpu':
            body_model = body_model.to(args.device or 'cuda')
        return None, body_model

    return OmniEgoCap(args.checkpoint, device=args.device, smplh_dir=args.smplh_dir), None


def run_omniegodb(args: argparse.Namespace):
    model, body_model = load_models(args)
    result = omniegodb.evaluate(
        args.root,
        args.track,
        model=model,
        body_model=body_model,
        pred_dir=args.pred_dir,
        save_dir=args.save_pred,
        optimization=not args.no_optimization,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    if args.per_capture:
        for name, metrics in result['captures'].items():
            print(f'{name}: ' + ', '.join(f'{key} {metrics[key]:.2f}' for key in omniegodb.METRICS))
    frames = sum(metrics['num_frames'] for metrics in result['captures'].values())
    print(f'Track: {args.track}, captures: {len(result["captures"])}, frames: {frames}')
    for name, unit in omniegodb.METRICS.items():
        value = result['mean'][name]
        print(f'{name}({unit}): {value:.3f}' if name == 'Jerk' else f'{name}({unit}): {value:.2f}')
    print(f'Runtime(s): {result["mean"]["Runtime"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='benchmark', required=True)

    parser_amass = subparsers.add_parser('amass', help='AMASS evaluation h5')
    parser_amass.add_argument('--h5', type=str, required=True, help='Evaluation h5 from scripts/preprocess_amass.py.')
    parser_amass.add_argument('--checkpoint', type=str, default=repo_path('checkpoints', 'omniegocap.pt'))
    parser_amass.add_argument('--pred_dir', type=str, default=None, help='Score saved SMPL-H npz instead.')
    parser_amass.add_argument('--save_pred', type=str, default=None, help='Folder to save the predictions to.')
    parser_amass.add_argument('--export_inputs', type=str, default=None, help='Only write the input npz files.')
    parser_amass.add_argument('--no_optimization', action='store_true', help='Skip the guidance optimization.')
    parser_amass.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    parser_amass.add_argument('--seed', type=int, default=SEED)
    parser_amass.add_argument('--device', type=str, default=None)
    parser_amass.add_argument('--smplh_dir', type=str, default=None, help='Folder of the SMPL-H model files.')
    parser_amass.set_defaults(func=run_amass)

    parser_omni = subparsers.add_parser('omniegodb', help='OmniEgoDB captures')
    parser_omni.add_argument('--root', type=str, required=True, help='OmniEgoDB root folder.')
    parser_omni.add_argument('--track', type=str, required=True, choices=list(omniegodb.TRACKS))
    parser_omni.add_argument('--checkpoint', type=str, default=repo_path('checkpoints', 'omniegocap.pt'))
    parser_omni.add_argument('--pred_dir', type=str, default=None, help='Score saved SMPL-H npz (<capture>.npz).')
    parser_omni.add_argument('--save_pred', type=str, default=None, help='Folder to save the predictions to.')
    parser_omni.add_argument('--per_capture', action='store_true', help='Print every capture.')
    parser_omni.add_argument('--no_optimization', action='store_true', help='Skip the guidance optimization.')
    parser_omni.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    parser_omni.add_argument('--seed', type=int, default=SEED)
    parser_omni.add_argument('--device', type=str, default=None)
    parser_omni.add_argument('--smplh_dir', type=str, default=None, help='Folder of the SMPL-H model files.')
    parser_omni.set_defaults(func=run_omniegodb)

    args = parser.parse_args()
    args.func(args)
