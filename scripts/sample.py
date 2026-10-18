"""
Usage
python scripts/sample.py --input seq_input.npz --output seq_smplh.npz
python scripts/sample.py --input inputs/ --output outputs/      (every *.npz in the folder)
    [--checkpoint checkpoints/omniegocap.pt] [--no_optimization] [--guidance_scale 30] [--seed 42]
    [--device cuda] [--smplh_dir data/smplh]
"""

import argparse
import glob
import os

from tqdm import tqdm

from omniegocap.core.inference import GUIDANCE_SCALE, SEED, OmniEgoCap
from omniegocap.io.sequence import EgoSequence
from omniegocap.utils.paths import repo_path


def main(args: argparse.Namespace):
    if os.path.isdir(args.input):
        inputs = sorted(glob.glob(os.path.join(args.input, '*.npz')))
        os.makedirs(args.output, exist_ok=True)
        outputs = [os.path.join(args.output, os.path.basename(path)) for path in inputs]
    else:
        inputs, outputs = [args.input], [args.output]
        if os.path.dirname(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not inputs:
        raise ValueError(f'No input npz found in {args.input}')

    model = OmniEgoCap(args.checkpoint, device=args.device, smplh_dir=args.smplh_dir)
    print(f'Using device: {model.device}')

    for input_path, output_path in tqdm(list(zip(inputs, outputs))):
        seq = EgoSequence.load(input_path)
        result = model.predict(
            seq,
            optimization=not args.no_optimization,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        result.meta['input'] = os.path.basename(input_path)
        result.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input npz file, or a folder of them.')
    parser.add_argument('--output', type=str, required=True, help='Output SMPL-H npz file, or a folder.')
    parser.add_argument('--checkpoint', type=str, default=repo_path('checkpoints', 'omniegocap.pt'))
    parser.add_argument('--no_optimization', action='store_true', help='Skip the guidance optimization.')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--smplh_dir', type=str, default=None, help='Folder of the SMPL-H model files.')
    args = parser.parse_args()

    main(args)
