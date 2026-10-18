"""
Usage
python scripts/preprocess_amass.py --amass_dir /path/to/AMASS_SMPLH                    (every paper set)
python scripts/preprocess_amass.py --amass_dir /path/to/AMASS_SMPLH --sets train diverse
python scripts/preprocess_amass.py --amass_dir /path/to/AMASS_SMPLH --camera 100,70,20 --camera 150,120
    [--out_dir data/preprocess] [--smplh_dir data/smplh] [--overwrite]

Paper sets: train (train.h5), diverse (test_diverse.h5, Tab. 1), fov90 / fov120 / fov180 / fov120x60 / tilt30
(Tab. 2). --camera H,V[,T] adds a test set seen through an H x V degree field of view tilted T degrees down,
written to test_fov{H}x{V}[_tilt{T}].h5.
"""

import argparse
import os

from omniegocap.training.preprocess import PAPER_SETS, SEED, PreprocessOmniEgoCap, camera_setting, setting_filename
from omniegocap.utils.paths import repo_path
from omniegocap.utils.transforms import set_seed


def parse_camera(text: str):
    values = [float(value) for value in text.split(',')]
    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError(f'--camera expects H,V or H,V,T in degrees, got "{text}"')
    return camera_setting(*values)


def main(args: argparse.Namespace):
    settings = [PAPER_SETS[name] for name in args.sets] + args.camera
    for setting in settings:
        output = os.path.join(args.out_dir, setting_filename(setting))
        if os.path.exists(output) and not args.overwrite:
            print(f'Skip {output} (exists; use --overwrite to rebuild)')
            continue
        print(f'Building {output}')
        set_seed(SEED)
        PreprocessOmniEgoCap(args.amass_dir, setting, out_dir=args.out_dir, smplh_dir=args.smplh_dir).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amass_dir', type=str, required=True, help='AMASS root (SMPL+H G).')
    parser.add_argument('--sets', type=str, nargs='*', choices=list(PAPER_SETS), default=None,
                        help='Paper sets to build. Default: all, unless --camera is given.')
    parser.add_argument('--camera', type=parse_camera, action='append', default=[],
                        help='Custom test set: horizontal,vertical[,downward tilt] field of view in degrees.')
    parser.add_argument('--out_dir', type=str, default=repo_path('data', 'preprocess'))
    parser.add_argument('--smplh_dir', type=str, default=None, help='Folder of the SMPL-H model files.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.sets is None:
        args.sets = [] if args.camera else list(PAPER_SETS)

    main(args)
