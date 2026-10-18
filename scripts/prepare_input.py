"""
Usage
python scripts/prepare_input.py aria_mps --vrs recording.vrs --mps_dir mps/ --output seq_input.npz
python scripts/prepare_input.py trajectory --trajectory traj.txt --wrist wrist.npz --cam_to_head cam_to_head.txt \
    --output seq_input.npz
python scripts/prepare_input.py --plugin my_package.my_device my_device ... --output seq_input.npz

Writes the model input (omniegocap.io.sequence.EgoSequence) of one recording. See omniegocap/adapters/ for the
arguments of each adapter and omniegocap/adapters/base.py for adding one.
"""

import argparse
import importlib

import omniegocap.adapters.aria_mps  # noqa: F401  (registers aria_mps)
import omniegocap.adapters.trajectory  # noqa: F401  (registers trajectory)
from omniegocap.adapters.base import ADAPTERS


def main():
    # plugins register their adapters on import, before the sub-commands are built
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--plugin', type=str, action='append', default=[])
    for module in pre_parser.parse_known_args()[0].plugin:
        importlib.import_module(module)

    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', type=str, action='append', default=[], help='Module that registers an adapter.')
    subparsers = parser.add_subparsers(dest='adapter', required=True)
    for name, adapter in ADAPTERS.items():
        subparser = subparsers.add_parser(name)
        adapter.add_arguments(subparser)
        subparser.add_argument('--output', type=str, required=True, help='Output input npz.')
    args = parser.parse_args()

    seq = ADAPTERS[args.adapter]().load(args)
    seq.save(args.output)
    visible = seq.visible.mean(axis=0).round(3).tolist()
    print(f'{args.output}: {len(seq)} frames, wrists visible {visible} (left, right)')


if __name__ == '__main__':
    main()
