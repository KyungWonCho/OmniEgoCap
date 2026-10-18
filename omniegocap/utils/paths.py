"""
Files that live outside the package: the repository's data/ folder and the SMPL-H model files.

SMPL-H is not redistributed (download it from https://mano.is.tue.mpg.de, "Extended SMPL+H model").
It is looked up in this order:
    1. an explicit directory (e.g. a --smplh_dir argument)
    2. the environment variable OMNIEGOCAP_SMPLH_DIR
    3. <repo>/data/smplh
"""

import os
from typing import Optional

from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMPLH_ENV = 'OMNIEGOCAP_SMPLH_DIR'
SMPLH_FILES = {
    'neutral': 'SMPLH_NEUTRAL.npz',
    'male': 'SMPLH_MALE.npz',
    'female': 'SMPLH_FEMALE.npz',
}


def repo_path(*parts: str) -> str:
    """
    Absolute path inside the repository, independent of the working directory.

    Args:
        parts (str):
            Path components relative to the repository root.

    Returns:
        path (str):
            Absolute path.
    """

    return os.path.join(REPO_ROOT, *parts)


def smplh_path(
    gender: str = 'neutral',
    directory: Optional[str] = None,
) -> str:
    """
    Path to an SMPL-H model file.

    Args:
        gender (str):
            'neutral', 'male' or 'female'.
        directory (str or None):
            Directory holding the SMPL-H npz files. If None, use $OMNIEGOCAP_SMPLH_DIR, then <repo>/data/smplh.

    Returns:
        path (str):
            Path to the npz file.
    """

    if gender not in SMPLH_FILES:
        raise ValueError(f'Unknown gender: {gender}')

    if directory is None:
        directory = os.environ.get(SMPLH_ENV, repo_path('data', 'smplh'))
    path = os.path.join(directory, SMPLH_FILES[gender])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'SMPL-H model not found: {path}\n'
            f'Download the "Extended SMPL+H model" from https://mano.is.tue.mpg.de and put {SMPLH_FILES[gender]} in '
            f'<repo>/data/smplh, or point ${SMPLH_ENV} (or --smplh_dir) to its directory.'
        )

    return path


def register_resolvers() -> None:
    """
    Register the OmegaConf resolvers used by conf/: ${repo:data/...} and ${smplh:neutral}.
    Call once before Hydra composes the configuration.
    """

    if not OmegaConf.has_resolver('repo'):
        OmegaConf.register_new_resolver('repo', lambda path: repo_path(path))
    if not OmegaConf.has_resolver('smplh'):
        OmegaConf.register_new_resolver('smplh', lambda gender: smplh_path(gender))
