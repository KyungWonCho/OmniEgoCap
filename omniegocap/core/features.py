from typing import Dict

import numpy as np

from omniegocap.utils.transforms import inv_T_numpy, make_canonical


def make_features(
    head: np.ndarray,
    wrist: np.ndarray,
    visible: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Make the conditioning features from head and wrist poses in world coordinates, as described in Sec 3.2
    (Preprocessing and Representation) in the paper. Frame 0 only serves as the previous frame of frame 1, so every
    output starts at frame 1.

    Args:
        head (L, 4, 4):
            Head transformation in world coordinates.
        wrist (L, 2, 4, 4):
            Left and right wrist transformation in world coordinates.
        visible (L, 2):
            Visibility of left and right wrists.

    Returns:
        Dictionary containing the following items:
            head (L - 1, 4, 4):
                head pose in world coordinate
            head_delta (L - 1, 4, 4):
                head pose delta from the previous frame
            canonical (L - 1, 4, 4):
                canonical frame in world coordinate
            yaw_delta (L - 1,):
                canonical frame's yaw angle change (radian)
            wrist (L - 1, 2, 4, 4):
                left/right wrist pose in head coordinate
            visible (L - 1, 2):
                left/right wrist visibility
    """

    head_delta = inv_T_numpy(head[:-1]) @ head[1:]                      # (L - 1, 4, 4)
    canonical, yaw_delta = make_canonical(head)                         # (L - 1, 4, 4), (L - 1,)
    wrist_head = inv_T_numpy(head[1:, None]) @ wrist[1:]                # (L - 1, 2, 4, 4)

    return {
        'head': head[1:].astype(np.float32),
        'head_delta': head_delta.astype(np.float32),
        'canonical': canonical.astype(np.float32),
        'yaw_delta': yaw_delta.astype(np.float32),
        'wrist': wrist_head.astype(np.float32),
        'visible': visible[1:].astype(np.bool_),
    }
