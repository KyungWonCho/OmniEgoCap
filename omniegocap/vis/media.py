"""
Frames of the raw recording for the viewer's image panel: Aria VRS (RGB camera) or a video file (GoPro MP4).
The readers import their libraries only when a recording is opened, so the viewer runs without them.
"""

from typing import Optional

import numpy as np

MAX_WIDTH = 640


def resize(image: np.ndarray, max_width: int = MAX_WIDTH) -> np.ndarray:
    """
    Shrink an image to at most max_width pixels wide.

    Args:
        image (H, W, 3):
            Image.
        max_width (int):
            Maximum width.

    Returns:
        image (H', W', 3):
            Resized image.
    """

    import cv2

    h, w = image.shape[:2]
    if w <= max_width:
        return image

    return cv2.resize(image, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)


class VRSFrames:
    def __init__(self, path: str):
        """
        RGB frames of an Aria recording, looked up by device time (EgoSequence.raw_time_s).

        Args:
            path (str):
                Path to the VRS file.
        """

        from projectaria_tools.core.data_provider import create_vrs_data_provider
        from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

        self.provider = create_vrs_data_provider(path)
        self.stream = self.provider.get_stream_id_from_label('camera-rgb')
        self.query = (TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)


    def read(self, time_s: float) -> Optional[np.ndarray]:
        image, _ = self.provider.get_image_data_by_time_ns(self.stream, int(round(time_s * 1e9)), *self.query)
        if not image.is_valid():
            return None

        # Aria stores the RGB image sideways; turn it upright
        return resize(np.ascontiguousarray(np.rot90(image.to_numpy_array(), k=3)))


class VideoFrames:
    def __init__(self, path: str):
        """
        Frames of a video file, looked up by time from its first frame (EgoSequence.raw_time_s).

        Args:
            path (str):
                Path to the video file.
        """

        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.position = -1


    def read(self, time_s: float) -> Optional[np.ndarray]:
        index = int(round(time_s * self.fps))
        # decode forward while playing, seek only on jumps
        if index != self.position + 1:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        self.position = index
        if not ok:
            return None

        return resize(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))


def open_frames(path: str):
    """
    Frame reader for a VRS or video file, or None if it cannot be opened (missing library or file).

    Args:
        path (str):
            Path to the recording.

    Returns:
        reader (VRSFrames or VideoFrames or None):
            Reader with read(time_s).
    """

    try:
        if path.lower().endswith('.vrs'):
            return VRSFrames(path)
        return VideoFrames(path)
    except Exception as e:
        print(f'[video] not shown: {path} ({e})')
        return None
