"""
Viewer (viser) for one sequence at a time: the input poses, any number of predictions, and the ground truth and the
raw video when they exist. Every layer is drawn on the input's frames, and a layer appears only when its data is given.
Run it through scripts/visualize.py.
"""

import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as vtf
from human_body_prior.body_model.body_model import BodyModel

from omniegocap.benchmarks.omniegodb import METRICS, GroundTruth, score
from omniegocap.io.sequence import EgoSequence
from omniegocap.io.smplh import SMPLHSequence
from omniegocap.utils.paths import repo_path
from omniegocap.vis.media import open_frames

PRED_COLORS = [(125, 151, 212), (232, 145, 89), (106, 176, 120), (196, 109, 167), (200, 180, 90), (120, 120, 120)]
GT_BONE_COLOR = (255, 120, 80)
GT_JOINT_COLOR = (255, 220, 80)
WRIST_COLORS = ((230, 162, 154), (154, 217, 194))
MAX_POINTS = 500_000
CHUNK = 256

# device mesh drawn at the device pose: file in data/, rotation in the device frame (meshes are in millimeters)
DEVICE_MESHES = {
    'aria': ('glasses.stl', np.eye(3)),
    'gopro': ('gopro.stl', np.diag([-1.0, -1.0, 1.0])),
}


class Scene:
    def __init__(
        self,
        name: str,
        seq: EgoSequence,
        preds: Dict[str, SMPLHSequence],
        gt: Optional[GroundTruth] = None,
        video_path: Optional[str] = None,
    ):
        """
        Data of one sequence.

        Args:
            name (str):
                Display name.
            seq (EgoSequence):
                Model input.
            preds (Dict[str, SMPLHSequence]):
                Predictions by method name, on the input frames.
            gt (GroundTruth or None):
                Ground truth on the input frames.
            video_path (str or None):
                Raw recording (VRS or video file) for the image panel.
        """

        for label, pred in preds.items():
            if len(pred) != len(seq):
                raise ValueError(f'{label}: {len(pred)} frames, input has {len(seq)}')
        if gt is not None and len(gt) != len(seq):
            raise ValueError(f'ground truth has {len(gt)} frames, input has {len(seq)}')

        self.name = name
        self.seq = seq
        self.preds = preds
        self.gt = gt
        self.video_path = video_path


def load_scene(
    name: str,
    input_path: str,
    pred_paths: Dict[str, str],
    gt_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Scene:
    """
    Read the files of one sequence; missing prediction files are skipped.

    Args:
        name (str):
            Display name.
        input_path (str):
            Input npz.
        pred_paths (Dict[str, str]):
            SMPL-H npz by method name.
        gt_path (str or None):
            OmniEgoDB gt.npz.
        video_path (str or None):
            Raw recording.

    Returns:
        scene (Scene):
            Loaded sequence.
    """

    preds = {label: SMPLHSequence.load(path) for label, path in pred_paths.items() if os.path.exists(path)}
    gt = GroundTruth.load(gt_path) if gt_path is not None and os.path.exists(gt_path) else None

    return Scene(name, EgoSequence.load(input_path), preds, gt=gt, video_path=video_path)


class MeshCache:
    def __init__(
        self,
        pred: SMPLHSequence,
        body_model: BodyModel,
    ):
        """
        SMPL-H vertices computed CHUNK frames at a time on demand, so long captures do not fill the memory.

        Args:
            pred (SMPLHSequence):
                Prediction.
            body_model (BodyModel):
                Neutral SMPL-H body model.
        """

        self.pred = pred
        self.body_model = body_model
        self.chunks = {}


    def vertices(self, frame: int) -> np.ndarray:
        chunk = frame // CHUNK
        if chunk not in self.chunks:
            if len(self.chunks) >= 4:
                self.chunks.pop(next(iter(self.chunks)))
            s, e = chunk * CHUNK, min((chunk + 1) * CHUNK, len(self.pred))
            device = self.body_model.shapedirs.device
            with torch.no_grad():
                body = self.body_model(
                    root_orient=torch.from_numpy(self.pred.root_orient[s:e]).to(device),
                    pose_body=torch.from_numpy(self.pred.pose_body[s:e]).to(device),
                    trans=torch.from_numpy(self.pred.trans[s:e]).to(device),
                    betas=torch.from_numpy(self.pred.betas).to(device)[None],
                )
            self.chunks[chunk] = body.v.cpu().numpy()                                  # (CHUNK, V, 3)

        return self.chunks[chunk][frame % CHUNK]


class Viewer:
    def __init__(
        self,
        loaders: Dict[str, Callable[[], Scene]],
        body_model: BodyModel,
        port: int = 8080,
    ):
        """
        Viser viewer; with more than one sequence, a dropdown switches between them.

        Args:
            loaders (Dict[str, Callable[[], Scene]]):
                Scene loader by name.
            body_model (BodyModel):
                Neutral SMPL-H body model.
            port (int):
                Port of the viser server.
        """

        self.loaders = loaders
        self.body_model = body_model
        self.faces = body_model.f.cpu().numpy()
        self.server = viser.ViserServer(port=port)
        self.server.scene.add_grid('ground', width=300.0, height=300.0)
        self.nodes, self.gui_nodes = [], []
        self.scene, self.frame, self.playing = None, 0, False

        # gui
        if len(loaders) > 1:
            self.scene_gui = self.server.gui.add_dropdown(
                'Sequence',
                options=list(loaders),
                initial_value=list(loaders)[0],
            )
            self.scene_gui.on_update(lambda _: self.load(self.scene_gui.value))
        self.frame_gui = self.server.gui.add_slider('frame', 0, 1, step=1, initial_value=0)
        self.play_gui = self.server.gui.add_button('Play')
        self.fps_gui = self.server.gui.add_slider('fps', min=1, max=120, step=1, initial_value=30)
        self.wrist_gui = self.server.gui.add_checkbox('Wrist frames', initial_value=True)
        self.points_gui = self.server.gui.add_checkbox('Point cloud', initial_value=True)
        self.gt_gui = self.server.gui.add_checkbox('Ground truth', initial_value=True)
        self.video_gui = self.server.gui.add_checkbox('Video', initial_value=True)
        self.pred_folder = self.server.gui.add_folder('Predictions')

        @self.play_gui.on_click
        def _(_) -> None:
            self.playing = not self.playing
            self.play_gui.label = 'Pause' if self.playing else 'Play'

        @self.frame_gui.on_update
        def _(_) -> None:
            if not self.playing:
                self.frame = int(self.frame_gui.value)

        @self.points_gui.on_update
        def _(_) -> None:
            if self.points_node is not None:
                self.points_node.visible = self.points_gui.value

        self.load(list(loaders)[0])


    def clear(self) -> None:
        for node in self.nodes + self.gui_nodes:
            node.remove()
        self.nodes, self.gui_nodes = [], []
        self.points_node, self.device_node, self.head_node = None, None, None
        self.pred_nodes, self.gt_nodes, self.image_node, self.frames = {}, None, None, None


    def load(self, name: str) -> None:
        """
        Show one sequence.

        Args:
            name (str):
                Key of the loader.
        """

        self.clear()
        scene = self.loaders[name]()
        seq = scene.seq
        self.scene, self.frame = scene, 0
        self.frame_gui.max = len(seq) - 1
        self.frame_gui.value = 0

        # point cloud
        if seq.points is not None and len(seq.points):
            points = seq.points
            if len(points) > MAX_POINTS:
                points = points[np.random.default_rng(0).choice(len(points), MAX_POINTS, replace=False)]
            self.points_node = self.server.scene.add_point_cloud(
                'points',
                points,
                point_size=0.005,
                colors=(0.8, 0.8, 0.8),
                point_shape='circle',
                visible=self.points_gui.value,
            )
            self.nodes.append(self.points_node)

        # head: device mesh at the device pose, otherwise axes at the head pose
        mesh = DEVICE_MESHES.get(seq.device)
        if mesh is not None and seq.device_pose is not None:
            device_mesh = trimesh.load_mesh(repo_path('data', mesh[0]))
            device_mesh.apply_scale(0.001)
            device_mesh.vertices = device_mesh.vertices @ mesh[1].T
            device_mesh.visual.vertex_colors = [10, 20, 20, 255]
            self.device_node = self.server.scene.add_mesh_trimesh('device', device_mesh)
            self.nodes.append(self.device_node)
        else:
            self.head_node = self.server.scene.add_frame('head', axes_length=0.1, axes_radius=0.005)
            self.nodes.append(self.head_node)

        # wrists
        self.wrist_nodes = [
            self.server.scene.add_frame(
                f'{side}_wrist',
                axes_length=0.08,
                axes_radius=0.005,
                origin_radius=0.02,
                origin_color=color,
                visible=False,
            )
            for side, color in zip(('left', 'right'), WRIST_COLORS)
        ]
        self.nodes += self.wrist_nodes

        # predictions, one color and checkbox each
        for i, (label, pred) in enumerate(scene.preds.items()):
            cache = MeshCache(pred, self.body_model)
            node = self.server.scene.add_mesh_simple(
                f'pred/{label}',
                vertices=cache.vertices(0),
                faces=self.faces,
                color=PRED_COLORS[i % len(PRED_COLORS)],
            )
            with self.pred_folder:
                checkbox = self.server.gui.add_checkbox(label, initial_value=True)
            self.pred_nodes[label] = (node, checkbox, cache)
            self.nodes.append(node)
            self.gui_nodes.append(checkbox)

        # ground truth skeleton and per-method metrics
        if scene.gt is not None:
            bones = np.array(scene.gt.bones())
            positions = scene.gt.segments[:, :, :3, 3]                                  # (L, 23, 3)
            bone_node = self.server.scene.add_line_segments(
                'gt/bones',
                points=positions[0][bones],
                colors=np.broadcast_to(np.array(GT_BONE_COLOR, dtype=np.uint8), (len(bones), 2, 3)).copy(),
                line_width=4.0,
            )
            joint_node = self.server.scene.add_point_cloud(
                'gt/joints',
                points=positions[0],
                colors=np.tile(np.array([GT_JOINT_COLOR], dtype=np.uint8), (positions.shape[1], 1)),
                point_size=0.025,
            )
            self.gt_nodes = (bone_node, joint_node, positions, bones)
            self.nodes += [bone_node, joint_node]

            if scene.preds:
                rows = ['| method | ' + ' | '.join(f'{name} ({unit})' for name, unit in METRICS.items()) + ' |',
                        '|---' * (len(METRICS) + 1) + '|']
                for label, pred in scene.preds.items():
                    metrics = score(pred, scene.gt, seq.visible, self.body_model)
                    rows.append(f'| {label} | ' + ' | '.join(f'{metrics[name]:.2f}' for name in METRICS) + ' |')
                self.gui_nodes.append(self.server.gui.add_markdown('\n'.join(rows)))

        # raw video
        if scene.video_path is not None and seq.raw_time_s is not None:
            self.frames = open_frames(scene.video_path)
            if self.frames is not None:
                image = self.read_image(0)
                if image is not None:
                    self.image_node = self.server.gui.add_image(image, label=os.path.basename(scene.video_path))
                    self.gui_nodes.append(self.image_node)

        self.update(0)


    def read_image(self, frame: int) -> Optional[np.ndarray]:
        return self.frames.read(float(self.scene.seq.raw_time_s[frame]))


    def update(self, frame: int) -> None:
        """
        Move every layer to one frame.

        Args:
            frame (int):
                Frame index of the input.
        """

        seq = self.scene.seq
        with self.server.atomic():
            if self.device_node is not None:
                self.device_node.wxyz = vtf.SO3.from_matrix(seq.device_pose[frame, :3, :3]).wxyz
                self.device_node.position = seq.device_pose[frame, :3, 3]
            if self.head_node is not None:
                self.head_node.wxyz = vtf.SO3.from_matrix(seq.head[frame, :3, :3]).wxyz
                self.head_node.position = seq.head[frame, :3, 3]

            for side, node in enumerate(self.wrist_nodes):
                node.wxyz = vtf.SO3.from_matrix(seq.wrist[frame, side, :3, :3]).wxyz
                node.position = seq.wrist[frame, side, :3, 3]
                node.visible = bool(self.wrist_gui.value and seq.visible[frame, side])

            for label, (node, checkbox, cache) in self.pred_nodes.items():
                visible = bool(checkbox.value and self.scene.preds[label].frame_valid[frame])
                node.visible = visible
                if visible:
                    node.vertices = cache.vertices(frame)

            if self.gt_nodes is not None:
                bone_node, joint_node, positions, bones = self.gt_nodes
                visible = bool(self.gt_gui.value and self.scene.gt.valid[frame])
                bone_node.visible = joint_node.visible = visible
                if visible:
                    bone_node.points = positions[frame][bones]
                    joint_node.points = positions[frame]

            self.frame_gui.value = frame

        if self.image_node is not None and self.video_gui.value:
            image = self.read_image(frame)
            if image is not None:
                self.image_node.image = image


    def run(self) -> None:
        while True:
            self.update(self.frame)
            if self.playing:
                self.frame = (self.frame + 1) % len(self.scene.seq)
            time.sleep(1.0 / max(1, int(self.fps_gui.value)))


def capture_loaders(
    captures: List[tuple],
    input_name: str,
    pred_dirs: Dict[str, str],
) -> Dict[str, Callable[[], Scene]]:
    """
    Scene loaders for OmniEgoDB capture folders.

    Args:
        captures (List[tuple]):
            (name, folder) pairs.
        input_name (str):
            input.npz or input_rgb.npz.
        pred_dirs (Dict[str, str]):
            Prediction folder (<capture>.npz inside) by method name.

    Returns:
        loaders (Dict[str, Callable[[], Scene]]):
            Loader by capture name.
    """

    def loader(name: str, folder: str) -> Callable[[], Scene]:
        video = [os.path.join(folder, f) for f in ('recording.vrs', 'video.MP4', 'video.mp4')
                 if os.path.exists(os.path.join(folder, f))]

        return lambda: load_scene(
            name,
            os.path.join(folder, input_name),
            {label: os.path.join(path, f'{name}.npz') for label, path in pred_dirs.items()},
            gt_path=os.path.join(folder, 'gt.npz'),
            video_path=video[0] if video else None,
        )

    return {name: loader(name, folder) for name, folder in captures}
