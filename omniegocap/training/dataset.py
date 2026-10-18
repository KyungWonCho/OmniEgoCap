import random
from typing import Any, Dict, List

import h5py
import torch
import torch.utils.data as data

GT_KEYS = ('root_transl', 'root_orient', 'joint_pos', 'contact', 'joint_orient', 'betas')


class OmniEgoCapDataset(data.Dataset):
    def __init__(
        self,
        file: str,
        max_length: int,
    ):
        """
        Training dataset (train.h5 from scripts/preprocess_amass.py). Each item is a random crop of one sequence.

        Args:
            file (str):
                Path to the h5py file containing the dataset.
            max_length (int):
                Length of the random crop.
        """

        super(OmniEgoCapDataset, self).__init__()
        self.data = h5py.File(file, 'r')
        self.keys = list(self.data.keys())
        self.max_length = max_length
        assert all(key in self.data[self.keys[0]] for key in GT_KEYS), \
            f'{file} has no ground truth annotations ({", ".join(GT_KEYS)}).'


    def __len__(self) -> int:
        return len(self.keys)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary containing the following items:
                head (length, 4, 4):
                    head pose in world coordinate
                head_delta (length, 4, 4):
                    head pose delta
                canonical (length, 4, 4):
                    canonical frame in world coordinate
                yaw_delta (length,):
                    canonical frame's yaw angle change (radian)
                wrist (length, 2, 4, 4):
                    left/right wrist pose in head coordinate
                length (int):
                    sequence length
                root (length, 4, 4):
                    root transformation in world coordinate
                joint_orient (length, 21, 3, 3):
                    joint orientations in local coordinate
                betas (16,):
                    body shape parameters
                joint_pos (length, 22, 3):
                    joint positions in world coordinate
                contact (length, 2):
                    left/right foot contact label
        """

        key = self.keys[index]
        length = self.data[key]['head'].shape[0]

        st = random.randint(0, max(0, length - self.max_length))
        ed = min(st + self.max_length, length)
        length = ed - st

        head = torch.from_numpy(self.data[key]['head'][st:ed])
        head_delta = torch.from_numpy(self.data[key]['head_delta'][st:ed])
        canonical = torch.from_numpy(self.data[key]['canonical'][st:ed])
        yaw_delta = torch.from_numpy(self.data[key]['yaw_delta'][st:ed])
        wrist = torch.from_numpy(self.data[key]['wrist'][st:ed])

        root = torch.zeros((length, 4, 4))
        root[:, :3, 3] = torch.from_numpy(self.data[key]['root_transl'][st:ed])
        root[:, :3, :3] = torch.from_numpy(self.data[key]['root_orient'][st:ed])
        root[:, 3, 3] = 1.0
        joint_pos = torch.from_numpy(self.data[key]['joint_pos'][st:ed])
        contact = torch.from_numpy(self.data[key]['contact'][st:ed])
        joint_orient = torch.from_numpy(self.data[key]['joint_orient'][st:ed])
        betas = torch.from_numpy(self.data[key]['betas'][()])

        return {
            'head': head.float(),
            'head_delta': head_delta.float(),
            'canonical': canonical.float(),
            'yaw_delta': yaw_delta.float(),
            'wrist': wrist.float(),
            'length': length,
            'root': root.float(),
            'joint_orient': joint_orient.float(),
            'betas': betas.float(),
            'joint_pos': joint_pos.float(),
            'contact': contact.bool(),
        }


def collate_omniegocap(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for OmniEgoCapDataset. Pad the sequences in the batch to the maximum length in the batch.

    Args:
        batch: List of dictionaries returned by OmniEgoCapDataset.__getitem__.

    Returns:
        Dictionary containing the following items:
            head (batch, max_length, 4, 4):
                head pose in world coordinate
            head_delta (batch, max_length, 4, 4):
                head pose delta
            canonical (batch, max_length, 4, 4):
                canonical frame in world coordinate
            yaw_delta (batch, max_length):
                canonical frame's yaw angle change (radian)
            wrist (batch, max_length, 2, 4, 4):
                left/right wrist pose in head coordinate
            length (batch,):
                sequence length
            root (batch, max_length, 4, 4):
                root transformation in world coordinate
            joint_orient (batch, max_length, 21, 3, 3):
                joint orientations in local coordinate
            betas (batch, 16):
                body shape parameters
            joint_pos (batch, max_length, 22, 3):
                joint positions in world coordinate
            contact (batch, max_length, 2):
                left/right foot contact label
    """

    B = len(batch)
    length = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    L = int(length.max().item())

    head = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, L, 1, 1)
    head_delta = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, L, 1, 1)
    canonical = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, L, 1, 1)
    yaw_delta = torch.zeros((B, L), dtype=torch.float32)
    wrist = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(B, L, 2, 1, 1)
    root = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, L, 1, 1)
    joint_orient = torch.eye(3, dtype=torch.float32).view(1, 1, 1, 3, 3).repeat(B, L, 21, 1, 1)
    betas = torch.zeros((B, 16), dtype=torch.float32)
    joint_pos = torch.zeros((B, L, 22, 3), dtype=torch.float32)
    contact = torch.zeros((B, L, 2), dtype=torch.bool)

    for i, item in enumerate(batch):
        l = item['length']
        head[i, :l] = item['head']
        head_delta[i, :l] = item['head_delta']
        canonical[i, :l] = item['canonical']
        yaw_delta[i, :l] = item['yaw_delta']
        wrist[i, :l] = item['wrist']
        root[i, :l] = item['root']
        joint_orient[i, :l] = item['joint_orient']
        betas[i] = item['betas']
        joint_pos[i, :l] = item['joint_pos']
        contact[i, :l] = item['contact']

    return {
        'head': head,
        'head_delta': head_delta,
        'canonical': canonical,
        'yaw_delta': yaw_delta,
        'wrist': wrist,
        'length': length,
        'root': root,
        'joint_orient': joint_orient,
        'betas': betas,
        'joint_pos': joint_pos,
        'contact': contact,
    }
