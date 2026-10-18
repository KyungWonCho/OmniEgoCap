"""
Checkpoint = model weights + the configuration needed to rebuild the model and run inference.

    {
        'state_dict': {...},                                    # model weights
        'config': {'model': ..., 'representation': ..., 'diffusion': ...},
        'meta': {...},                                          # e.g. training step, EMA rate
    }
"""

from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from omniegocap.core.alignment import get_x_dim
from omniegocap.core.model import OmniEgoCapModel

CONFIG_KEYS = ('model', 'representation', 'diffusion')


def save_checkpoint(
    path: str,
    state_dict: Dict[str, torch.Tensor],
    cfg: DictConfig,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save weights together with the model, representation and diffusion configuration.

    Args:
        path (str):
            Output path.
        state_dict (Dict[str, torch.Tensor]):
            Model weights.
        cfg (DictConfig):
            Full configuration; only the sections in CONFIG_KEYS are stored.
        meta (Dict[str, Any] or None):
            Extra information (training step, EMA rate, ...).
    """

    config = {key: OmegaConf.to_container(cfg[key], resolve=True) for key in CONFIG_KEYS}
    torch.save({'state_dict': state_dict, 'config': config, 'meta': meta or {}}, path)


def load_checkpoint(
    path: str,
    map_location: str = 'cpu',
) -> Tuple[Dict[str, torch.Tensor], DictConfig, Dict[str, Any]]:
    """
    Load weights and configuration.

    Args:
        path (str):
            Checkpoint path.
        map_location (str):
            Device to load the tensors to.

    Returns:
        state_dict (Dict[str, torch.Tensor]):
            Model weights.
        config (DictConfig):
            Sections in CONFIG_KEYS.
        meta (Dict[str, Any]):
            Extra information stored with the weights.
    """

    data = torch.load(path, map_location=map_location, weights_only=True)
    if 'state_dict' not in data:
        raise ValueError(f'{path} is not an OmniEgoCap checkpoint (no state_dict and config)')

    return data['state_dict'], OmegaConf.create(data['config']), data['meta']


def build_model(
    path: str,
    device: str = 'cpu',
) -> Tuple[OmniEgoCapModel, DictConfig, Dict[str, Any]]:
    """
    Rebuild the model from a checkpoint, in eval mode.

    Args:
        path (str):
            Checkpoint path.
        device (str):
            Device to put the model on.

    Returns:
        model (OmniEgoCapModel):
            Model with the weights loaded.
        config (DictConfig):
            Model, representation and diffusion configuration.
        meta (Dict[str, Any]):
            Extra information stored with the weights.
    """

    state_dict, config, meta = load_checkpoint(path)
    model = OmniEgoCapModel(config.model, get_x_dim(config.representation))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config, meta
