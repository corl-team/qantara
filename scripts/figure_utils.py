"""Helpers for the Qantara rollout figure: eval-data loading, observation encoding,
latent decoding, and checkpoint loading. Imported by make_rollout_figure.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402

from module import Qantara  # noqa: E402  (needed so torch.load can unpickle a checkpoint)
from jepa import JEPA  # noqa: E402


ENV_NAMES = {
    "pusht":   "pusht_expert_train",
    "tworoom": "tworoom",
    "reacher": "reacher",
    "cube":    "ogbench/cube_single_expert",
}

# ImageNet stats — match train.py decoder denormalisation.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_ckpt(path, device):
    """Load a pickled JEPA object checkpoint in eval mode with gradients off."""
    obj: JEPA = torch.load(path, map_location=device, weights_only=False)
    return obj.to(device).eval().requires_grad_(False)


def build_horizon_loader(env, batch_size, num_workers, cache_dir, num_steps):
    """Val-slice DataLoader over an env's HDF5, with a configurable num_steps for
    multi-step rollouts."""
    import h5py
    from utils import get_column_normalizer, get_img_preprocessor

    stablewm = cache_dir or os.environ.get("STABLEWM_HOME")
    assert stablewm, "set STABLEWM_HOME to the datasets/checkpoints directory"
    name = ENV_NAMES[env]
    with h5py.File(os.path.join(stablewm, f"{name}.h5"), "r") as f:
        all_cols = set(f.keys())
    cache_cols = [c for c in ("action", "proprio", "state") if c in all_cols]
    load_cols = [c for c in ("pixels", "action", "proprio", "state") if c in all_cols]
    dataset = swm.data.HDF5Dataset(num_steps=num_steps, frameskip=5, name=name,
                                   keys_to_load=load_cols, keys_to_cache=cache_cols,
                                   cache_dir=cache_dir, transform=None)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=224)]
    for col in cache_cols:
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    rnd = torch.Generator().manual_seed(11)
    _, val_set = spt.data.random_split(dataset, lengths=[0.98, 0.02], generator=rnd)
    return torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, drop_last=False)


def encode_batch(model: JEPA, info: dict, is_qantara: bool):
    """Returns (z_clean, a_for_predictor). Qantara feeds the raw action to the
    predictor; LeWM feeds the action-encoder output."""
    info = dict(info)
    action = info.get("action")
    if action is not None:
        action = torch.nan_to_num(action, 0.0)
        info["action"] = action
    info = model.encode(info)
    if is_qantara:
        return info["emb"], action
    return info["emb"], info.get("act_emb")


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) ImageNet-normalised -> [0,1] clamped."""
    mean = _IMAGENET_MEAN.to(x.device)
    std = _IMAGENET_STD.to(x.device)
    return (x * std + mean).clamp(0, 1)


@torch.no_grad()
def decode_z(model, z_flat: torch.Tensor) -> torch.Tensor:
    """(B, D_emb) -> (B, 3, H, W) in [0,1]."""
    return denorm_to_01(model.decoder(z_flat))
