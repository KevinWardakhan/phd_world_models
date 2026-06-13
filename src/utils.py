"""Small helpers shared across stages: config loading, seeding, IO."""

import json
import os

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_rng(seed):
    return np.random.default_rng(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_eval_noise(path, n, dim, seed, device):
    """Fixed evaluation noise x_T, shared across milestones for fair comparison.

    Saved once on CPU and reloaded so M2/M3/M4 all start from the same x_T,
    independent of the run device.
    """
    if os.path.exists(path):
        noise = torch.load(path, map_location="cpu")
    else:
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(n, dim, generator=gen)
        ensure_dir(os.path.dirname(path))
        torch.save(noise.cpu(), path)
    return noise.to(device)
