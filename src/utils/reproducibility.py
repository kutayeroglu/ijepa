"""Helpers for reproducible PyTorch runs (seeding and deterministic backends)."""

import random

import numpy as np
import torch


def set_deterministic_backend():
    """cuDNN / matmul behavior only — does not touch Python or PyTorch RNG."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_rng(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
