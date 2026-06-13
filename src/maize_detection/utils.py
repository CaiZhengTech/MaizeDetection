"""Small shared helpers: deterministic seeding and device selection.

Kept intentionally tiny — these are reused by train/evaluate/predict so the seed
and device logic lives in exactly one place.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU-only machines


def get_device() -> torch.device:
    """GPU when available (Colab/Kaggle training); CPU otherwise (local)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
