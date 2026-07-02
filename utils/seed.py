"""Centralized random seed helpers for PertDiffBench experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def resolve_seed(cli_seed: Optional[int] = None, default: int = 0) -> int:
    """Pick run seed: RUN_SEED env (0-based run index) overrides CLI default."""
    env_val = os.environ.get("RUN_SEED")
    if env_val is not None and env_val != "":
        return int(env_val)
    if cli_seed is not None:
        return int(cli_seed)
    return default


def set_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
