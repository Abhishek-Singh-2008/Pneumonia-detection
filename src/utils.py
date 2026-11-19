"""Reproducibility helpers shared by all stages."""
import os
import random

import numpy as np

from src.config import SEED


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker) so each worker is seeded."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
