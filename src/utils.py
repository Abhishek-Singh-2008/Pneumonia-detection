"""Reproducibility helpers shared by all stages (owned by Stage 1, Abhishek).

Deep-learning results change from run to run unless every source of randomness is
fixed: Python's `random`, NumPy, PyTorch on CPU and GPU, and the DataLoader worker
processes that apply random augmentation. These two helpers fix all of them with the
project seed (src.config.SEED = 42, #21 row 9).

Usage:
    from src.utils import set_seed
    set_seed()                      # first line of every training / evaluation script
    DataLoader(..., worker_init_fn=seed_worker, generator=g)   # done for you in src/data/dataset.py
"""
import os
import random

import numpy as np

from src.config import SEED


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    Args:
        seed: the random seed; defaults to the project seed 42.
        deterministic: if True, ask cuDNN to use deterministic algorithms and turn off its
            auto-tuner. Training becomes slightly slower but repeated runs give the same result.

    Side effects:
        Changes the global random state of `random`, NumPy and PyTorch. Call it again right
        before anything that must be repeatable (e.g. creating a model, so its initial weights
        are always the same).
    """
    random.seed(seed)
    np.random.seed(seed)
    # Makes Python's hash() of strings repeatable (affects e.g. the order of some set operations).
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch is imported here, not at the top, so scripts that never use torch
    # (e.g. making the splits) still work without it installed.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)            # CPU generator (also used by augmentation when num_workers=0)
    torch.cuda.manual_seed_all(seed)   # every GPU generator; harmless when there is no GPU
    if deterministic:
        torch.backends.cudnn.deterministic = True   # same convolution algorithm every run
        torch.backends.cudnn.benchmark = False      # no auto-tuning, which picks algorithms by timing


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and `random` inside each DataLoader worker process.

    PyTorch already gives each worker its own torch seed (derived from the DataLoader's
    `generator`), but NumPy and `random` would otherwise start from the same state in
    every worker, or from a different state in every run. Passing this function as
    DataLoader(worker_init_fn=seed_worker) derives their seeds from the worker's torch seed,
    so augmentation is repeatable and different across workers.

    Args:
        worker_id: index of the worker, passed in by PyTorch (unused; the torch seed already differs per worker).
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32   # NumPy seeds must fit in 32 bits
    np.random.seed(worker_seed)
    random.seed(worker_seed)
