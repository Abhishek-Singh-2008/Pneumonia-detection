"""Stage 1.3: dataset, preprocessing, augmentation and data loaders.

Decisions (issue #21, rows 5 and 6; issue #22):
- Every image is converted to grayscale first, then copied to 3 channels. All RGB
  files in the dataset are PNEUMONIA, so this removes colour mode as a shortcut.
- Plain resize to 224x224 (no pad, no crop), then ImageNet mean/std.
- Training images only: random rotation of up to +/-10 degrees. No horizontal flip.
- Each item is (image, label, image_id). label is an int (0 = NORMAL, 1 = PNEUMONIA);
  use labels.float() for BCEWithLogitsLoss.

Usage:
    from src.data.dataset import get_dataloaders
    loaders = get_dataloaders(batch_size=32)
    for images, labels, image_ids in loaders["train"]: ...

Stage 4 (masked training) can pass data_root pointing at the masked copies, which
keep the same relative paths as the original images.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, ROTATION_DEGREES, SEED, get_data_root
from src.data.splits import load_split
from src.utils import seed_worker


def load_image(path) -> Image.Image:
    """Open an X-ray as grayscale copied to 3 channels (RGB mode)."""
    with Image.open(path) as im:
        return im.convert("L").convert("RGB")


def build_transforms(train: bool) -> transforms.Compose:
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps.append(transforms.RandomRotation(ROTATION_DEGREES))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation (for plotting). Works on (C,H,W) or (N,C,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)


class ChestXrayDataset(Dataset):
    """Images listed in one split CSV (columns from src/data/splits.py)."""

    def __init__(self, split_df: pd.DataFrame, data_root=None, transform=None):
        self.df = split_df.reset_index(drop=True)
        self.data_root = Path(data_root) if data_root is not None else get_data_root()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = load_image(self.data_root / row["filepath"])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row["image_id"]


def get_datasets(data_root=None, augment_train: bool = True) -> dict:
    """train / val / test datasets. Only train is augmented."""
    data_root = Path(data_root) if data_root is not None else get_data_root()
    return {
        name: ChestXrayDataset(load_split(name), data_root, build_transforms(train=(name == "train" and augment_train)))
        for name in ("train", "val", "test")
    }


def get_dataloaders(batch_size: int = 32, num_workers: int = 2, data_root=None,
                    augment_train: bool = True, seed: int = SEED) -> dict:
    """train / val / test loaders. Only train is shuffled; shuffling and workers are seeded."""
    datasets = get_datasets(data_root, augment_train)
    loaders = {}
    for name, ds in datasets.items():
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders
