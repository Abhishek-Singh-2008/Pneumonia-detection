"""Stage 1.3: dataset, preprocessing, augmentation and data loaders (owned by Abhishek).

Every model in the project (custom CNN, DenseNet-121, masked models) gets its images
through this file, so all stages see exactly the same preprocessing.

Decisions (issue #21, rows 5 and 6; issue #22):
- Each image is converted to grayscale first, then copied to 3 channels. All 283 RGB
  files in the dataset are PNEUMONIA, so without this step the colour mode itself
  would be a shortcut (#22). Three channels are needed by ImageNet-pretrained networks.
- Plain resize to 224x224 (no padding, no cropping): padding would add black borders
  that can become a shortcut, and cropping could cut off the edges of the lungs.
- ImageNet mean/std normalisation (needed by the pretrained DenseNet-121 in Stage 2).
- Training images only: random rotation of up to +/-10 degrees. No horizontal flip:
  flipping a chest X-ray moves the heart to the other side and mirrors the L/R markers,
  which are exactly the shortcuts this project studies.
- Each item is (image, label, image_id):
      image     float tensor of shape (3, 224, 224), normalised
      label     int, 0 = NORMAL, 1 = PNEUMONIA (use labels.float() for BCEWithLogitsLoss)
      image_id  str, e.g. "NORMAL_train_IM-0115-0001", to match predictions with masks later

Usage:
    from src.data.dataset import get_dataloaders
    loaders = get_dataloaders(batch_size=32, cache=True)
    for images, labels, image_ids in loaders["train"]: ...

Stage 4 (masked training) can pass data_root= pointing at the masked copies of the
images, which keep the same relative file paths as the originals.

cache=True decodes and resizes every image once and keeps the 224x224 grayscale result
in memory (about 300 MB for the whole dataset). The output is identical to cache=False
(resizing an image that is already 224x224 is a no-op in PIL; checked by
src.train_baseline.check_cache_matches). It only avoids decoding the large JPEGs
(up to ~2900 px wide) again in every epoch, which made an epoch take minutes.
"""
from concurrent.futures import ThreadPoolExecutor
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
    """Open one X-ray file as grayscale copied to 3 channels.

    Args:
        path: full path to the image file.

    Returns:
        PIL image in RGB mode whose three channels are identical (grayscale copied 3 times).
        convert("L") first makes RGB and grayscale files look the same, removing the
        colour-mode shortcut (#22).
    """
    # "with" closes the file handle right away; convert() returns a new in-memory image.
    with Image.open(path) as im:
        return im.convert("L").convert("RGB")


def build_transforms(train: bool) -> transforms.Compose:
    """Return the preprocessing pipeline for one split.

    Args:
        train: True for the training split (adds random rotation); False for val/test
            (deterministic, so evaluation always sees the same pixels).

    Returns:
        A torchvision Compose that turns a PIL image into a normalised (3, 224, 224) tensor.
    """
    # 1. Plain resize to 224x224. The aspect ratio is NOT kept (decision #21 row 5);
    #    the class-dependent distortion this causes is documented as a confound in #22.
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        # 2. Train only: rotate by a random angle in [-10, +10] degrees (empty corners are filled with black).
        steps.append(transforms.RandomRotation(ROTATION_DEGREES))
    # 3. PIL image (0..255) -> float tensor (0..1), shape (3, H, W).
    # 4. Subtract ImageNet mean and divide by std, per channel.
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation so an image can be plotted.

    Args:
        images: tensor of shape (3, H, W) or (N, 3, H, W) as produced by build_transforms().

    Returns:
        Tensor of the same shape with values clipped to [0, 1].
    """
    # view(-1, 1, 1) reshapes the 3 channel values so they broadcast over H and W
    # (and over N for a batch).
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for the images listed in one split CSV.

    Args:
        split_df: DataFrame from src.data.splits.load_split() (needs columns filepath, label, image_id).
        data_root: folder that contains train/, val/, test/; defaults to src.config.get_data_root().
            Stage 4 points this at the masked copies of the images.
        transform: pipeline from build_transforms(); None returns the PIL image.
        cache: if True, decode + resize all images once in __init__ and keep them in memory
            (identical output, much faster epochs; see the module docstring).
    """

    def __init__(self, split_df: pd.DataFrame, data_root=None, transform=None, cache: bool = False):
        # reset_index so that position i in the dataset is row i of the DataFrame.
        self.df = split_df.reset_index(drop=True)
        self.data_root = Path(data_root) if data_root is not None else get_data_root()
        self.transform = transform
        self._cache = self._preload() if cache else None

    def _load_resized_gray(self, rel_path) -> Image.Image:
        """Decode one image, convert to grayscale and resize to 224x224 (what the cache stores).
        Grayscale ("L") is stored instead of RGB to use a third of the memory; it is copied to
        3 channels in __getitem__, which gives exactly the same pixels as resizing the RGB copy."""
        with Image.open(self.data_root / rel_path) as im:
            return transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(im.convert("L"))

    def _preload(self) -> list:
        """Load every image of the split into memory, using 8 threads (JPEG decoding releases the GIL).
        pool.map keeps the original order, so cache[i] belongs to row i."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(self._load_resized_gray, self.df["filepath"]))

    def __len__(self) -> int:
        """Number of images in the split."""
        return len(self.df)

    def __getitem__(self, i):
        """Return (image tensor (3, 224, 224), label int, image_id str) for row i."""
        row = self.df.iloc[i]
        if self._cache is not None:
            # Cached image is already 224x224 grayscale; copy it to 3 channels.
            # The Resize inside self.transform is then a no-op.
            image = self._cache[i].convert("RGB")
        else:
            image = load_image(self.data_root / row["filepath"])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row["image_id"]


def get_datasets(data_root=None, augment_train: bool = True, cache: bool = False) -> dict:
    """Create the train, val and test datasets from the committed split CSVs.

    Args:
        data_root: image folder (default: found automatically).
        augment_train: if False, the train split also gets the deterministic val/test pipeline
            (used for plots and checks).
        cache: keep decoded images in memory (see ChestXrayDataset).

    Returns:
        {"train": Dataset, "val": Dataset, "test": Dataset}. Only train is augmented.
    """
    data_root = Path(data_root) if data_root is not None else get_data_root()
    return {
        name: ChestXrayDataset(load_split(name), data_root,
                               build_transforms(train=(name == "train" and augment_train)), cache=cache)
        for name in ("train", "val", "test")
    }


def get_dataloaders(batch_size: int = 32, num_workers: int = 2, data_root=None,
                    augment_train: bool = True, seed: int = SEED, cache: bool = False) -> dict:
    """Create seeded DataLoaders for train, val and test.

    Args:
        batch_size: images per batch.
        num_workers: background processes that prepare batches (2 suits Kaggle; use 0 on Windows if workers fail).
        data_root, augment_train, cache: passed to get_datasets().
        seed: seed for the shuffle order (default 42).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}.
        Each batch is (images (B, 3, 224, 224), labels (B,) int64, image_ids list of str).

    Notes:
        Only train is shuffled; val and test keep the CSV order, so predictions line up with the CSV.
        Each loader gets its own torch.Generator seeded with `seed`, so the shuffle order and the
        worker seeds are the same in every run. With num_workers=0 the augmentation uses the global
        torch RNG instead, so call src.utils.set_seed() first.
    """
    datasets = get_datasets(data_root, augment_train, cache)
    loaders = {}
    for name, ds in datasets.items():
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            worker_init_fn=seed_worker,               # seeds NumPy/random inside each worker
            generator=generator,                      # fixes shuffle order and worker torch seeds
            pin_memory=torch.cuda.is_available(),     # faster copies to the GPU
            persistent_workers=num_workers > 0,       # keep workers alive between epochs (faster)
        )
    return loaders
