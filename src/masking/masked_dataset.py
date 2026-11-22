"""Stage 4.1: create lung-masked copies of the committed dataset splits (issue #16).

Apply Stage 3's binary 224x224 lung masks to grayscale, resized X-rays, setting
pixels outside each mask to black. The original images and committed split CSVs
are never modified. Masked copies mirror the split folder layout and use PNG
files so masking does not add JPEG compression artifacts.

Run from the repository root:
    python -m src.masking.masked_dataset --mask-dir /path/to/lung_masks

Outputs:
    /kaggle/working/masked_chest_xray/ by default on Kaggle, otherwise
    masked_chest_xray/ in the repository root (git-ignored).
    The output contains masked PNG images and masked_dataset_summary.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from src.config import IMAGE_SIZE, REPO_ROOT, RESULTS_DIR, SEED, SPLITS_DIR, get_data_root
from src.data.dataset import ChestXrayDataset, build_transforms
from src.data.splits import load_split
from src.masking.lung_masks import load_mask
from src.utils import seed_worker


def default_output_root() -> Path:
    """Return a writable location for generated masked images."""
    kaggle_working = Path("/kaggle/working")
    return kaggle_working / "masked_chest_xray" if kaggle_working.is_dir() else REPO_ROOT / "masked_chest_xray"


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Resize a source image and zero pixels outside its boolean lung mask.

    Args:
        image: source X-ray as a PIL image, in any supported mode or size.
        mask: boolean array with shape (IMAGE_SIZE, IMAGE_SIZE).

    Returns:
        Grayscale PIL image of size (IMAGE_SIZE, IMAGE_SIZE), with outside pixels black.

    Raises:
        ValueError: if the mask shape differs from the configured image dimensions.
    """
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape is {mask.shape}; expected {expected_shape}")
    resized = ImageOps.grayscale(image).resize(
        expected_shape, resample=Image.Resampling.BILINEAR
    )
    pixels = np.asarray(resized, dtype=np.uint8)
    return Image.fromarray(np.where(mask, pixels, 0).astype(np.uint8))


def _is_nested(path: Path, parent: Path) -> bool:
    """Return whether path is equal to or nested under parent, independent of OS."""
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def _read_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the already-committed train, validation and test CSVs."""
    return {name: load_split(name, splits_dir=splits_dir) for name in ("train", "val", "test")}


def create_masked_copies(
    data_root: Path,
    mask_dir: Path,
    output_root: Path,
    splits_dir: Path = SPLITS_DIR,
    overwrite: bool = False,
) -> dict:
    """Create lossless masked image copies without changing the source data or splits.

    Args:
        data_root: directory containing the official train/val/test image folders.
        mask_dir: directory containing Stage 3 `{image_id}.png` masks.
        output_root: destination for images and the summary JSON.
        splits_dir: directory containing the committed split CSVs.
        overwrite: allow replacing generated files already present in output_root.

    Returns:
        Summary metadata, also written to `masked_dataset_summary.json`.

    Raises:
        FileNotFoundError: for missing source images or masks.
        FileExistsError: if output_root contains files and overwrite is False.
        ValueError: for unsafe source/output nesting, duplicate IDs, or invalid masks.
    """
    data_root = Path(data_root).resolve()
    mask_dir = Path(mask_dir).resolve()
    output_root = Path(output_root).resolve()
    if _is_nested(output_root, data_root) or _is_nested(data_root, output_root):
        raise ValueError("The masked output folder must not be the same as or nested with the source dataset folder.")

    split_frames = _read_splits(Path(splits_dir))
    all_rows = pd.concat(split_frames.values(), ignore_index=True)
    if all_rows["image_id"].duplicated().any():
        duplicates = all_rows.loc[all_rows["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Committed split CSVs contain duplicate image IDs, e.g. {duplicates}")

    if output_root.exists():
        existing_files = [path for path in output_root.rglob("*") if path.is_file()]
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output folder already contains files: {output_root}. "
                "Choose a new folder or pass --overwrite."
            )

    # Validate all inputs before writing any images, so a missing file cannot leave a
    # seemingly complete partial output.
    image_paths = {}
    for row in all_rows.itertuples(index=False):
        source = data_root / row.filepath
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found for {row.image_id}: {source}")
        mask_path = mask_dir / f"{row.image_id}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Lung mask not found for {row.image_id}: {mask_path}")
        image_paths[row.image_id] = (source, Path(row.filepath).with_suffix(".png"))

    output_root.mkdir(parents=True, exist_ok=True)
    for row in all_rows.itertuples(index=False):
        source, relative_output = image_paths[row.image_id]
        mask = load_mask(mask_dir, row.image_id)
        with Image.open(source) as source_image:
            masked = apply_mask(source_image, mask)
        destination = output_root / relative_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        masked.save(destination, format="PNG")

    summary = {
        "image_size": IMAGE_SIZE,
        "source_data_root": str(data_root),
        "mask_dir": str(mask_dir),
        "output_root": str(output_root),
        "mask_policy": "grayscale and resize image, then set pixels outside nonzero mask to black",
        "split_counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "total_images": int(len(all_rows)),
        "output_format": "PNG",
        "splits_dir": str(Path(splits_dir).resolve()),
    }

    # Stage 3 supplies per-image mask areas and a flag for masks that may miss
    # consolidated lung regions. The Drive mask folder only has areas; the flag lives
    # in the committed results/stage3/lung_mask_summary.csv. Record counts from
    # whichever is available; never silently drop those cases.
    for source in (mask_dir / "areas.csv", RESULTS_DIR / "stage3" / "lung_mask_summary.csv"):
        if not source.is_file():
            continue
        areas = pd.read_csv(source)
        if not {"image_id", "flag"}.issubset(areas.columns):
            continue
        flagged = areas["flag"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "flagged"}
        )
        flagged_ids = set(areas.loc[flagged, "image_id"].astype(str))
        summary["flagged_masks_by_split"] = {
            name: int(frame["image_id"].isin(flagged_ids).sum())
            for name, frame in split_frames.items()
        }
        summary["flagged_masks_total"] = int(len(flagged_ids))
        summary["flagged_masks_source"] = str(source)
        summary["flagged_mask_policy"] = "included; Stage 3 flagged-mask caveat applies"
        break

    summary_path = output_root / "masked_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def get_masked_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 2,
    splits_dir: Path = SPLITS_DIR,
    seed: int = SEED,
    cache: bool = False,
) -> dict:
    """Build standard project DataLoaders from generated masked PNG copies.

    The committed split membership, labels and image IDs remain unchanged; only
    each source JPEG filepath is mapped to its generated PNG counterpart.

    Args:
        data_root: output folder created by `create_masked_copies`.
        batch_size: number of images per batch.
        num_workers: DataLoader worker count.
        splits_dir: location of the committed split CSVs.
        seed: DataLoader generator seed.
        cache: preload resized images in memory for faster epochs.

    Returns:
        `{"train": ..., "val": ..., "test": ...}` with the same item format as
        `src.data.dataset.get_dataloaders`.

    Raises:
        FileNotFoundError: if any masked image referenced by a committed split is missing.
    """
    data_root = Path(data_root)
    loaders = {}
    for name in ("train", "val", "test"):
        frame = load_split(name, splits_dir=splits_dir).copy()
        frame["filepath"] = frame["filepath"].map(lambda value: str(Path(value).with_suffix(".png")))
        missing = [relative for relative in frame["filepath"] if not (data_root / relative).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} masked {name} images under {data_root}; "
                f"first missing path: {missing[0]}"
            )
        dataset = ChestXrayDataset(
            frame,
            data_root=data_root,
            transform=build_transforms(train=(name == "train")),
            cache=cache,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders


def parse_args():
    """Parse paths and overwrite behavior for the masked-copy generation command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None, help="source folder containing train/, val/, and test/")
    parser.add_argument("--mask-dir", type=Path, required=True, help="folder containing Stage 3 image_id.png masks")
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace generated images already in output-root")
    return parser.parse_args()


def main() -> None:
    """Generate masked image copies and print their split counts."""
    args = parse_args()
    data_root = args.data_root or get_data_root()
    summary = create_masked_copies(data_root, args.mask_dir, args.output_root, args.splits_dir, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
