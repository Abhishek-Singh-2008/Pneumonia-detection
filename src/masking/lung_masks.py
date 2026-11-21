"""Stage 3.2: lung masks for every image, with a pretrained lung segmentation model (owned by Aman, issue #12).

The attention metric (Stage 3.3) and the masked training (Stage 4, Himanshu) both need to know which pixels
of each X-ray are lung. This file makes one mask per image and is also the place to LOAD masks from.

Segmentation model: torchxrayvision's PSPNet (xrv.baseline_models.chestx_det.PSPNet), trained on the
ChestX-Det dataset (adult chest X-rays from NIH ChestX-ray14) to segment 14 structures. We keep its
"Left Lung" and "Right Lung" outputs and join them into one lung mask. It was NOT trained on children, and our
images are pediatric, so the masks must be checked by eye (results/stage3/lung_masks_*.png, issue #12).

Mask format (#21 row 14, CLAUDE.md):
    {image_id}.png, 224 x 224, binary, 8-bit grayscale: 0 = not lung, 255 = lung
    (255 instead of 1 so the files can be opened and checked in any image viewer; load_mask() treats any
    value > 0 as lung, so 0/1 files would also work). One folder for all 5,856 images.

How each mask is made, and why it lines up with the Grad-CAM heatmaps:
    The classifiers see the WHOLE image squashed to 224 x 224 (plain resize, no crop, #21 row 5). So here the
    whole image is also plainly resized (to 512 x 512, the segmentation model's input size), segmented, and the
    mask is resized to 224 x 224 with NEAREST-neighbour interpolation (keeps it binary, #21 row 5).
    torchxrayvision's own examples centre-crop the image first; we deliberately do NOT, because a cropped
    mask would no longer line up pixel by pixel with the classifier's input.

Run from the repo root (GPU recommended; Kaggle with Internet ON, because the PSPNet weights are downloaded
on first use):
    python -m src.masking.lung_masks                      # all images; skips masks that already exist
    python -m src.masking.lung_masks --limit 40           # quick try on 40 images
    python -m src.masking.lung_masks --help

Writes:
    <out-dir>/{image_id}.png                     the masks (NOT committed; zip and upload to the team Drive)
    <out-dir>/areas.csv                          left and right lung area of every mask (used for the flags)
    <out-dir>.zip                                the same folder zipped for Drive, with --zip
    results/stage3/lung_mask_summary.csv         one row per image: total, left and right lung area, and a "flag"
                                                 for masks that look implausible (e.g. a lung missed because it is
                                                 white from pneumonia); flagged masks must be looked at by eye
    results/stage3/lung_masks_random.png         24 random masks drawn over their X-rays, for the eye check
    results/stage3/lung_masks_flagged.png        up to 24 flagged masks, for the eye check
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import IMAGE_SIZE, REPO_ROOT, RESULTS_DIR, SEED, get_data_root
from src.data.splits import load_split
from src.utils import set_seed

SEG_MODEL_NAME = "torchxrayvision PSPNet (chestx_det), Left Lung + Right Lung"
SEG_INPUT_SIZE = 512                      # input size the PSPNet was trained with
LUNG_TARGETS = ("Left Lung", "Right Lung")
OUT_DIR_NAME = "lung_masks"
STAGE3_DIR = RESULTS_DIR / "stage3"

AREAS_FILE = "areas.csv"                  # per-image left/right lung areas, written next to the masks

# A mask is flagged for the eye check if it looks implausible. Thresholds are fractions of the 224 x 224 image.
# Both lungs together normally cover roughly 25-45% of these X-rays and each lung roughly 13-23%.
# Known failure (seen on real test images): a lung that is white from pneumonia (consolidation) can be missed
# by the segmentation model, which was trained on adult X-rays. That shows up as one lung far smaller than the
# other, so a lung under 5% of the image, or under 40% of the other lung's area, is flagged.
FLAG_MIN_TOTAL, FLAG_MAX_TOTAL = 0.08, 0.60
FLAG_MIN_ONE_LUNG, FLAG_MIN_LUNG_RATIO = 0.05, 0.40


# ---------------------------------------------------------------------------
# Mask files: the format every stage uses
# ---------------------------------------------------------------------------
def save_mask(mask: np.ndarray, mask_dir, image_id: str) -> Path:
    """Write one binary mask as {image_id}.png (0 = background, 255 = lung).

    Args:
        mask: bool or 0/1 array of shape (224, 224).
        mask_dir: output folder (created if missing).
        image_id: the image's id from the split CSVs, e.g. "NORMAL_train_IM-0115-0001".

    Returns:
        Path of the written file.
    """
    mask = np.asarray(mask)
    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"mask must be {IMAGE_SIZE} x {IMAGE_SIZE}, got {mask.shape}")
    path = Path(mask_dir) / f"{image_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((mask > 0) * 255).astype(np.uint8), mode="L").save(path)
    return path


def load_mask(mask_dir, image_id: str) -> np.ndarray:
    """Read one lung mask. This is the function Stage 3.3 and Stage 4 use to get masks.

    Args:
        mask_dir: folder with the {image_id}.png files (unzipped from the team Drive).
        image_id: id from the split CSVs.

    Returns:
        bool numpy array (224, 224), True = lung.
    """
    with Image.open(Path(mask_dir) / f"{image_id}.png") as im:
        return np.asarray(im.convert("L")) > 0


def load_masks(mask_dir, image_ids) -> np.ndarray:
    """Read the masks of many images at once.

    Args:
        mask_dir: folder with the mask files.
        image_ids: iterable of image ids.

    Returns:
        bool numpy array (N, 224, 224), in the order of image_ids.
    """
    return np.stack([load_mask(mask_dir, i) for i in image_ids])


def default_out_dir() -> Path:
    """/kaggle/working/lung_masks on Kaggle (downloadable output), otherwise data/lung_masks (git-ignored)."""
    return Path("/kaggle/working") / OUT_DIR_NAME if os.path.isdir("/kaggle/working") else REPO_ROOT / "data" / OUT_DIR_NAME


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def load_segmenter(device):
    """Load the pretrained PSPNet lung segmentation model (downloads its weights the first time).

    Args:
        device: torch device.

    Returns:
        (model in eval mode on device, list of the channel indices of LUNG_TARGETS in its output)
    """
    import torchxrayvision as xrv   # imported here so the rest of the repo does not need this library
    model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    channels = [list(model.targets).index(t) for t in LUNG_TARGETS]
    return model, channels


def preprocess_for_segmenter(gray: Image.Image) -> np.ndarray:
    """Turn one grayscale X-ray into the PSPNet input.

    Args:
        gray: PIL image in mode "L" (any size).

    Returns:
        float32 numpy array (1, 512, 512) in the [-1024, 1024] range torchxrayvision models expect
        (same scaling as xrv.datasets.normalize(img, 255)).
    """
    # Plain resize of the WHOLE image (no crop), like the classifier's 224 x 224 resize; see the module docstring.
    arr = np.asarray(gray.resize((SEG_INPUT_SIZE, SEG_INPUT_SIZE), Image.BILINEAR), dtype=np.float32)
    arr = (2 * (arr / 255.0) - 1.0) * 1024    # 0..255 -> -1024..1024
    return arr[None]


@torch.no_grad()
def segment_lungs(model, channels, grays: list, device) -> tuple:
    """Segment the lungs of a batch of X-rays.

    Args:
        model, channels: from load_segmenter().
        grays: list of PIL "L" images.
        device: torch device of the model.

    Returns:
        (masks, left, right): three bool numpy arrays of shape (N, 224, 224): both lungs, left lung only,
        right lung only. ("Left"/"Right" are the patient's sides as named by the segmentation model.)
    """
    x = torch.from_numpy(np.stack([preprocess_for_segmenter(g) for g in grays])).to(device)  # (N, 1, 512, 512)
    probs = torch.sigmoid(model(x))[:, channels]                     # (N, 2, h, w): left and right lung
    # Back to the classifier's 224 x 224 grid. Thresholding at 0.5 first and then NEAREST resizing keeps the
    # mask strictly binary, as agreed in #21 row 5.
    binary = (probs > 0.5).float()
    binary = torch.nn.functional.interpolate(binary, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest") > 0.5
    left, right = binary[:, 0].cpu().numpy(), binary[:, 1].cpu().numpy()
    return left | right, left, right


# ---------------------------------------------------------------------------
# Eye-check figures
# ---------------------------------------------------------------------------
def save_overlay_grid(rows: pd.DataFrame, data_root: Path, mask_dir: Path, path: Path, title: str) -> None:
    """Draw masks over their X-rays in a grid (6 columns) for the eye check.

    Args:
        rows: DataFrame with columns image_id, filepath, lung_area (one panel per row, at most 24 used).
        data_root: dataset folder (contains train/, val/, test/).
        mask_dir: folder with the masks.
        path: PNG file to write.
        title: figure title.
    """
    import matplotlib
    matplotlib.use("Agg")                     # files only, no screen (Kaggle, servers)
    import matplotlib.pyplot as plt

    rows = rows.head(24)
    if rows.empty:
        return
    ncols = 6
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        with Image.open(data_root / r["filepath"]) as im:
            # Same plain 224 x 224 resize the classifiers see, so this is exactly what the metric uses.
            img = np.asarray(im.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        mask = load_mask(mask_dir, r["image_id"])
        ax.imshow(img, cmap="gray")
        ax.imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.35)   # lung in orange
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_title(f"{r['image_id']}\nlung area {r['lung_area']:.2f}", fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=default_out_dir(), help="folder for the {image_id}.png masks")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (for a quick try)")
    p.add_argument("--overwrite", action="store_true", help="re-make masks that already exist")
    p.add_argument("--zip", action="store_true", help="also write <out-dir>.zip for the team Drive")
    return p.parse_args()


def main() -> None:
    """Make (or reuse) a mask for every image in the three split CSVs, then write the summary and figures."""
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = get_data_root()
    # All images of the dataset: the masks are needed for train and val too (Stage 4 trains on masked images).
    images = pd.concat([load_split(s) for s in ("train", "val", "test")], ignore_index=True)
    if args.limit:
        images = images.head(args.limit)
    print(f"Device: {device} | images: {len(images)} | masks -> {args.out_dir}")

    todo = images if args.overwrite else images[[not (args.out_dir / f"{i}.png").exists() for i in images["image_id"]]]
    print(f"{len(todo)} masks to make ({len(images) - len(todo)} already exist)")
    if len(todo):
        model, channels = load_segmenter(device)
        # Process in batches; each batch is decoded from disk, segmented on the GPU and saved.
        for start in range(0, len(todo), args.batch_size):
            batch = todo.iloc[start:start + args.batch_size]
            grays = []
            for rel in batch["filepath"]:
                with Image.open(data_root / rel) as im:
                    grays.append(im.convert("L"))   # grayscale first, as in the classifier pipeline (#22)
            masks, left, right = segment_lungs(model, channels, grays, device)
            for image_id, mask in zip(batch["image_id"], masks):
                save_mask(mask, args.out_dir, image_id)
            # Only the joined mask is saved as PNG, so the separate left/right areas (needed to spot a missed
            # lung) are appended to areas.csv in the mask folder. Appending per batch keeps them even if the
            # run is interrupted; a re-run keeps the newest row of each image.
            areas_path = args.out_dir / AREAS_FILE
            pd.DataFrame({"image_id": list(batch["image_id"]), "left_lung_area": left.mean(axis=(1, 2)),
                          "right_lung_area": right.mean(axis=(1, 2))}).round(4).to_csv(
                areas_path, mode="a", header=not areas_path.exists(), index=False, lineterminator="\n")
            if (start // args.batch_size) % 50 == 0:
                print(f"  {start + len(batch)}/{len(todo)}")

    # Summary over ALL images (also the ones made in an earlier run), from the saved files.
    rows = []
    for r in images.itertuples(index=False):
        rows.append({"image_id": r.image_id, "split": r.split, "label": r.label, "filepath": r.filepath,
                     "lung_area": load_mask(args.out_dir, r.image_id).mean()})
    summary = pd.DataFrame(rows)
    areas = pd.read_csv(args.out_dir / AREAS_FILE).drop_duplicates("image_id", keep="last")
    summary = summary.merge(areas, on="image_id", how="left")
    small = summary[["left_lung_area", "right_lung_area"]].min(axis=1)
    large = summary[["left_lung_area", "right_lung_area"]].max(axis=1)
    summary["flag"] = ((summary["lung_area"] < FLAG_MIN_TOTAL) | (summary["lung_area"] > FLAG_MAX_TOTAL)
                       | (small < FLAG_MIN_ONE_LUNG) | (small < FLAG_MIN_LUNG_RATIO * large))
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "split", "label", "lung_area", "left_lung_area", "right_lung_area", "flag"]
    summary[cols].round(4).to_csv(STAGE3_DIR / "lung_mask_summary.csv", index=False, lineterminator="\n")

    # Figures for the eye check: a seeded random sample, and the flagged masks (most extreme areas first).
    rng = np.random.RandomState(SEED)
    sample = summary.iloc[rng.permutation(len(summary))[:24]]
    save_overlay_grid(sample, data_root, args.out_dir, STAGE3_DIR / "lung_masks_random.png",
                      "Lung masks: 24 random images (seed 42)")
    flagged = summary[summary["flag"]].sort_values("lung_area")
    save_overlay_grid(flagged, data_root, args.out_dir, STAGE3_DIR / "lung_masks_flagged.png",
                      f"Flagged lung masks ({len(flagged)} of {len(summary)}), smallest area first")

    print(summary.groupby(["split", "label"])["lung_area"].describe().round(3).to_string())
    print(f"Flagged for the eye check: {int(summary['flag'].sum())} of {len(summary)}")
    if args.zip:
        archive = shutil.make_archive(str(args.out_dir), "zip", root_dir=args.out_dir)
        print("Zipped for the team Drive:", archive)


if __name__ == "__main__":
    main()
