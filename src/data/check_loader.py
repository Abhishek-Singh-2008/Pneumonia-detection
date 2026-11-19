"""Stage 1.3 check: prove the data loader works on the real dataset (owned by Abhishek).

What it checks:
1. Every image of train, val and test loads once without errors, with shape (3, 224, 224)
   and dtype float32, and the image counts match the split CSVs.
2. Grayscale -> 3 channels really gives three identical channels (before normalisation).
3. Reproducibility: after set_seed(), two fresh train loaders give the same first batch
   (same images in the same order, same random rotations).
4. A picture of four training images without and with augmentation, to check by eye that
   the rotation is small and nothing is flipped.

Run from the repo root (takes a few minutes on Kaggle because it decodes every JPEG):
    python -m src.data.check_loader
Writes results/stage1_loader/loader_check.json and augmentation_samples.png.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")   # draw to files only; no screen needed (Kaggle, servers)
import matplotlib.pyplot as plt
import torch

from src.config import IMAGE_SIZE, RESULTS_DIR, SEED
from src.data.dataset import denormalize, get_dataloaders, get_datasets
from src.utils import set_seed


def main(batch_size: int = 64, num_workers: int = 2) -> None:
    """Run all checks and save the report and the augmentation picture.

    Args:
        batch_size: images per batch while scanning the dataset.
        num_workers: DataLoader worker processes.

    Raises:
        AssertionError: if any check fails (wrong shape/dtype, missing images, unequal channels).
    """
    set_seed(SEED)
    out = RESULTS_DIR / "stage1_loader"
    out.mkdir(parents=True, exist_ok=True)

    # ---- Check 1: load every image of every split once -------------------------------
    loaders = get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    report = {}
    for name, loader in loaders.items():
        start = time.time()
        n, n_pos, ids = 0, 0, set()          # images seen, pneumonia images seen, unique image_ids
        channel_sum = torch.zeros(3)         # running sums to compute the mean/std per channel
        channel_sq = torch.zeros(3)
        for images, labels, image_ids in loader:
            assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE), images.shape
            assert images.dtype == torch.float32
            n += images.shape[0]
            n_pos += int(labels.sum())
            ids.update(image_ids)
            # Mean over H and W gives one value per image and channel; summing over the batch
            # accumulates the per-image means (all images have the same size, so this is exact).
            channel_sum += images.mean(dim=(2, 3)).sum(0)
            channel_sq += (images ** 2).mean(dim=(2, 3)).sum(0)
        expected = len(loader.dataset)
        # Every row of the CSV must have been loaded exactly once.
        assert n == expected == len(ids), (name, n, expected, len(ids))
        mean = channel_sum / n
        report[name] = {
            "images": n,
            "pneumonia": n_pos,
            "batches": len(loader),
            "seconds": round(time.time() - start, 1),
            # After ImageNet normalisation these should be roughly mean 0, std 1 (a sanity check, not exact).
            "channel_mean": [round(float(x), 3) for x in mean],
            "channel_std": [round(float(x), 3) for x in (channel_sq / n - mean ** 2).sqrt()],
        }
        print(name, report[name], flush=True)

    # ---- Check 2: grayscale -> 3 channels gives three identical channels --------------
    # After normalisation the channels differ (each has its own mean/std), so undo it first.
    ds = get_datasets(augment_train=False)["val"]
    raw = denormalize(ds[0][0])
    report["channels_equal_before_norm"] = bool(torch.allclose(raw[0], raw[1], atol=1e-6) and torch.allclose(raw[1], raw[2], atol=1e-6))
    assert report["channels_equal_before_norm"]

    # ---- Check 3: reproducibility -------------------------------------------------------
    # After set_seed, a fresh train loader must give the same first batch (order and rotations).
    # With num_workers=0 the augmentation uses the global torch RNG, which is why set_seed
    # is called right before each loader is created.
    set_seed(SEED)
    a = next(iter(get_dataloaders(batch_size=8, num_workers=num_workers)["train"]))
    set_seed(SEED)
    b = next(iter(get_dataloaders(batch_size=8, num_workers=num_workers)["train"]))
    report["train_order_reproducible"] = list(a[2]) == list(b[2])           # same image_ids
    report["train_augmentation_reproducible"] = bool(torch.equal(a[0], b[0]))  # same pixels
    print({k: report[k] for k in ("channels_equal_before_norm", "train_order_reproducible", "train_augmentation_reproducible")})

    # ---- Check 4: picture of augmentation -------------------------------------------------
    # Row 0: four training images with the deterministic pipeline. Rows 1-2: the same images
    # with two random augmentations each. Rotation must be small and nothing flipped.
    plain = get_datasets(augment_train=False)["train"]
    augmented = get_datasets(augment_train=True)["train"]
    idx = [0, 1, len(plain) - 2, len(plain) - 1]   # first two (NORMAL) and last two (PNEUMONIA) rows
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for col, i in enumerate(idx):
        img, label, image_id = plain[i]
        axes[0, col].imshow(denormalize(img).permute(1, 2, 0))   # (3,H,W) -> (H,W,3) for matplotlib
        axes[0, col].set_title(f"{image_id}\n224x224, no augmentation", fontsize=8)
        for row in (1, 2):
            axes[row, col].imshow(denormalize(augmented[i][0]).permute(1, 2, 0))
            axes[row, col].set_title(f"train augmentation #{row}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "augmentation_samples.png", dpi=110)

    (out / "loader_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", out)


if __name__ == "__main__":
    main()
