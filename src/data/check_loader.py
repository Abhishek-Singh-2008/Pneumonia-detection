"""Stage 1.3 check: load every image of all three splits once and save a sample grid.

Run from the repo root:
    python -m src.data.check_loader
Writes results/stage1_loader/loader_check.json and augmentation_samples.png.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.config import IMAGE_SIZE, RESULTS_DIR, SEED
from src.data.dataset import denormalize, get_dataloaders, get_datasets
from src.utils import set_seed


def main(batch_size: int = 64, num_workers: int = 2) -> None:
    set_seed(SEED)
    out = RESULTS_DIR / "stage1_loader"
    out.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    report = {}
    for name, loader in loaders.items():
        start = time.time()
        n, n_pos, ids = 0, 0, set()
        channel_sum = torch.zeros(3)
        channel_sq = torch.zeros(3)
        for images, labels, image_ids in loader:
            assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE), images.shape
            assert images.dtype == torch.float32
            n += images.shape[0]
            n_pos += int(labels.sum())
            ids.update(image_ids)
            channel_sum += images.mean(dim=(2, 3)).sum(0)
            channel_sq += (images ** 2).mean(dim=(2, 3)).sum(0)
        expected = len(loader.dataset)
        assert n == expected == len(ids), (name, n, expected, len(ids))
        mean = channel_sum / n
        report[name] = {
            "images": n,
            "pneumonia": n_pos,
            "batches": len(loader),
            "seconds": round(time.time() - start, 1),
            "channel_mean": [round(float(x), 3) for x in mean],
            "channel_std": [round(float(x), 3) for x in (channel_sq / n - mean ** 2).sqrt()],
        }
        print(name, report[name], flush=True)

    # Grayscale -> 3 channels: before normalisation all three channels must be equal
    ds = get_datasets(augment_train=False)["val"]
    raw = denormalize(ds[0][0])
    report["channels_equal_before_norm"] = bool(torch.allclose(raw[0], raw[1], atol=1e-6) and torch.allclose(raw[1], raw[2], atol=1e-6))
    assert report["channels_equal_before_norm"]

    # Determinism: after set_seed, a fresh train loader gives the same first batch
    # (with num_workers=0 augmentation uses the global torch RNG, so set_seed matters)
    set_seed(SEED)
    a = next(iter(get_dataloaders(batch_size=8, num_workers=num_workers)["train"]))
    set_seed(SEED)
    b = next(iter(get_dataloaders(batch_size=8, num_workers=num_workers)["train"]))
    report["train_order_reproducible"] = list(a[2]) == list(b[2])
    report["train_augmentation_reproducible"] = bool(torch.equal(a[0], b[0]))
    print({k: report[k] for k in ("channels_equal_before_norm", "train_order_reproducible", "train_augmentation_reproducible")})

    # Visual check: the same 4 training images without and with augmentation
    plain = get_datasets(augment_train=False)["train"]
    augmented = get_datasets(augment_train=True)["train"]
    idx = [0, 1, len(plain) - 2, len(plain) - 1]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for col, i in enumerate(idx):
        img, label, image_id = plain[i]
        axes[0, col].imshow(denormalize(img).permute(1, 2, 0))
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
