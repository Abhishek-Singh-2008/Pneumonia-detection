"""Stage 1.4: train and evaluate the custom CNN baseline (owned by Abhishek, issue #5).

Steps:
1. Seed everything, pick the GPU if there is one.
2. Check that the in-memory image cache gives exactly the same tensors as reading from disk.
3. Build the train/val/test loaders from the committed splits (data/splits/).
4. Create the model, the weighted loss, Adam and a learning-rate scheduler.
5. Train with early stopping on the validation loss (src/train.py: fit).
6. Evaluate the best model ONCE on val and test (threshold 0.5; nothing is chosen on test, #21 row 8).
7. Save metrics, per-image predictions, training history, settings and plots.

Run from the repo root (GPU recommended; a few minutes on a Kaggle T4):
    python -m src.train_baseline                    # default settings, as used for the reported results
    python -m src.train_baseline --help             # list the options

Outputs:
    <checkpoint dir>/custom_cnn_best.pth      best weights (by val loss). NOT committed (too large);
                                              upload to the team Google Drive folder
    results/stage1_metrics.csv                val and test metrics (agreed columns, #21 row 8)
    results/stage1_baseline/                  history.csv, settings.json, {val,test}_predictions.csv,
                                              training_curves.png, confusion_matrices.png, roc.png
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files only; no screen needed (Kaggle, servers)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LAST_CONV_LAYER, REPO_ROOT, RESULTS_DIR,
                        ROTATION_DEGREES, SEED, THRESHOLD)
from src.data.dataset import ChestXrayDataset, get_dataloaders, get_datasets
from src.data.splits import load_pos_weight
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.models.custom_cnn import CHANNELS, CustomCNN
from src.train import fit, get_device, make_criterion, predict
from src.utils import set_seed

# Name used in the metrics CSV, the checkpoint file name and LAST_CONV_LAYER.
MODEL_NAME = "custom_cnn"


def default_checkpoint_dir() -> Path:
    """Where to save weights: /kaggle/working/checkpoints on Kaggle (downloadable as kernel output),
    otherwise checkpoints/ in the repo (git-ignored)."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def parse_args():
    """Command-line options. The defaults are the settings used for the reported results."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=30, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without val-loss improvement)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty on the weights (reduces overfitting)")
    p.add_argument("--dropout", type=float, default=0.3, help="dropout before the final layer")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--no-cache", action="store_true", help="decode images every epoch instead of caching in memory")
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    return p.parse_args()


def check_cache_matches(n: int = 16) -> None:
    """Make sure the in-memory cache gives exactly the same tensors as reading from disk.

    The cache is only a speed-up; if it ever changed the pixels, training would silently use
    different data. Compares the first n validation images (no augmentation) pixel by pixel.

    Raises:
        AssertionError: if any image differs.
    """
    plain = get_datasets(cache=False)["val"]
    cached = ChestXrayDataset(plain.df.iloc[:n], plain.data_root, plain.transform, cache=True)
    for i in range(n):
        assert torch.equal(plain[i][0], cached[i][0]), f"cache mismatch at val item {i}"


def save_plots(history: pd.DataFrame, evals: dict, out: Path) -> None:
    """Save the training curves, confusion matrices and ROC curves as PNG files.

    Args:
        history: per-epoch DataFrame returned by fit().
        evals: {"val": predict(...), "test": predict(...)} for the best model.
        out: folder to write into (results/stage1_baseline/).
    """
    # Training curves: train vs val loss (left), val AUC and F1 (right).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set_title("Loss (BCE with pos_weight)"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.epoch, history.val_auc, label="val AUC")
    axes[1].plot(history.epoch, history.val_f1, label="val F1")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=130); plt.close()

    # Confusion matrices for val and test, at the fixed threshold.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("val", "test")):
        ConfusionMatrixDisplay.from_predictions(evals[split]["labels"], (evals[split]["probs"] >= THRESHOLD).astype(int),
                                                display_labels=["NORMAL", "PNEUMONIA"], ax=ax, colorbar=False)
        ax.set_title(f"Custom CNN, {split} (threshold {THRESHOLD})")
    plt.tight_layout(); plt.savefig(out / "confusion_matrices.png", dpi=130); plt.close()

    # ROC curves (threshold-free view of how well the model ranks pneumonia above normal).
    fig, ax = plt.subplots(figsize=(5, 5))
    for split in ("val", "test"):
        RocCurveDisplay.from_predictions(evals[split]["labels"], evals[split]["probs"], name=split, ax=ax)
    ax.set_title("Custom CNN ROC"); plt.tight_layout(); plt.savefig(out / "roc.png", dpi=130); plt.close()


def main() -> None:
    """Train, evaluate and save everything (see the module docstring for the steps)."""
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # ---- Data --------------------------------------------------------------------------
    if not args.no_cache:
        check_cache_matches()
    start = time.time()
    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, cache=not args.no_cache)
    print(f"Data ready in {time.time() - start:.0f}s:", {k: len(v.dataset) for k, v in loaders.items()})

    # ---- Model, loss, optimiser --------------------------------------------------------
    pos_weight = load_pos_weight()   # computed from the train split only (#21 row 7)
    set_seed(SEED)  # same initial weights regardless of how much randomness data loading used
    model = CustomCNN(dropout=args.dropout).to(device)
    criterion = make_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Halve the learning rate when the val loss has not improved for 2 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Everything needed to reproduce or describe the run; saved in the checkpoint and settings.json.
    settings = {
        "model": MODEL_NAME, "channels": list(CHANNELS), "dropout": args.dropout,
        "last_conv_layer": LAST_CONV_LAYER[MODEL_NAME], "seed": SEED, "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "rotation_degrees": ROTATION_DEGREES,
        "pos_weight": pos_weight, "threshold": THRESHOLD, "optimizer": "Adam", "lr": args.lr,
        "weight_decay": args.weight_decay, "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2) on val loss",
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "selection": "lowest val loss", "amp": device.type == "cuda",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(settings, indent=2))

    # ---- Train (model selection on val only) ----------------------------------------------
    ckpt_path = args.checkpoint_dir / f"{MODEL_NAME}_best.pth"
    model, history, best = fit(model, loaders, criterion, optimizer, device, max_epochs=args.epochs,
                               patience=args.patience, scheduler=scheduler, checkpoint_path=ckpt_path, extra=settings)
    print(f"Best epoch {best['epoch']} (val loss {best['val_loss']:.4f}); checkpoint: {ckpt_path}")

    # ---- Evaluate the best model once on val and test ----------------------------------
    # This is the only place the test set is used, and only to report numbers.
    evals = {split: predict(model, loaders[split], device, criterion) for split in ("val", "test")}
    rows = [metrics_row(MODEL_NAME, "unmasked", split, compute_metrics(evals[split]["labels"], evals[split]["probs"]))
            for split in ("val", "test")]
    metrics = save_metrics(rows, RESULTS_DIR / "stage1_metrics.csv")
    print(metrics.to_string(index=False))

    # ---- Save everything else ---------------------------------------------------------------
    out = RESULTS_DIR / "stage1_baseline"
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "history.csv", index=False, lineterminator="\n")
    # Per-image predictions, keyed by image_id, so Stage 3 can pick correct and wrong predictions
    # for Grad-CAM and join them with the lung masks.
    for split in ("val", "test"):
        pd.DataFrame({"image_id": evals[split]["image_ids"], "label": evals[split]["labels"],
                      "prob_pneumonia": evals[split]["probs"].round(6),
                      "pred": (evals[split]["probs"] >= THRESHOLD).astype(int)}
                     ).to_csv(out / f"{split}_predictions.csv", index=False, lineterminator="\n")
    (out / "settings.json").write_text(json.dumps({**settings, "best_epoch": best["epoch"],
                                                   "best_val_loss": round(best["val_loss"], 6),
                                                   "device": str(device)}, indent=2) + "\n")
    save_plots(history, evals, out)
    print("Saved results to", out)


if __name__ == "__main__":
    main()
