"""Stage 1.4: train and evaluate the custom CNN baseline.

Run from the repo root (GPU recommended):
    python -m src.train_baseline
Outputs:
    <checkpoint dir>/custom_cnn_best.pth      best weights (by val loss), NOT committed; upload to Drive
    results/stage1_metrics.csv                val and test metrics (agreed columns, issue #21 row 8)
    results/stage1_baseline/                  history, test predictions, curves, confusion matrices, ROC
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

MODEL_NAME = "custom_cnn"


def default_checkpoint_dir() -> Path:
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without val-loss improvement)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-cache", action="store_true", help="decode images every epoch instead of caching in memory")
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    return p.parse_args()


def check_cache_matches(n: int = 16) -> None:
    """The in-memory cache must give exactly the same tensors as reading from disk."""
    plain = get_datasets(cache=False)["val"]
    cached = ChestXrayDataset(plain.df.iloc[:n], plain.data_root, plain.transform, cache=True)
    for i in range(n):
        assert torch.equal(plain[i][0], cached[i][0]), f"cache mismatch at val item {i}"


def save_plots(history: pd.DataFrame, evals: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set_title("Loss (BCE with pos_weight)"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.epoch, history.val_auc, label="val AUC")
    axes[1].plot(history.epoch, history.val_f1, label="val F1")
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("val", "test")):
        ConfusionMatrixDisplay.from_predictions(evals[split]["labels"], (evals[split]["probs"] >= THRESHOLD).astype(int),
                                                display_labels=["NORMAL", "PNEUMONIA"], ax=ax, colorbar=False)
        ax.set_title(f"Custom CNN, {split} (threshold {THRESHOLD})")
    plt.tight_layout(); plt.savefig(out / "confusion_matrices.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    for split in ("val", "test"):
        RocCurveDisplay.from_predictions(evals[split]["labels"], evals[split]["probs"], name=split, ax=ax)
    ax.set_title("Custom CNN ROC"); plt.tight_layout(); plt.savefig(out / "roc.png", dpi=130); plt.close()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    if not args.no_cache:
        check_cache_matches()
    start = time.time()
    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, cache=not args.no_cache)
    print(f"Data ready in {time.time() - start:.0f}s:", {k: len(v.dataset) for k, v in loaders.items()})

    pos_weight = load_pos_weight()
    set_seed(SEED)  # same initial weights regardless of how data loading went
    model = CustomCNN(dropout=args.dropout).to(device)
    criterion = make_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

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

    ckpt_path = args.checkpoint_dir / f"{MODEL_NAME}_best.pth"
    model, history, best = fit(model, loaders, criterion, optimizer, device, max_epochs=args.epochs,
                               patience=args.patience, scheduler=scheduler, checkpoint_path=ckpt_path, extra=settings)
    print(f"Best epoch {best['epoch']} (val loss {best['val_loss']:.4f}); checkpoint: {ckpt_path}")

    # Evaluate the best model once on val and test
    evals = {split: predict(model, loaders[split], device, criterion) for split in ("val", "test")}
    rows = [metrics_row(MODEL_NAME, "unmasked", split, compute_metrics(evals[split]["labels"], evals[split]["probs"]))
            for split in ("val", "test")]
    metrics = save_metrics(rows, RESULTS_DIR / "stage1_metrics.csv")
    print(metrics.to_string(index=False))

    out = RESULTS_DIR / "stage1_baseline"
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "history.csv", index=False, lineterminator="\n")
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
