"""Stage 2.1: DenseNet-121 Phase A, train only the new final layer (owned by Ajit, issue #7).

Steps:
1. Seed everything, pick the GPU if there is one.
2. Build the train/val/test loaders from the committed splits (same pipeline as Stage 1, no new splits).
3. Load ImageNet-pretrained DenseNet-121, freeze the backbone, train only the 1-logit head.
4. Early stopping on the validation loss only (src/train.py: fit; #21 row 8).
5. Save the best weights, history, settings and VAL metrics.

The test set is NOT evaluated here. It is only used once per model for final numbers
(Stage 2.3, issue #9), after Phase B, so nothing is ever chosen on it (#21 row 8).

Run from the repo root (GPU recommended, e.g. Kaggle T4):
    python -m src.train_densenet
    python -m src.train_densenet --help

Outputs:
    <checkpoint dir>/densenet121_phaseA_best.pth   best weights (by val loss). NOT committed; upload to Drive
    results/stage2_phaseA/history.csv              per-epoch losses and val metrics
    results/stage2_phaseA/settings.json            all settings used
    results/stage2_phaseA/val_metrics.csv          val metrics in the agreed columns (#21 row 8)
"""
import argparse
import json
import os
from pathlib import Path

import torch

from src.config import (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, REPO_ROOT, RESULTS_DIR, ROTATION_DEGREES, SEED,
                        THRESHOLD)
from src.data.dataset import get_dataloaders
from src.data.splits import load_pos_weight
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.models.densenet121 import DenseNet121Classifier
from src.train import fit, get_device, make_criterion, predict
from src.utils import set_seed

# Name used in the metrics CSV and the checkpoint file name.
MODEL_NAME = "densenet121"
PHASE = "phaseA"


def default_checkpoint_dir() -> Path:
    """/kaggle/working/checkpoints on Kaggle (downloadable as kernel output), otherwise checkpoints/ (git-ignored)."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without val-loss improvement)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for the new layer")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout before the final layer")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-cache", action="store_true", help="decode images every epoch instead of caching in memory")
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # Same loaders as Stage 1: committed splits, grayscale->3 channels, 224x224, ImageNet mean/std,
    # train-only rotation of +/-10 degrees and no horizontal flip (#21 rows 5 and 6).
    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, cache=not args.no_cache)
    print("Data ready:", {k: len(v.dataset) for k, v in loaders.items()})

    pos_weight = load_pos_weight()  # from the train split only (#21 row 7)
    set_seed(SEED)  # same initial head weights regardless of how data loading went
    model = DenseNet121Classifier(pretrained=True, dropout=args.dropout).to(device)
    model.freeze_backbone()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable parameters: {n_trainable} of {sum(p.numel() for p in model.parameters())}")

    criterion = make_criterion(pos_weight, device)
    # Only the head is given to the optimizer; the frozen backbone has no gradients anyway.
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    settings = {
        "model": MODEL_NAME, "phase": "A (backbone frozen, head only)", "pretrained": "ImageNet (IMAGENET1K_V1)",
        "dropout": args.dropout, "seed": SEED, "image_size": IMAGE_SIZE, "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD, "rotation_degrees": ROTATION_DEGREES, "pos_weight": pos_weight,
        "threshold": THRESHOLD, "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2) on val loss", "batch_size": args.batch_size,
        "max_epochs": args.epochs, "patience": args.patience, "selection": "lowest val loss",
        "amp": device.type == "cuda", "trainable_parameters": n_trainable,
    }
    print(json.dumps(settings, indent=2))

    ckpt_path = args.checkpoint_dir / f"{MODEL_NAME}_{PHASE}_best.pth"
    model, history, best = fit(model, loaders, criterion, optimizer, device, max_epochs=args.epochs,
                               patience=args.patience, scheduler=scheduler, checkpoint_path=ckpt_path, extra=settings)
    print(f"Best epoch {best['epoch']} (val loss {best['val_loss']:.4f}); checkpoint: {ckpt_path}")

    # Validation metrics of the best epoch. Test is deliberately not touched (see module docstring).
    val = predict(model, loaders["val"], device, criterion)
    row = metrics_row(MODEL_NAME, "unmasked_phaseA", "val", compute_metrics(val["labels"], val["probs"]))
    out = RESULTS_DIR / "stage2_phaseA"
    out.mkdir(parents=True, exist_ok=True)
    print(save_metrics([row], out / "val_metrics.csv").to_string(index=False))
    history.to_csv(out / "history.csv", index=False, lineterminator="\n")
    (out / "settings.json").write_text(json.dumps({**settings, "best_epoch": best["epoch"],
                                                   "best_val_loss": round(best["val_loss"], 6),
                                                   "device": str(device)}, indent=2) + "\n")
    print("Saved results to", out)


if __name__ == "__main__":
    main()
