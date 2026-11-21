"""Stage 2.2: DenseNet-121 Phase B, unfreeze and fine-tune with a low learning rate (owned by Ajit, issue #8).

Steps:
1. Seed everything, pick the GPU if there is one.
2. Build the train/val/test loaders from the committed splits (same pipeline as Stage 1, no new splits).
3. Load the Phase A weights (head already trained), then UNFREEZE the whole network.
4. Fine-tune with a small learning rate. The backbone is already good (ImageNet), so large steps
   would destroy what it learned; this is why Phase B uses a lower learning rate than Phase A.
5. Early stopping on the validation loss only (src/train.py: fit; #21 row 8).
6. Save weights, history, settings and VAL metrics of this run, and add one row to the tuning table.

Tuning (issue #8): run this script a few times with different --lr / --batch-size / --epochs and a
different --run-name. Every run adds a row to results/stage2_phaseB/tuning.csv. The best run is the one
with the lowest validation loss. The test set is NOT evaluated here: it is used once, for the chosen
model only, in Stage 2.3 (issue #9), so nothing is ever chosen on it (#21 row 8, CLAUDE.md).

Run from the repo root (GPU recommended, e.g. Kaggle T4). The Phase A checkpoint must exist
(python -m src.train_densenet); on Kaggle upload it from the team Drive folder or re-run Phase A first:
    python -m src.finetune_densenet --run-name lr1e-4
    python -m src.finetune_densenet --run-name lr5e-5 --lr 5e-5
    python -m src.finetune_densenet --help

Outputs (RUN = the --run-name):
    <checkpoint dir>/densenet121_phaseB_RUN_best.pth   best weights of the run (by val loss). NOT committed; upload to Drive
    results/stage2_phaseB/RUN/                         history.csv, settings.json, val_metrics.csv
    results/stage2_phaseB/tuning.csv                   one row per run: settings + best val loss/AUC/F1
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from src.config import (IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, REPO_ROOT, RESULTS_DIR, ROTATION_DEGREES, SEED,
                        THRESHOLD)
from src.data.dataset import get_dataloaders
from src.data.splits import load_pos_weight
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.models.densenet121 import DenseNet121Classifier
from src.train import fit, get_device, make_criterion, predict
from src.utils import set_seed

# Name used in the metrics CSV and the checkpoint file names.
MODEL_NAME = "densenet121"


def default_checkpoint_dir() -> Path:
    """/kaggle/working/checkpoints on Kaggle (downloadable as kernel output), otherwise checkpoints/ (git-ignored)."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def parse_args():
    """Command-line options. Each tuning run changes --lr, --batch-size and/or --epochs."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="lr1e-4", help="short name of this tuning run (used in file names)")
    p.add_argument("--epochs", type=int, default=15, help="maximum number of epochs")
    p.add_argument("--patience", type=int, default=4, help="early stopping patience (epochs without val-loss improvement)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate for the WHOLE network (low on purpose)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout before the final layer (same as Phase A)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-cache", action="store_true", help="decode images every epoch instead of caching in memory")
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    p.add_argument("--phase-a-checkpoint", type=Path, default=None,
                   help="Phase A weights (default: <checkpoint dir>/densenet121_phaseA_best.pth)")
    return p.parse_args()


def update_tuning_table(row: dict, path: Path) -> pd.DataFrame:
    """Add this run to tuning.csv (replacing an older row with the same run name) and sort by val loss.

    Args:
        row: one dict per run (settings and best validation numbers).
        path: results/stage2_phaseB/tuning.csv.

    Returns:
        The whole table, best run (lowest val loss) first.
    """
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[table["run"] != row["run"]]  # re-running a name overwrites its row
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("val_loss")
    table.to_csv(path, index=False, lineterminator="\n")
    return table


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # Same loaders as Stage 1 and Phase A (#21 rows 5 and 6).
    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, cache=not args.no_cache)
    print("Data ready:", {k: len(v.dataset) for k, v in loaders.items()})

    pos_weight = load_pos_weight()  # from the train split only (#21 row 7)
    set_seed(SEED)

    # Start from the Phase A model: ImageNet backbone + trained head.
    phase_a_path = args.phase_a_checkpoint or args.checkpoint_dir / f"{MODEL_NAME}_phaseA_best.pth"
    if not phase_a_path.exists():
        raise FileNotFoundError(f"Phase A checkpoint not found: {phase_a_path}. Run python -m src.train_densenet first.")
    # weights_only=False because the checkpoint also stores the settings dict (see CLAUDE.md).
    phase_a = torch.load(phase_a_path, map_location="cpu", weights_only=False)
    # pretrained=False: the weights come from the checkpoint, no need to download ImageNet again.
    model = DenseNet121Classifier(pretrained=False, dropout=args.dropout)
    model.load_state_dict(phase_a["model_state"])
    model.unfreeze_backbone()  # Phase B: every layer gets gradients, BatchNorm layers train again
    model = model.to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"Loaded Phase A (epoch {phase_a['epoch']}, val loss {phase_a['val_loss']:.4f}). "
          f"Trainable parameters: {n_trainable}")

    criterion = make_criterion(pos_weight, device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    settings = {
        "model": MODEL_NAME, "phase": "B (all layers fine-tuned)", "run": args.run_name,
        "start_from": f"Phase A checkpoint, epoch {phase_a['epoch']}", "dropout": args.dropout, "seed": SEED,
        "image_size": IMAGE_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "rotation_degrees": ROTATION_DEGREES,
        "pos_weight": pos_weight, "threshold": THRESHOLD, "optimizer": "Adam", "lr": args.lr,
        "weight_decay": args.weight_decay, "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2) on val loss",
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "selection": "lowest val loss", "amp": device.type == "cuda", "trainable_parameters": n_trainable,
    }
    print(json.dumps(settings, indent=2))

    ckpt_path = args.checkpoint_dir / f"{MODEL_NAME}_phaseB_{args.run_name}_best.pth"
    model, history, best = fit(model, loaders, criterion, optimizer, device, max_epochs=args.epochs,
                               patience=args.patience, scheduler=scheduler, checkpoint_path=ckpt_path, extra=settings)
    print(f"Best epoch {best['epoch']} (val loss {best['val_loss']:.4f}); checkpoint: {ckpt_path}")

    # Validation metrics of the best epoch. Test is deliberately not touched (see module docstring).
    val = predict(model, loaders["val"], device, criterion)
    val_metrics = compute_metrics(val["labels"], val["probs"])
    row = metrics_row(MODEL_NAME, f"unmasked_phaseB_{args.run_name}", "val", val_metrics)
    out = RESULTS_DIR / "stage2_phaseB"
    run_dir = out / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(save_metrics([row], run_dir / "val_metrics.csv").to_string(index=False))
    history.to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
    (run_dir / "settings.json").write_text(json.dumps({**settings, "best_epoch": best["epoch"],
                                                       "best_val_loss": round(best["val_loss"], 6),
                                                       "device": str(device)}, indent=2) + "\n")

    table = update_tuning_table({
        "run": args.run_name, "lr": args.lr, "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "max_epochs": args.epochs, "best_epoch": best["epoch"], "val_loss": round(best["val_loss"], 4),
        "val_auc": round(val_metrics["auc"], 4), "val_f1": round(val_metrics["f1"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4), "val_recall": round(val_metrics["recall"], 4),
    }, out / "tuning.csv")
    print("\nTuning table (best run first, chosen by val loss):")
    print(table.to_string(index=False))
    print("Saved results to", run_dir)


if __name__ == "__main__":
    main()
