"""Stage 2.3: evaluate the chosen DenseNet-121 once on val and test and compare it with the baseline CNN (owned by Ajit, issue #9).

Steps:
1. Pick the model chosen in Stage 2.2: the run with the lowest VALIDATION loss in
   results/stage2_phaseB/tuning.csv (or the run given with --run-name). Nothing is chosen on test (#21 row 8).
2. Load its weights and predict val and test ONCE, threshold 0.5 (src.config.THRESHOLD).
3. Save the metrics in the agreed columns, the per-image predictions and the plots.
4. Build the comparison table baseline CNN vs DenseNet-121 from the committed Stage 1 results.

Run from the repo root (GPU recommended). The chosen Phase B checkpoint must exist in the checkpoint
directory (it is created by python -m src.finetune_densenet and is NOT committed; it lives on Drive):
    python -m src.evaluate_densenet
    python -m src.evaluate_densenet --run-name lr1e-4_bs16 --checkpoint-dir /path/to/checkpoints

Outputs:
    results/stage2_metrics.csv                  val and test metrics of DenseNet-121 (agreed columns, #21 row 8)
    results/stage2_comparison.csv               test metrics of baseline CNN and DenseNet-121 side by side
    results/stage2_densenet/                    {val,test}_predictions.csv, confusion_matrices.png, roc.png, settings.json
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw to files only; no screen needed (Kaggle, servers)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import LAST_CONV_LAYER, METRICS_COLUMNS, REPO_ROOT, RESULTS_DIR, SEED, THRESHOLD
from src.data.dataset import get_dataloaders
from src.data.splits import load_pos_weight
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.models.densenet121 import DenseNet121Classifier
from src.train import get_device, make_criterion, predict
from src.utils import set_seed

MODEL_NAME = "densenet121"


def default_checkpoint_dir() -> Path:
    """Same rule as the training scripts: /kaggle/working/checkpoints on Kaggle, else checkpoints/ in the repo."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None,
                   help="Phase B run to evaluate (default: the run with the lowest val loss in tuning.csv)")
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def best_run_name() -> str:
    """Name of the tuning run with the lowest validation loss (tuning.csv is sorted by val loss, best first)."""
    table = pd.read_csv(RESULTS_DIR / "stage2_phaseB" / "tuning.csv")
    return str(table.sort_values("val_loss").iloc[0]["run"])


def save_plots(evals: dict, out: Path) -> None:
    """Confusion matrices (val, test) and ROC curves, in the same style as the Stage 1 baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("val", "test")):
        ConfusionMatrixDisplay.from_predictions(evals[split]["labels"], (evals[split]["probs"] >= THRESHOLD).astype(int),
                                                display_labels=["NORMAL", "PNEUMONIA"], ax=ax, colorbar=False)
        ax.set_title(f"DenseNet-121, {split} (threshold {THRESHOLD})")
    plt.tight_layout(); plt.savefig(out / "confusion_matrices.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    for split in ("val", "test"):
        RocCurveDisplay.from_predictions(evals[split]["labels"], evals[split]["probs"], name=split, ax=ax)
    ax.set_title("DenseNet-121 ROC"); plt.tight_layout(); plt.savefig(out / "roc.png", dpi=130); plt.close()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    # 1. Which model? Chosen on validation loss only (Stage 2.2), never on test.
    run = args.run_name or best_run_name()
    ckpt_path = args.checkpoint_dir / f"{MODEL_NAME}_phaseB_{run}_best.pth"
    print("Evaluating run:", run, "| checkpoint:", ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Copy it from the team Drive folder or re-run Stage 2.2.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # dict with model_state + settings (CLAUDE.md)
    settings = ckpt["extra"]

    # 2. Rebuild the model and load the weights. Dropout is inactive in eval mode (predict() calls model.eval()).
    model = DenseNet121Classifier(pretrained=False, dropout=settings["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, cache=True)
    criterion = make_criterion(load_pos_weight(), device)

    # 3. Predict val and test once with the final model (no augmentation: only train is augmented).
    evals = {split: predict(model, loaders[split], device, criterion) for split in ("val", "test")}
    rows = [metrics_row(MODEL_NAME, "unmasked", split, compute_metrics(evals[split]["labels"], evals[split]["probs"]))
            for split in ("val", "test")]
    metrics = save_metrics(rows, RESULTS_DIR / "stage2_metrics.csv")
    print(metrics.to_string(index=False))

    out = RESULTS_DIR / "stage2_densenet"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        pd.DataFrame({"image_id": evals[split]["image_ids"], "label": evals[split]["labels"],
                      "prob_pneumonia": evals[split]["probs"].round(6),
                      "pred": (evals[split]["probs"] >= THRESHOLD).astype(int)}
                     ).to_csv(out / f"{split}_predictions.csv", index=False, lineterminator="\n")
    (out / "settings.json").write_text(json.dumps({
        "chosen_run": run, "chosen_by": "lowest validation loss (results/stage2_phaseB/tuning.csv)",
        "best_epoch": ckpt["epoch"], "best_val_loss": round(ckpt["val_loss"], 6), "threshold": THRESHOLD,
        "last_conv_layer": LAST_CONV_LAYER[MODEL_NAME], "training_settings": settings, "device": str(device),
    }, indent=2) + "\n")
    save_plots(evals, out)

    # 4. Comparison on the SAME test set with the SAME metrics: Stage 1 baseline (committed) vs DenseNet-121.
    baseline = pd.read_csv(RESULTS_DIR / "stage1_metrics.csv")
    both = pd.concat([baseline[baseline["split"] == "test"], metrics[metrics["split"] == "test"]], ignore_index=True)
    both = both[list(METRICS_COLUMNS)]
    both.to_csv(RESULTS_DIR / "stage2_comparison.csv", index=False, lineterminator="\n")
    print("\nTest comparison (baseline CNN vs DenseNet-121):")
    print(both.to_string(index=False))
    print("Saved results to", out)


if __name__ == "__main__":
    main()
