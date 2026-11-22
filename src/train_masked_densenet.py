"""Stage 4.1: train DenseNet-121 on the Stage 3 lung-masked images (issue #16).

This repeats Stage 2's two training phases and selected Phase B settings on the
masked copies. Early stopping/model selection use validation loss only; this
script never evaluates the test split.

Issue #16 is done only when the masked model is trained with the SAME settings as
Stage 2, so before training the script reads the committed Stage 2 settings back
from results/stage2_phaseA/settings.json and results/stage2_densenet/settings.json
and stops if anything here has drifted from them. It also checks the masked copies
(split sizes, and that no surviving pixel lies outside its lung mask) so a resize
bug cannot silently train the model on the wrong pixels.

First create masked copies with:
    python -m src.masking.masked_dataset --mask-dir /path/to/lung_masks
Then run from the repository root:
    python -m src.train_masked_densenet --masked-data-root /path/to/masked_chest_xray \
        --mask-dir /path/to/lung_masks

--mask-dir is optional; when given, the pixel/mask alignment check runs.

Outputs:
    <checkpoint dir>/densenet121_masked_phase{A,B}_best.pth
    results/stage4_masked_densenet/phase{A,B}/{history.csv,settings.json,val_metrics.csv,val_predictions.csv}
    results/stage4_masked_densenet/summary.json      both phases, and the Stage 2 settings they repeat
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
    RESULTS_DIR,
    ROTATION_DEGREES,
    SEED,
    THRESHOLD,
)
from src.data.splits import load_pos_weight, load_split
from src.masking.lung_masks import load_mask
from src.masking.masked_dataset import get_masked_dataloaders
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.models.densenet121 import DenseNet121Classifier
from src.train import fit, get_device, make_criterion, predict
from src.utils import set_seed

MODEL_NAME = "densenet121"

# Agreed vocabulary from src.metrics.metrics_row: this model is trained on masked data.
# Intermediate rows use "masked_phaseA" / "masked_phaseB"; Stage 4.2 writes "masked".
MASKED_SETTING = "masked"

# Stage 2's selected settings. Source of truth:
#   results/stage2_phaseA/settings.json            (Phase A, best epoch 19)
#   results/stage2_densenet/settings.json          (chosen Phase B run lr1e-4_bs16)
#   results/stage2_phaseB/lr1e-4_bs16/settings.json
# Issue #16 is done only when these are repeated EXACTLY, so check_stage2_settings()
# compares every value below with those files before training starts.
PHASE_A = {"max_epochs": 20, "patience": 5, "batch_size": 32, "lr": 1e-3}
PHASE_B = {"max_epochs": 15, "patience": 4, "batch_size": 16, "lr": 1e-4}
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
SCHEDULER = "ReduceLROnPlateau(factor=0.5, patience=2) on val loss"

# Committed split sizes (data/splits/split_summary.json); a different count means the
# masked copies were built from the wrong split files.
EXPECTED_SPLIT_SIZES = {"train": 4578, "val": 654, "test": 624}

STAGE4_DIR = RESULTS_DIR / "stage4_masked_densenet"
STAGE2_PHASE_A_SETTINGS = RESULTS_DIR / "stage2_phaseA" / "settings.json"
STAGE2_PHASE_B_SUMMARY = RESULTS_DIR / "stage2_densenet" / "settings.json"


def default_checkpoint_dir() -> Path:
    """Return the ignored checkpoint folder used on Kaggle or a local machine."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def read_json(path: Path) -> dict:
    """Read one JSON file, with a clear error if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage2_reference() -> dict:
    """Load the committed Stage 2 settings that the masked run must repeat."""
    phase_a = read_json(STAGE2_PHASE_A_SETTINGS)
    summary = read_json(STAGE2_PHASE_B_SUMMARY)
    run = str(summary["chosen_run"])
    phase_b_path = RESULTS_DIR / "stage2_phaseB" / run / "settings.json"
    return {
        "chosen_run": run,
        "best_epoch": summary.get("best_epoch"),
        "best_val_loss": summary.get("best_val_loss"),
        "phase_a": phase_a,
        "phase_b": read_json(phase_b_path),
        "sources": {
            "phase_a": STAGE2_PHASE_A_SETTINGS.relative_to(REPO_ROOT).as_posix(),
            "phase_b": phase_b_path.relative_to(REPO_ROOT).as_posix(),
            "phase_b_summary": STAGE2_PHASE_B_SUMMARY.relative_to(REPO_ROOT).as_posix(),
        },
    }


def _same_value(got, expected) -> bool:
    """Compare one setting, treating int/float the same when the values are equal."""
    if isinstance(expected, bool) or isinstance(got, bool):
        return got == expected
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(got) == len(expected) and all(_same_value(g, e) for g, e in zip(got, expected))
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(expected)) <= 1e-12
    return got == expected


def check_stage2_settings(args) -> dict:
    """Prove this run repeats Stage 2's settings, or stop before training.

    Compares the values used here with results/stage2_phaseA/settings.json and the
    chosen run's results/stage2_phaseB/<run>/settings.json. Keys that only make sense
    for one stage (start_from, trainable_parameters, best_epoch, ...) are skipped.

    CLI epoch/patience overrides are allowed for smoke tests, but then the run does
    NOT repeat Stage 2 and summary.json records that.

    Returns:
        dict for summary.json: whether the settings match, and any overrides.
    """
    reference = stage2_reference()
    here = {
        "model": MODEL_NAME,
        "dropout": args.dropout,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "rotation_degrees": ROTATION_DEGREES,
        "pos_weight": load_pos_weight(),
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "scheduler": SCHEDULER,
        "selection": "lowest val loss",
    }
    planned = {
        "A": {**here, "phase": PHASE_A, "reference": reference["phase_a"]},
        "B": {**here, "phase": PHASE_B, "reference": reference["phase_b"]},
    }

    problems = []
    for phase, plan in planned.items():
        expected = plan["reference"]
        actual = {**{k: v for k, v in plan.items() if k not in ("phase", "reference")}, **plan["phase"]}
        for key, value in expected.items():
            if key not in actual or actual[key] is None:
                continue
            if not _same_value(actual[key], value):
                problems.append(f"phase {phase}: {key} is {actual[key]!r} here, Stage 2 has {value!r}")
    if problems:
        raise SystemExit(
            "Stage 2 settings check FAILED (issue #16 asks for the same settings):\n  "
            + "\n  ".join(problems)
        )

    overrides = {}
    if args.phase_a_epochs != PHASE_A["max_epochs"]:
        overrides["phase_a_max_epochs"] = args.phase_a_epochs
    if args.phase_a_patience != PHASE_A["patience"]:
        overrides["phase_a_patience"] = args.phase_a_patience
    if args.phase_b_epochs != PHASE_B["max_epochs"]:
        overrides["phase_b_max_epochs"] = args.phase_b_epochs
    if args.phase_b_patience != PHASE_B["patience"]:
        overrides["phase_b_patience"] = args.phase_b_patience

    print("Stage 2 settings match (Phase A and Phase B).")
    if overrides:
        print(f"WARNING: command-line overrides mean this run does NOT repeat Stage 2: {overrides}")

    return {
        "checked": True,
        "matches_stage2": not overrides,
        "overrides": overrides,
        "stage2_chosen_run": reference["chosen_run"],
        "stage2_phase_a_settings": reference["sources"]["phase_a"],
        "stage2_phase_b_settings": reference["sources"]["phase_b"],
    }


def check_masked_data(loaders: dict, args) -> dict:
    """Verify the masked copies before training, so a data bug cannot run silently.

    Checks:
        - split sizes equal the committed splits (4,578 / 654 / 624);
        - with --mask-dir, every masked PNG has no nonzero pixel outside its lung mask,
          and every mask still keeps some lung pixels (i.e. the copy is not all black).

    Returns:
        dict for summary.json describing what was checked.
    """
    sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    if sizes != EXPECTED_SPLIT_SIZES:
        raise SystemExit(f"Masked split sizes {sizes} do not match the committed splits {EXPECTED_SPLIT_SIZES}")
    print("Masked split sizes match the committed splits:", sizes)

    result = {"split_sizes": sizes}
    if args.mask_dir is None:
        result["alignment_check"] = "skipped (no --mask-dir)"
        print("No --mask-dir given: skipping the pixel/mask alignment check.")
        return result

    failures, checked = [], 0
    for name in ("train", "val", "test"):
        for row in load_split(name).itertuples(index=False):
            masked_path = Path(args.masked_data_root) / Path(row.filepath).with_suffix(".png")
            with Image.open(masked_path) as masked_image:
                pixels = np.asarray(masked_image.convert("L"))
            mask = load_mask(args.mask_dir, row.image_id)
            outside = int(np.count_nonzero((pixels > 0) & ~mask))
            kept = int(np.count_nonzero((pixels > 0) & mask))
            checked += 1
            if outside or kept == 0:
                failures.append({"image_id": row.image_id, "split": name,
                                 "pixels_outside_mask": outside, "lung_pixels_kept": kept})
            if checked % 1000 == 0:
                print(f"  checked {checked} masked images", flush=True)

    if failures:
        print("First failures:", json.dumps(failures[:10], indent=2))
        raise SystemExit(
            f"Masked image check FAILED for {len(failures)} of {checked} images: masked copies "
            "and lung masks are not aligned, so training would use the wrong pixels."
        )
    print(f"Masked image check passed: {checked} images, no pixel outside its mask, lungs kept.")
    result["alignment_check"] = (
        f"{checked} images: every masked pixel lies inside its lung mask and every mask keeps lung pixels"
    )
    result["mask_dir"] = str(args.mask_dir.resolve())
    return result


def run_phase(
    phase: str,
    loaders: dict,
    model: DenseNet121Classifier,
    device: torch.device,
    args,
    phase_settings: dict,
    checkpoint_path: Path,
    start_from: str | None = None,
) -> tuple[DenseNet121Classifier, dict, dict]:
    """Train one DenseNet phase, save its checkpoint and validation-only artifacts.

    Args:
        phase: `"A"` for frozen-head training or `"B"` for full fine-tuning.
        loaders: DataLoaders built from the committed splits and masked data root.
        model: configured DenseNet model; caller freezes or unfreezes its backbone.
        device: PyTorch device for model and loss.
        args: parsed CLI arguments.
        phase_settings: phase hyperparameters and shared reproducibility metadata.
        checkpoint_path: output path for the best-by-validation-loss checkpoint.
        start_from: provenance string for Phase B (Phase A epoch and val loss).

    Returns:
        The best-checkpoint model, its fit result dictionary, and the val metrics row.
    """
    criterion = make_criterion(load_pos_weight(), device)
    optimizer = torch.optim.Adam(
        model.trainable_parameters(), lr=phase_settings["lr"], weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    settings = {
        **phase_settings,
        "model": MODEL_NAME,
        "phase": "A (backbone frozen, head only)" if phase == "A" else "B (all layers fine-tuned)",
        "data_setting": MASKED_SETTING,
        "dropout": args.dropout,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "rotation_degrees": ROTATION_DEGREES,
        "pos_weight": load_pos_weight(),
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "scheduler": SCHEDULER,
        "selection": "lowest val loss",
        "amp": device.type == "cuda",
        "trainable_parameters": sum(p.numel() for p in model.trainable_parameters()),
        "masked_data_root": str(args.masked_data_root.resolve()),
    }
    if phase == "A":
        settings["pretrained"] = "ImageNet (IMAGENET1K_V1)"
    else:
        settings["start_from"] = start_from or "masked Phase A checkpoint"
    print(f"Phase {phase} settings:\n{json.dumps(settings, indent=2)}")

    model, history, best = fit(
        model,
        loaders,
        criterion,
        optimizer,
        device,
        max_epochs=phase_settings["max_epochs"],
        patience=phase_settings["patience"],
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        extra=settings,
    )

    # Persist validation metrics and per-image predictions only; the test split is
    # held back for Stage 4.2.
    val = predict(model, loaders["val"], device, criterion)
    val_metrics = compute_metrics(val["labels"], val["probs"])
    val_row = metrics_row(MODEL_NAME, f"{MASKED_SETTING}_phase{phase}", "val", val_metrics)
    output_dir = STAGE4_DIR / f"phase{phase}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics([val_row], output_dir / "val_metrics.csv")
    pd.DataFrame({
        "image_id": val["image_ids"],
        "label": val["labels"].astype(int),
        "prob_pneumonia": np.round(val["probs"], 6),
        "pred": (val["probs"] >= THRESHOLD).astype(int),
    }).to_csv(output_dir / "val_predictions.csv", index=False, lineterminator="\n")
    history.to_csv(output_dir / "history.csv", index=False, lineterminator="\n")
    (output_dir / "settings.json").write_text(
        json.dumps(
            {
                **settings,
                "best_epoch": best["epoch"],
                "best_val_loss": round(best["val_loss"], 6),
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase {phase}: best epoch {best['epoch']}, val loss {best['val_loss']:.6f}")
    return model, best, val_row


def masked_dataset_summary(data_root: Path) -> dict | None:
    """Read the masked-copy summary when it exists (split counts and flagged masks)."""
    path = Path(data_root) / "masked_dataset_summary.json"
    return read_json(path) if path.is_file() else None


def write_summary(phase_a: dict, phase_b: dict, args, checks: dict) -> Path:
    """Write the one JSON the Stage 4 report cites for the training run."""
    summary = {
        "stage": "4.1 (masked DenseNet-121 training)",
        "issue": 16,
        "model": MODEL_NAME,
        "data_setting": MASKED_SETTING,
        "stage2_settings_check": checks["stage2"],
        "masked_data_check": checks["data"],
        "masked_dataset": masked_dataset_summary(args.masked_data_root),
        "phase_a": {
            "best_epoch": phase_a["epoch"],
            "best_val_loss": round(phase_a["val_loss"], 6),
            "best_val_auc": round(phase_a["val_auc"], 6),
            "checkpoint": str((args.checkpoint_dir / f"{MODEL_NAME}_masked_phaseA_best.pth").resolve()),
        },
        "phase_b": {
            "best_epoch": phase_b["epoch"],
            "best_val_loss": round(phase_b["val_loss"], 6),
            "best_val_auc": round(phase_b["val_auc"], 6),
            "checkpoint": str((args.checkpoint_dir / f"{MODEL_NAME}_masked_phaseB_best.pth").resolve()),
        },
        "selection": "lowest validation loss (never the test split)",
        "test_split": "not evaluated in Stage 4.1; see Stage 4.2 (src/evaluate_masked.py)",
    }
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGE4_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def parse_args():
    """Parse the masked-data path and Stage 2-compatible training options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masked-data-root", type=Path, required=True, help="root produced by src.masking.masked_dataset")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Stage 3 mask folder; enables the pixel/mask alignment check")
    parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true", help="decode masked images every epoch")
    parser.add_argument("--phase-a-epochs", type=int, default=PHASE_A["max_epochs"])
    parser.add_argument("--phase-a-patience", type=int, default=PHASE_A["patience"])
    parser.add_argument("--phase-b-epochs", type=int, default=PHASE_B["max_epochs"])
    parser.add_argument("--phase-b-patience", type=int, default=PHASE_B["patience"])
    return parser.parse_args()


def main() -> None:
    """Run masked Phase A and Phase B without looking at the test split."""
    args = parse_args()
    set_seed(SEED)
    device = get_device()
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    checks = {"stage2": check_stage2_settings(args)}

    # Phase A: ImageNet weights, frozen backbone, head only (Stage 2 Phase A settings).
    phase_a_settings = {**PHASE_A, "max_epochs": args.phase_a_epochs, "patience": args.phase_a_patience}
    loaders_a = get_masked_dataloaders(
        data_root=args.masked_data_root,
        batch_size=phase_a_settings["batch_size"],
        num_workers=args.num_workers,
        cache=not args.no_cache,
    )
    print("Masked data ready:", {name: len(loader.dataset) for name, loader in loaders_a.items()})
    checks["data"] = check_masked_data(loaders_a, args)

    set_seed(SEED)
    model = DenseNet121Classifier(pretrained=True, dropout=args.dropout).to(device)
    model.freeze_backbone()
    phase_a_path = args.checkpoint_dir / f"{MODEL_NAME}_masked_phaseA_best.pth"
    _, phase_a_best, _ = run_phase("A", loaders_a, model, device, args, phase_a_settings, phase_a_path)
    del loaders_a, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase B: all layers, from the best Phase A checkpoint (Stage 2 Phase B settings).
    # Reseed so Phase B's dropout and shuffling do not depend on how long Phase A ran.
    phase_b_settings = {**PHASE_B, "max_epochs": args.phase_b_epochs, "patience": args.phase_b_patience}
    loaders_b = get_masked_dataloaders(
        data_root=args.masked_data_root,
        batch_size=phase_b_settings["batch_size"],
        num_workers=args.num_workers,
        cache=not args.no_cache,
    )
    set_seed(SEED)
    phase_b_model = DenseNet121Classifier(pretrained=False, dropout=args.dropout)
    phase_b_model.load_state_dict(phase_a_best["model_state"])
    phase_b_model.unfreeze_backbone()
    phase_b_model = phase_b_model.to(device)
    phase_b_path = args.checkpoint_dir / f"{MODEL_NAME}_masked_phaseB_best.pth"
    start_from = (f"masked Phase A checkpoint, epoch {phase_a_best['epoch']} "
                  f"(val loss {phase_a_best['val_loss']:.6f})")
    _, phase_b_best, _ = run_phase(
        "B", loaders_b, phase_b_model, device, args, phase_b_settings, phase_b_path, start_from=start_from
    )

    write_summary(phase_a_best, phase_b_best, args, checks)


if __name__ == "__main__":
    main()
