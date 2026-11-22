"""Stage 4.2/4.3: masked test evaluation, cross-test and Grad-CAM attention.

Run after Stage 4.1 with the masked copies, lung masks and both DenseNet
checkpoints available. The final masked-model test predictions also drive the
Grad-CAM attention metric, so that model's test split is traversed only once.

Outputs are written under results/: the agreed masked test metrics, a separate
unmasked-trained-on-masked cross-test file, per-image predictions/attention,
and the final three-row comparison table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CLASSES, METRICS_COLUMNS, REPO_ROOT, RESULTS_DIR, SEED, THRESHOLD
from src.data.splits import load_split
from src.explain.attention import attention_in_lung, border_fraction, lung_area_fraction
from src.explain.common import find_checkpoint, load_model
from src.explain.gradcam import GRADCAM_LAYER, grad_cam_loader
from src.masking.lung_masks import default_out_dir, load_masks
from src.masking.masked_dataset import get_masked_dataloaders
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.train import predict
from src.utils import set_seed

STAGE4_DIR = RESULTS_DIR / "stage4_masked_densenet"
STAGE4_METRICS = RESULTS_DIR / "stage4_metrics.csv"
COMPARISON_PATH = RESULTS_DIR / "stage4_comparison.csv"
CROSS_TEST_PATH = RESULTS_DIR / "stage4_cross_test_metrics.csv"
MASKED_CHECKPOINT_NAME = "densenet121_masked_phaseB_best.pth"


def _find_checkpoint(filename: str, root: Path) -> Path:
    """Find a named checkpoint under a root, failing clearly when it is absent."""
    hits = sorted(Path(root).rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return hits[0]


def _validate_checkpoint_metadata(path: Path, training_data: str) -> None:
    """Reject checkpoints whose saved metadata does not identify the selected Stage 2/4 model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("extra"), dict):
        raise ValueError(f"{path} does not contain the expected checkpoint metadata")
    extra = checkpoint["extra"]
    if extra.get("model") != "densenet121" or extra.get("phase") != "B (all layers fine-tuned)":
        raise ValueError(f"{path} is not the final DenseNet-121 Phase B checkpoint")
    if training_data == "masked":
        valid = extra.get("data_setting") == "masked"
    else:
        valid = extra.get("run") == "lr1e-4_bs16" and extra.get("batch_size") == 16
    if not valid:
        raise ValueError(f"{path} metadata does not match the selected {training_data} training run")


def _assert_aligned(result: dict, test_df: pd.DataFrame, model_name: str) -> None:
    """Fail if predictions no longer correspond exactly to the committed test split."""
    expected_ids = test_df["image_id"].astype(str).tolist()
    expected_labels = test_df["label"].to_numpy(dtype=int)
    if result["image_ids"] != expected_ids:
        raise RuntimeError(f"{model_name}: test predictions are not in committed split order")
    if not np.array_equal(np.asarray(result["labels"], dtype=int), expected_labels):
        raise RuntimeError(f"{model_name}: test labels do not match the committed split")


def _cross_test_row(metrics: dict) -> dict:
    """Label the cross-test distinctly from the matched masked-model metrics."""
    return {
        "model": "densenet121",
        "training_data": "unmasked",
        "evaluation_data": "masked",
        "split": "test",
        **metrics,
    }


def build_comparison_table(
    stage1: pd.DataFrame,
    stage3: pd.DataFrame,
    stage4: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final baseline, unmasked DenseNet and masked DenseNet test table."""
    custom = stage1.query("model == 'custom_cnn' and split == 'test'")
    dense_unmasked = stage3.query("model == 'densenet121' and split == 'test'")
    dense_masked = stage4.query("model == 'densenet121' and data_setting == 'masked' and split == 'test'")
    if len(custom) != 1 or len(dense_unmasked) != 1 or len(dense_masked) != 1:
        raise ValueError("Expected exactly one test-metric row for each final comparison model/setting")
    if pd.isna(dense_unmasked.iloc[0]["attention_in_lung"]):
        raise ValueError("Stage 3 DenseNet test row is missing its attention_in_lung metric")
    custom_attention = stage3.query("model == 'custom_cnn' and split == 'test'")
    if len(custom_attention) != 1 or pd.isna(custom_attention.iloc[0]["attention_in_lung"]):
        raise ValueError("Stage 3 custom CNN test row is missing its attention_in_lung metric")
    custom = custom.copy()
    custom.loc[:, "attention_in_lung"] = custom_attention.iloc[0]["attention_in_lung"]
    return pd.concat([custom, dense_unmasked, dense_masked], ignore_index=True).loc[:, list(METRICS_COLUMNS)]


def _attention_summary(attention: np.ndarray, areas: np.ndarray, borders: np.ndarray) -> dict:
    """Summarize attention while keeping all-zero heatmaps out of means."""
    valid = np.isfinite(attention)
    return {
        "n_test_images": int(len(attention)),
        "n_zero_heatmaps": int((~valid).sum()),
        "attention_in_lung_mean": float(attention[valid].mean()) if valid.any() else None,
        "attention_in_lung_median": float(np.median(attention[valid])) if valid.any() else None,
        "lung_area_mean": float(np.mean(areas)),
        "lift_mean": float(np.mean(attention[valid] / areas[valid])) if valid.any() else None,
        "border_fraction_mean": float(np.nanmean(borders)) if np.isfinite(borders).any() else None,
    }


def _default_data_root() -> Path:
    """Use the Stage 4 Kaggle output path, falling back to the documented local path."""
    kaggle_root = Path("/kaggle/temp/stage4/masked_chest_xray")
    if kaggle_root.is_dir():
        return kaggle_root
    return REPO_ROOT / "data" / "masked_chest_xray"


def _default_checkpoint_root() -> Path:
    """Search the Kaggle working tree for both checkpoints, otherwise the repository."""
    kaggle_root = Path("/kaggle/working")
    return kaggle_root if kaggle_root.is_dir() else REPO_ROOT


def _default_mask_dir() -> Path:
    """Use Stage 4's temporary Kaggle masks when present, otherwise the standard mask path."""
    kaggle_mask_dir = Path("/kaggle/temp/stage4/lung_masks")
    return kaggle_mask_dir if kaggle_mask_dir.is_dir() else default_out_dir()


def parse_args() -> argparse.Namespace:
    """Parse input paths and evaluation batch settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=_default_data_root(),
                        help="masked copies created by Stage 4.1")
    parser.add_argument("--mask-dir", type=Path, default=_default_mask_dir(),
                        help="Stage 3 lung masks, in image_id.png format")
    parser.add_argument("--checkpoint-root", type=Path, default=_default_checkpoint_root(),
                        help="root containing both final DenseNet checkpoint files")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2, help="use 0 if worker processes are unavailable")
    return parser.parse_args()


def main() -> None:
    """Run masked evaluation, unmasked-model cross-test and masked-model Grad-CAM."""
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)

    masked_checkpoint = _find_checkpoint(MASKED_CHECKPOINT_NAME, args.checkpoint_root)
    unmasked_checkpoint = find_checkpoint("densenet121", args.checkpoint_root)
    _validate_checkpoint_metadata(masked_checkpoint, "masked")
    _validate_checkpoint_metadata(unmasked_checkpoint, "unmasked")
    test_df = load_split("test")
    loaders = get_masked_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache=False,
    )
    test_loader = loaders["test"]
    expected_ids = test_df["image_id"].astype(str).tolist()
    masks = load_masks(args.mask_dir, expected_ids)
    areas = np.asarray(lung_area_fraction(masks), dtype=float)
    print(f"Device: {device} | masked test images: {len(expected_ids)}")
    print(f"Masked checkpoint: {masked_checkpoint}")
    print(f"Unmasked checkpoint: {unmasked_checkpoint}")

    # Grad-CAM returns the masked model's predictions too, avoiding a second test pass.
    masked_model = load_model("densenet121", masked_checkpoint, device)
    masked_out = grad_cam_loader(masked_model, test_loader, GRADCAM_LAYER["densenet121"], device)
    _assert_aligned(masked_out, test_df, "masked DenseNet-121")
    attention = np.asarray(attention_in_lung(masked_out["cams"], masks), dtype=float)
    borders = np.asarray(border_fraction(masked_out["cams"]), dtype=float)
    masked_metrics = compute_metrics(masked_out["labels"], masked_out["probs"])
    attention_mean = _attention_summary(attention, areas, borders)["attention_in_lung_mean"]
    masked_row = metrics_row(
        "densenet121", "masked", "test", masked_metrics,
        attention_in_lung=round(attention_mean, 4) if attention_mean is not None else None,
    )
    print(save_metrics([masked_row], STAGE4_METRICS).to_string(index=False))

    masked_predictions = pd.DataFrame({
        "image_id": masked_out["image_ids"],
        "label": masked_out["labels"],
        "label_name": [CLASSES[int(label)] for label in masked_out["labels"]],
        "prob_pneumonia": masked_out["probs"],
        "pred": masked_out["preds"],
        "attention_in_lung": attention,
        "lung_area": areas,
        "lift": attention / areas,
        "border_fraction": borders,
    })
    # Preserve probabilities at float precision so probability ties introduced by
    # display rounding do not make the committed AUC impossible to reproduce.
    masked_predictions.to_csv(
        STAGE4_DIR / "masked_test_predictions_attention.csv", index=False,
        lineterminator="\n", float_format="%.12g",
    )
    attention_summary = _attention_summary(attention, areas, borders)
    (STAGE4_DIR / "masked_attention_summary.json").write_text(
        json.dumps(attention_summary, indent=2) + "\n", encoding="utf-8"
    )
    del masked_model, masked_out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Cross-test the selected Stage 2 checkpoint on the same masked test images.
    unmasked_model = load_model("densenet121", unmasked_checkpoint, device)
    cross_out = predict(unmasked_model, test_loader, device)
    _assert_aligned(cross_out, test_df, "unmasked DenseNet-121 cross-test")
    cross_metrics = compute_metrics(cross_out["labels"], cross_out["probs"])
    cross_row = _cross_test_row(cross_metrics)
    cross_frame = pd.DataFrame([cross_row])
    for column in ("accuracy", "precision", "recall", "f1", "auc"):
        cross_frame[column] = cross_frame[column].astype(float).round(4)
    cross_frame.to_csv(CROSS_TEST_PATH, index=False, lineterminator="\n")
    cross_predictions = pd.DataFrame({
        "image_id": cross_out["image_ids"],
        "label": cross_out["labels"],
        "label_name": [CLASSES[int(label)] for label in cross_out["labels"]],
        "prob_pneumonia": cross_out["probs"],
        "pred": (cross_out["probs"] >= THRESHOLD).astype(int),
    })
    cross_predictions.to_csv(
        STAGE4_DIR / "unmasked_model_on_masked_test_predictions.csv", index=False,
        lineterminator="\n", float_format="%.12g",
    )
    del unmasked_model, cross_out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    comparison = build_comparison_table(
        pd.read_csv(RESULTS_DIR / "stage1_metrics.csv"),
        pd.read_csv(RESULTS_DIR / "stage3_metrics.csv"),
        pd.read_csv(STAGE4_METRICS),
    )
    comparison.to_csv(COMPARISON_PATH, index=False, lineterminator="\n")
    print("\nUnmasked DenseNet-121 evaluated on masked test images:")
    print(cross_frame.to_string(index=False))
    print("\nFinal matched-model comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved Stage 4.2/4.3 outputs under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
