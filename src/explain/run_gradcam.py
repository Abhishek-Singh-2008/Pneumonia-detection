"""Stage 3.3 (and the figures for 3.4): Grad-CAM heatmaps and the lung-attention metric for both unmasked models
(owned by Aman, issues #13 and #14).

Steps:
1. Load the final custom CNN (Stage 1) and DenseNet-121 (Stage 2) from their checkpoints.
2. Run Grad-CAM (src/explain/gradcam.py) on EVERY test image (624) for both models, explaining the predicted
   class (+logit for PNEUMONIA, -logit for NORMAL; #21). The guide asks for heatmaps on a fixed sample; the
   metric is computed on the full test set as well, because the per-group averages (e.g. "wrong NORMAL
   predictions") need more images than a 50-image sample contains. Figures use the fixed sample.
3. Safety check: the test metrics recomputed here must equal the committed Stage 1 / Stage 2 numbers, which
   proves the right weights were loaded. The test set is only EVALUATED here, nothing is chosen on it
   (#21 row 8): models, threshold and layers were all fixed before this script runs.
4. Load the lung masks (Stage 3.2) and compute, per image: attention_in_lung, lung_area (its chance level),
   lift = attention / area, and border_fraction (src/explain/attention.py).
5. Save the tables, the fixed sample, and the figures.

The fixed sample (issue #11): 50 test images, 25 NORMAL + 25 PNEUMONIA, the SAME images for both models.
Per class it contains up to 8 images that at least one model got wrong (the committed test predictions of
Stages 1 and 2 are used to find them, so the sample is fixed before any heatmap is made), and the rest are
images both models got right, drawn with RandomState(42).

Run from the repo root (GPU recommended; the masks of Stage 3.2 and both checkpoints must be available):
    python -m src.explain.run_gradcam --checkpoint-dir /kaggle/working/checkpoints --mask-dir /kaggle/working/lung_masks
    python -m src.explain.run_gradcam --help

Writes (results/stage3/):
    attention_per_image.csv     one row per (model, test image): probability, prediction, metric values, in_sample
    attention_summary.csv       mean/median per model and group (all, per class, correct vs wrong, class x correct)
    sample_images.csv           the fixed 50-image sample and why each image is in it
    heatmaps_sample_p{1..5}.png the sample: X-ray with lung outline | custom CNN heatmap | DenseNet-121 heatmap
    lowest_attention_{model}.png      12 test images with the least attention on the lungs (shortcut candidates)
    highest_border_{model}.png        12 test images with the most attention on the outer border (markers, edges)
    false_positives_{model}.png       12 most confident NORMAL -> PNEUMONIA mistakes
    attention_boxplot.png       attention_in_lung per model, class and correct/wrong, with the lung-area level
    settings.json               layers, target rule, files used, counts
results/stage3_metrics.csv      test metrics of both unmasked models with attention_in_lung filled in (#21 row 8)
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # write files only; no screen needed (Kaggle, servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CLASSES, IMAGE_SIZE, REPO_ROOT, RESULTS_DIR, SEED, THRESHOLD
from src.data.dataset import ChestXrayDataset, build_transforms, denormalize
from src.data.splits import load_split
from src.explain.attention import attention_in_lung, border_fraction, lung_area_fraction
from src.explain.common import MODELS, find_checkpoint, load_model
from src.explain.gradcam import GRADCAM_LAYER, grad_cam_loader
from src.masking.lung_masks import SEG_MODEL_NAME, default_out_dir, load_masks
from src.metrics import compute_metrics, metrics_row, save_metrics
from src.utils import set_seed

STAGE3_DIR = RESULTS_DIR / "stage3"
# Committed test predictions of the earlier stages, used to (a) choose the sample and (b) check the weights.
COMMITTED_PREDICTIONS = {
    "custom_cnn": RESULTS_DIR / "stage1_baseline" / "test_predictions.csv",
    "densenet121": RESULTS_DIR / "stage2_densenet" / "test_predictions.csv",
}
COMMITTED_METRICS = {"custom_cnn": RESULTS_DIR / "stage1_metrics.csv", "densenet121": RESULTS_DIR / "stage2_metrics.csv"}
MODEL_LABELS = {"custom_cnn": "Custom CNN", "densenet121": "DenseNet-121"}
N_SAMPLE = 50              # fixed sample size (issue #11)
MAX_WRONG_PER_CLASS = 8    # at most this many misclassified images per class in the sample


def default_checkpoint_dir() -> Path:
    """Same rule as the training scripts: /kaggle/working/checkpoints on Kaggle, else checkpoints/ (git-ignored)."""
    return Path("/kaggle/working/checkpoints") if os.path.isdir("/kaggle/working") else REPO_ROOT / "checkpoints"


def select_sample(test_df: pd.DataFrame, n: int = N_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """Choose the fixed sample of test images used for the heatmap figures (issue #11).

    Args:
        test_df: the test split (src.data.splits.load_split("test")).
        n: sample size, split equally between NORMAL and PNEUMONIA.
        seed: seed of the NumPy RandomState (version-independent, like the split in Stage 1.2).

    Returns:
        DataFrame with columns image_id, label, label_name, pred_custom_cnn, pred_densenet121, reason,
        sorted by label then image_id. reason is "wrong_both", "wrong_custom_cnn", "wrong_densenet121"
        or "correct_both".
    """
    df = test_df[["image_id", "label", "label_name"]].copy()
    for m in MODELS:
        pred = pd.read_csv(COMMITTED_PREDICTIONS[m])[["image_id", "pred"]].rename(columns={"pred": f"pred_{m}"})
        df = df.merge(pred, on="image_id", how="left", validate="1:1")
    wrong = {m: df[f"pred_{m}"] != df["label"] for m in MODELS}
    df["reason"] = np.select([wrong["custom_cnn"] & wrong["densenet121"], wrong["custom_cnn"], wrong["densenet121"]],
                             ["wrong_both", "wrong_custom_cnn", "wrong_densenet121"], default="correct_both")
    rng = np.random.RandomState(seed)
    picked = []
    for class_name in CLASSES:   # NORMAL first, then PNEUMONIA: fixed order, so the RNG stream is reproducible
        part = df[df["label_name"] == class_name].sort_values("image_id")
        bad, good = part[part["reason"] != "correct_both"], part[part["reason"] == "correct_both"]
        n_bad = min(len(bad), MAX_WRONG_PER_CLASS)
        picked.append(bad.iloc[rng.permutation(len(bad))[:n_bad]])
        picked.append(good.iloc[rng.permutation(len(good))[:n // 2 - n_bad]])
    return pd.concat(picked).sort_values(["label", "image_id"]).reset_index(drop=True)


def check_against_committed(model_name: str, labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compare the test metrics recomputed here with the committed Stage 1/2 numbers.

    Args:
        model_name: "custom_cnn" or "densenet121".
        labels, probs: test labels and P(pneumonia) from this run.

    Returns:
        The recomputed metrics dict (src.metrics.compute_metrics).

    Raises:
        RuntimeError: if accuracy or AUC differ by more than 0.005, which means the wrong checkpoint was loaded
        (tiny differences can come from GPU arithmetic, which is why exact equality is not required).
    """
    m = compute_metrics(labels, probs)
    ref = pd.read_csv(COMMITTED_METRICS[model_name]).query("split == 'test'").iloc[0]
    for k in ("accuracy", "auc"):
        if abs(m[k] - ref[k]) > 0.005:
            raise RuntimeError(f"{model_name}: test {k} {m[k]:.4f} != committed {ref[k]:.4f}. Wrong checkpoint?")
    print(f"  {model_name}: test accuracy {m['accuracy']:.4f} / AUC {m['auc']:.4f} match the committed results")
    return m


def summarise(per_image: pd.DataFrame) -> pd.DataFrame:
    """Average the metric per model and group of images.

    Args:
        per_image: the attention_per_image table.

    Returns:
        DataFrame with columns model, group, n, attention_mean, attention_median, lung_area_mean, lift_mean,
        border_mean, n_zero_heatmap. Groups: all; NORMAL; PNEUMONIA; correct; wrong; and the four class x
        correct combinations. Means ignore all-zero heatmaps (NaN metric), which are counted in n_zero_heatmap.
    """
    rows = []
    for model, d in per_image.groupby("model", sort=False):
        groups = {"all": d}
        for c in CLASSES:
            groups[c] = d[d["label_name"] == c]
        for ok, name in ((True, "correct"), (False, "wrong")):
            groups[name] = d[d["correct"] == ok]
            for c in CLASSES:
                groups[f"{c}_{name}"] = d[(d["label_name"] == c) & (d["correct"] == ok)]
        for g, x in groups.items():
            rows.append({"model": model, "group": g, "n": len(x),
                         "attention_mean": x["attention_in_lung"].mean(), "attention_median": x["attention_in_lung"].median(),
                         "lung_area_mean": x["lung_area"].mean(), "lift_mean": x["lift"].mean(),
                         "border_mean": x["border_fraction"].mean(), "n_zero_heatmap": int(x["attention_in_lung"].isna().sum())})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _show_xray(ax, image: torch.Tensor, mask: np.ndarray) -> None:
    """Draw one normalised image tensor (3, 224, 224) in gray with the lung outline in green."""
    ax.imshow(denormalize(image)[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.7)
    ax.axis("off")


def _show_cam(ax, image: torch.Tensor, cam: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Draw a Grad-CAM heatmap (jet colours, 50% transparent) over the X-ray, with the lung outline."""
    _show_xray(ax, image, mask)
    ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)


def _row_title(r) -> str:
    """Panel title: model prediction, probability and metric values of one per-image row."""
    pred = "PNEU" if r["pred"] == 1 else "NORM"
    return f"pred {pred} p={r['prob_pneumonia']:.2f}\nin-lung {r['attention_in_lung']:.2f} | border {r['border_fraction']:.2f}"


def plot_sample_pages(sample, dataset, index_of, cams, masks, per_image, out_dir: Path, rows_per_page: int = 10):
    """Save the fixed sample as pages: X-ray + lung outline | custom CNN heatmap | DenseNet-121 heatmap.

    Args:
        sample: select_sample() output.
        dataset: test ChestXrayDataset (deterministic transform), to get the exact model inputs.
        index_of: dict image_id -> position in the test set.
        cams: dict model -> (N, 224, 224) heatmaps in test-set order.
        masks: (N, 224, 224) bool lung masks in test-set order.
        per_image: attention_per_image table (indexed by model and image_id).
        out_dir: folder for heatmaps_sample_p{k}.png.
    """
    for page, start in enumerate(range(0, len(sample), rows_per_page), start=1):
        part = sample.iloc[start:start + rows_per_page]
        fig, axes = plt.subplots(len(part), 3, figsize=(7.5, 2.6 * len(part)), squeeze=False)
        for row_axes, (_, s) in zip(axes, part.iterrows()):
            i = index_of[s["image_id"]]
            image = dataset[i][0]
            _show_xray(row_axes[0], image, masks[i])
            row_axes[0].set_title(f"{s['image_id']}\ntrue {s['label_name']} ({s['reason']})", fontsize=7)
            for ax, m in zip(row_axes[1:], MODELS):
                r = per_image.loc[(m, s["image_id"])]
                _show_cam(ax, image, cams[m][i], masks[i], f"{MODEL_LABELS[m]}: {_row_title(r)}")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmaps_sample_p{page}.png", dpi=110)
        plt.close(fig)


def plot_gallery(rows: pd.DataFrame, model: str, dataset, index_of, cams, masks, path: Path, title: str) -> None:
    """Save a 4-column gallery of up to 12 images with one model's heatmaps (shortcut candidates, mistakes)."""
    rows = rows.head(12)
    if rows.empty:
        return
    ncols = 4
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        i = index_of[r["image_id"]]
        _show_cam(ax, dataset[i][0], cams[model][i], masks[i], f"{r['image_id']} (true {r['label_name']})\n{_row_title(r)}")
    fig.suptitle(f"{MODEL_LABELS[model]}: {title}")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def plot_boxplot(per_image: pd.DataFrame, path: Path) -> None:
    """Box plot of attention_in_lung per model and class x correct, with each group's mean lung area as a dash."""
    groups = [(c, ok) for c in CLASSES for ok in (True, False)]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), MODELS):
        d = per_image[per_image["model"] == m]
        data, labels, areas = [], [], []
        for c, ok in groups:
            x = d[(d["label_name"] == c) & (d["correct"] == ok)]
            data.append(x["attention_in_lung"].dropna().values)
            labels.append(f"{c}\n{'correct' if ok else 'wrong'}\n(n={len(x)})")
            areas.append(x["lung_area"].mean())
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        # Chance level: what an evenly spread heatmap would score on these images.
        ax.scatter(range(1, len(areas) + 1), areas, marker="_", s=600, color="red", label="lung area (chance level)")
        ax.set_title(MODEL_LABELS[m])
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("attention_in_lung")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir(),
                   help="folder containing custom_cnn_best.pth and the DenseNet-121 Phase B checkpoint (searched recursively)")
    p.add_argument("--mask-dir", type=Path, default=default_out_dir(), help="folder with the Stage 3.2 lung masks")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2, help="use 0 on Windows if workers fail")
    return p.parse_args()


def main() -> None:
    """Run Grad-CAM and the attention metric for both models on the test set and save every output."""
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)

    # Test images only, with the deterministic val/test pipeline (no augmentation), in CSV order.
    test_df = load_split("test")
    dataset = ChestXrayDataset(test_df, transform=build_transforms(train=False), cache=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    index_of = {img_id: i for i, img_id in enumerate(test_df["image_id"])}
    print(f"Device: {device} | test images: {len(dataset)}")

    # Lung masks of the test images, in the same order (Stage 3.2).
    masks = load_masks(args.mask_dir, test_df["image_id"])
    areas = lung_area_fraction(masks)

    sample = select_sample(test_df)
    sample.to_csv(STAGE3_DIR / "sample_images.csv", index=False, lineterminator="\n")
    in_sample = set(sample["image_id"])

    cams, tables, metric_rows, checkpoints = {}, [], [], {}
    for m in MODELS:
        ckpt = find_checkpoint(m, args.checkpoint_dir)
        checkpoints[m] = ckpt.name
        model = load_model(m, ckpt, device)
        out = grad_cam_loader(model, loader, GRADCAM_LAYER[m], device)   # explains the predicted class
        assert out["image_ids"] == list(test_df["image_id"]), "loader order must follow the test CSV"
        metrics = check_against_committed(m, out["labels"], out["probs"])
        cams[m] = out["cams"]
        att = attention_in_lung(out["cams"], masks)
        tables.append(pd.DataFrame({
            "model": m, "image_id": out["image_ids"], "label": out["labels"],
            "label_name": [CLASSES[y] for y in out["labels"]], "prob_pneumonia": out["probs"].round(6),
            "pred": out["preds"], "correct": out["preds"] == out["labels"],
            "attention_in_lung": att, "lung_area": areas, "lift": att / areas,
            "border_fraction": border_fraction(out["cams"]),
            "in_sample": [i in in_sample for i in out["image_ids"]],
        }))
        # attention_in_lung in the metrics CSV = mean over all test images with a non-zero heatmap.
        metric_rows.append(metrics_row(m, "unmasked", "test", metrics, attention_in_lung=round(float(np.nanmean(att)), 4)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_image = pd.concat(tables, ignore_index=True)
    per_image.round(4).to_csv(STAGE3_DIR / "attention_per_image.csv", index=False, lineterminator="\n")
    summary = summarise(per_image)
    summary.to_csv(STAGE3_DIR / "attention_summary.csv", index=False, lineterminator="\n")
    print(save_metrics(metric_rows, RESULTS_DIR / "stage3_metrics.csv").to_string(index=False))
    print(summary[summary["group"].isin(["all", "NORMAL", "PNEUMONIA", "correct", "wrong"])].to_string(index=False))

    # Figures. per_image indexed by (model, image_id) for quick lookups.
    lookup = per_image.set_index(["model", "image_id"])
    plot_sample_pages(sample, dataset, index_of, cams, masks, lookup, STAGE3_DIR)
    for m in MODELS:
        d = per_image[per_image["model"] == m]
        plot_gallery(d.sort_values("attention_in_lung"), m, dataset, index_of, cams, masks,
                     STAGE3_DIR / f"lowest_attention_{m}.png", "least attention on the lungs (shortcut candidates)")
        plot_gallery(d.sort_values("border_fraction", ascending=False), m, dataset, index_of, cams, masks,
                     STAGE3_DIR / f"highest_border_{m}.png", "most attention on the outer 10% border")
        fp = d[(d["label"] == 0) & (d["pred"] == 1)].sort_values("prob_pneumonia", ascending=False)
        plot_gallery(fp, m, dataset, index_of, cams, masks, STAGE3_DIR / f"false_positives_{m}.png",
                     "most confident NORMAL images predicted as PNEUMONIA")
    plot_boxplot(per_image, STAGE3_DIR / "attention_boxplot.png")

    settings = {
        "gradcam_layers": GRADCAM_LAYER, "target": "predicted class (+logit if PNEUMONIA, -logit if NORMAL)",
        "threshold": THRESHOLD, "heatmap_size": IMAGE_SIZE, "upsampling": "bilinear",
        "border_band": "outer 10% on each side", "checkpoints": checkpoints, "mask_model": SEG_MODEL_NAME,
        "n_test_images": len(test_df), "n_sample": len(sample), "seed": SEED,
        "n_zero_heatmaps": {m: int(per_image.query("model == @m")["attention_in_lung"].isna().sum()) for m in MODELS},
    }
    (STAGE3_DIR / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print("Saved everything to", STAGE3_DIR)


if __name__ == "__main__":
    main()
