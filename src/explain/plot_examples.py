"""Stage 3.4: figure of the hand-picked shortcut examples (owned by Aman, issue #14).

Issue #14 asks for 6-8 clear examples of shortcut attention. They were picked by eye from the galleries that
src/explain/run_gradcam.py writes (highest_border_*, lowest_attention_*, false_positives_*), and are listed in
EXAMPLES below with what each heatmap shows. This script redraws exactly those images in one figure.

Safety check: the probability, attention_in_lung and border_fraction recomputed here must match
results/stage3/attention_per_image.csv (the full Kaggle run). This proves the same weights and the same image
files were used; tiny differences come from CPU vs GPU arithmetic, hence the small tolerances.
The test set is only explained here, never used to choose anything (#21 row 8).

Needs only the 8 test images, both checkpoints and the lung masks, so it also runs on a CPU:
    python -m src.explain.plot_examples --checkpoint-dir <folder with both .pth> --mask-dir <lung mask folder>

Writes:
    results/stage3/shortcut_examples.png   2 x 4 panels: heatmap over the X-ray with the lung outline
    results/stage3/shortcut_examples.csv   the 8 examples with model, label, probability, metric values and the note
"""
import argparse

import matplotlib

matplotlib.use("Agg")   # files only, no screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import CLASSES, SEED
from src.data.dataset import ChestXrayDataset, build_transforms
from src.data.splits import load_split
from src.explain.attention import attention_in_lung, border_fraction
from src.explain.common import find_checkpoint, load_model
from src.explain.gradcam import GRADCAM_LAYER, grad_cam
from src.explain.run_gradcam import MODEL_LABELS, STAGE3_DIR, _show_cam, default_checkpoint_dir
from src.masking.lung_masks import default_out_dir, load_masks
from src.utils import set_seed

# (image_id, model whose heatmap is shown, what the heatmap shows). Picked by eye from the Stage 3 galleries.
EXAMPLES = [
    ("NORMAL_test_NORMAL2-IM-0252-0001", "densenet121", "false positive: heat on the 'KV59' label and the 'R' marker"),
    ("NORMAL_test_NORMAL2-IM-0221-0001", "densenet121", "false positive: heat on the timestamp (top right)"),
    ("PNEUMONIA_test_person28_virus_62", "densenet121", "correct, but heat on the timestamp (top right)"),
    ("PNEUMONIA_test_person60_virus_117", "densenet121", "correct, but heat on the scanner text block"),
    ("NORMAL_test_NORMAL2-IM-0206-0001", "custom_cnn", "false positive: hot spot on the 'R' marker"),
    ("NORMAL_test_IM-0022-0001", "custom_cnn", "false positive: hot spot on the 'R' marker"),
    ("PNEUMONIA_test_person141_bacteria_677", "custom_cnn", "correct, but heat on the 'R' marker and an electrode"),
    ("PNEUMONIA_test_person118_bacteria_560", "custom_cnn", "correct, but heat on the arm at the image edge"),
]
# Allowed difference between this run and the committed Kaggle run (CPU vs GPU arithmetic).
PROB_TOL, METRIC_TOL = 0.005, 0.02


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", default=default_checkpoint_dir(), help="folder containing both checkpoints")
    p.add_argument("--mask-dir", default=default_out_dir(), help="folder with the Stage 3.2 lung masks")
    p.add_argument("--data-root", default=None, help="dataset folder (default: found automatically, see src.config)")
    return p.parse_args()


def main() -> None:
    """Recompute the heatmaps of the picked examples, check them against the committed run, and save the figure."""
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    picked = pd.DataFrame(EXAMPLES, columns=["image_id", "model", "note"])
    test = load_split("test").set_index("image_id").loc[picked["image_id"]].reset_index()
    # Deterministic val/test pipeline, so the model input is exactly what Stage 3.3 explained.
    dataset = ChestXrayDataset(test, data_root=args.data_root, transform=build_transforms(train=False))
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])        # (8, 3, 224, 224)
    masks = load_masks(args.mask_dir, picked["image_id"])                    # (8, 224, 224) bool
    committed = pd.read_csv(STAGE3_DIR / "attention_per_image.csv").set_index(["image_id", "model"])

    rows, cams = [], [None] * len(picked)
    for model_name in picked["model"].unique():
        model = load_model(model_name, find_checkpoint(model_name, args.checkpoint_dir), device)
        idx = list(picked.index[picked["model"] == model_name])
        out = grad_cam(model, images[idx].to(device), GRADCAM_LAYER[model_name])   # explains the predicted class
        att, bor = attention_in_lung(out["cams"], masks[idx]), border_fraction(out["cams"])
        for k, i in enumerate(idx):
            ref = committed.loc[(picked.at[i, "image_id"], model_name)]
            # Stop if anything differs from the Kaggle run: then the images or weights are not the same.
            if abs(out["probs"][k] - ref["prob_pneumonia"]) > PROB_TOL or abs(att[k] - ref["attention_in_lung"]) > METRIC_TOL \
                    or abs(bor[k] - ref["border_fraction"]) > METRIC_TOL:
                raise RuntimeError(f"{picked.at[i, 'image_id']} ({model_name}) does not match attention_per_image.csv")
            cams[i] = out["cams"][k]
            # Numbers in the CSV are the committed ones, so the report cites one consistent source.
            rows.append({"n": i + 1, "image_id": picked.at[i, "image_id"], "model": model_name,
                         "label_name": CLASSES[int(ref["label"])], "prob_pneumonia": ref["prob_pneumonia"],
                         "pred": int(ref["pred"]), "attention_in_lung": ref["attention_in_lung"],
                         "border_fraction": ref["border_fraction"], "note": picked.at[i, "note"]})
        print(f"{model_name}: {len(idx)} examples match the committed Kaggle run")

    table = pd.DataFrame(rows).sort_values("n")
    table.to_csv(STAGE3_DIR / "shortcut_examples.csv", index=False, lineterminator="\n")

    fig, axes = plt.subplots(2, 4, figsize=(13, 7.6))
    for ax, (_, r) in zip(axes.ravel(), table.iterrows()):
        i = r["n"] - 1
        pred = "PNEUMONIA" if r["pred"] == 1 else "NORMAL"
        title = (f"({r['n']}) {MODEL_LABELS[r['model']]}: true {r['label_name']}\npred {pred} p={r['prob_pneumonia']:.2f} | "
                 f"in-lung {r['attention_in_lung']:.2f} | border {r['border_fraction']:.2f}")
        _show_cam(ax, images[i], cams[i], masks[i], title)
        ax.text(0.5, -0.04, r["note"], transform=ax.transAxes, ha="center", va="top", fontsize=7.5, wrap=True)
    fig.suptitle("Shortcut examples (Stage 3.4): Grad-CAM of the predicted class; green = lung mask")
    plt.tight_layout()
    plt.savefig(STAGE3_DIR / "shortcut_examples.png", dpi=130)
    plt.close(fig)
    print(table.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
