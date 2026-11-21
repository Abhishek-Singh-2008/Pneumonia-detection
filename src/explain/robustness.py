"""Stage 3.3 robustness table: do the lung-attention results depend on the flagged (doubtful) lung masks?
(owned by Aman, issue #13)

The lung segmentation model misses consolidated lung on some pneumonia images (Stage 3.2). Such masks are
flagged in results/stage3/lung_mask_summary.csv. If they distorted the metric, leaving them out would change
the averages. This script recomputes the per-model, per-class averages with ALL test images and WITHOUT the
flagged ones, and adds the share of images whose heatmap is more on the lungs than chance (lift > 1).

It only reads committed CSVs (no images, no model, no GPU), so it runs anywhere in a second:
    python -m src.explain.robustness

Reads:  results/stage3/attention_per_image.csv, results/stage3/lung_mask_summary.csv
Writes: results/stage3/attention_robustness.csv
    columns: model, label_name ("all", "NORMAL", "PNEUMONIA"), masks ("all" or "unflagged"), n,
             attention_mean, lung_area_mean, lift_mean, border_mean,
             share_lift_above_1 (share of images with attention_in_lung > lung_area)
"""
import pandas as pd

from src.config import RESULTS_DIR

STAGE3_DIR = RESULTS_DIR / "stage3"


def robustness_table(per_image: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """Average the attention metric with and without the flagged lung masks.

    Args:
        per_image: attention_per_image.csv (one row per model and test image).
        masks: lung_mask_summary.csv (one row per image, with a boolean "flag" column).

    Returns:
        DataFrame with the columns listed in the module docstring, rounded to 4 decimals.
    """
    d = per_image.merge(masks[["image_id", "flag"]], on="image_id", how="left", validate="m:1")
    rows = []
    for mask_set, part in (("all", d), ("unflagged", d[~d["flag"]])):
        for model, m in part.groupby("model", sort=False):
            # "all" first, then each class, so the table reads top-down like the report table.
            for label, x in [("all", m)] + list(m.groupby("label_name")):
                rows.append({"model": model, "label_name": label, "masks": mask_set, "n": len(x),
                             "attention_mean": x["attention_in_lung"].mean(), "lung_area_mean": x["lung_area"].mean(),
                             "lift_mean": x["lift"].mean(), "border_mean": x["border_fraction"].mean(),
                             "share_lift_above_1": (x["lift"] > 1).mean()})
    return pd.DataFrame(rows).round(4)


def main() -> None:
    """Read the committed Stage 3 tables, build the robustness table, save and print it."""
    table = robustness_table(pd.read_csv(STAGE3_DIR / "attention_per_image.csv"),
                             pd.read_csv(STAGE3_DIR / "lung_mask_summary.csv"))
    table.to_csv(STAGE3_DIR / "attention_robustness.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
