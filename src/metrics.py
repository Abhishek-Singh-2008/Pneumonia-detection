"""Evaluation metrics shared by every stage (owned by Stage 1, Abhishek).

All stages must report results in exactly the same way so Stage 4 can put them in one
table (issue #21 row 8). Use these functions instead of writing your own:

    m = compute_metrics(labels, probs)                          # dict of metrics
    row = metrics_row("densenet121", "unmasked", "test", m)     # one CSV row with the agreed columns
    save_metrics([row_val, row_test], "results/stage2_metrics.csv")

PNEUMONIA (label 1) is the positive class: precision, recall and F1 are for pneumonia,
and AUC uses P(pneumonia). The decision threshold is src.config.THRESHOLD (0.5).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import METRICS_COLUMNS, THRESHOLD


def compute_metrics(labels, probs, threshold: float = THRESHOLD) -> dict:
    """Compute the agreed metrics for one model on one split.

    Args:
        labels: true labels, array-like of 0 (NORMAL) / 1 (PNEUMONIA).
        probs: predicted probabilities of PNEUMONIA (sigmoid of the logit), same length.
        threshold: probability at or above which an image counts as PNEUMONIA (default 0.5;
            never choose it on the test set).

    Returns:
        dict with
            accuracy   share of correct predictions
            precision  of the images predicted PNEUMONIA, the share that really are (TP / (TP + FP))
            recall     of the real PNEUMONIA images, the share that were found (TP / (TP + FN)); also called sensitivity
            f1         harmonic mean of precision and recall
            auc        area under the ROC curve; threshold-free (uses probs directly); NaN if only one class is present
            tn, fp, fn, tp   confusion-matrix counts (ints)
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    # labels=[0, 1] fixes the matrix layout even if a class is missing from this split.
    # ravel() order for binary labels is tn, fp, fn, tp.
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        # zero_division=0: report 0 instead of a warning when nothing is predicted positive.
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_row(model: str, data_setting: str, split: str, metrics: dict, attention_in_lung=None) -> dict:
    """Build one row of results/stageN_metrics.csv with the agreed columns, in order.

    Args:
        model: e.g. "custom_cnn" or "densenet121".
        data_setting: "unmasked" or "masked" (Stage 4).
        split: "val" or "test".
        metrics: output of compute_metrics().
        attention_in_lung: Grad-CAM lung-attention metric (Stage 3/4); leave None before that.

    Returns:
        dict with exactly the keys of src.config.METRICS_COLUMNS.
    """
    row = {"model": model, "data_setting": data_setting, "split": split, **metrics,
           "attention_in_lung": attention_in_lung}
    return {c: row.get(c) for c in METRICS_COLUMNS}


def save_metrics(rows: list, path) -> pd.DataFrame:
    """Write metric rows to a CSV with the agreed column order.

    Accuracy, precision, recall, F1 and AUC are rounded to 4 decimals, the precision used
    in the report. Line endings are "\\n" on every OS.

    Args:
        rows: list of dicts from metrics_row().
        path: output CSV, e.g. results/stage1_metrics.csv (parent folders are created).

    Returns:
        The DataFrame that was written.
    """
    df = pd.DataFrame(rows, columns=list(METRICS_COLUMNS))
    for c in ("accuracy", "precision", "recall", "f1", "auc"):
        df[c] = df[c].astype(float).round(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df
