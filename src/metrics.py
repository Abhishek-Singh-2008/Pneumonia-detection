"""Metrics shared by every stage (issue #21 row 8). PNEUMONIA (label 1) is the positive class."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import METRICS_COLUMNS, THRESHOLD


def compute_metrics(labels, probs, threshold: float = THRESHOLD) -> dict:
    """labels: 0/1 array; probs: predicted P(PNEUMONIA)."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metrics_row(model: str, data_setting: str, split: str, metrics: dict, attention_in_lung=None) -> dict:
    """One row of results/stageN_metrics.csv, with the agreed columns."""
    row = {"model": model, "data_setting": data_setting, "split": split, **metrics,
           "attention_in_lung": attention_in_lung}
    return {c: row.get(c) for c in METRICS_COLUMNS}


def save_metrics(rows: list, path) -> pd.DataFrame:
    """Write rows to CSV (rounded to 4 decimals) with the agreed column order."""
    df = pd.DataFrame(rows, columns=list(METRICS_COLUMNS))
    for c in ("accuracy", "precision", "recall", "f1", "auc"):
        df[c] = df[c].astype(float).round(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df
