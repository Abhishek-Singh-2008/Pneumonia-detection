"""Stage 1.2: patient-level stratified train/val split and class weights.

Decisions (issue #21, rows 3, 7, 14):
- The official test/ folder is the test set, untouched.
- Pool = official train/ + the 16 official val/ images.
- Val = about 12.5% of the pool, grouped and stratified by class.
- pos_weight = #NORMAL / #PNEUMONIA in the train split only (for BCEWithLogitsLoss).

Grouping (stricter than the patient ID, PR #23 review): the patient_id column keeps
the #21 rule (PNEUMONIA_person{N}_{type}), but the val split groups all pneumonia
images of person{N} together, bacterial and viral. Evidence suggests these are
different children, but this way train and val share no person{N} under either
reading. NORMAL images are grouped by patient_id. See split_group().

Method: every group belongs to exactly one class, so stratifying by class and
grouping can be done class by class. For each class (NORMAL, then PNEUMONIA) the
groups are sorted, shuffled with NumPy's legacy RandomState(42) and moved into
val until val holds VAL_FRACTION of that class's images.
RandomState's stream is frozen by NumPy, so the split is identical on every
library version and OS. (scikit-learn's StratifiedGroupKFold was tried first but
gives different splits on 1.6 and 1.8 with the same seed.)

Run from the repo root:
    python -m src.data.splits            # writes data/splits/{train,val,test}.csv + split_summary.json
"""
import json

import numpy as np
import pandas as pd

from src.config import CLASSES, LABELS, SEED, SPLITS_DIR, VAL_FRACTION, get_data_root
from src.data.index import build_index

SPLIT_COLUMNS = ["image_id", "filepath", "label", "label_name", "patient_id", "official_folder", "split"]


def split_group(patient_ids: pd.Series) -> pd.Series:
    """Group used for the train/val split: PNEUMONIA_person{N}_{type} -> PNEUMONIA_person{N}."""
    return patient_ids.str.replace(r"^(PNEUMONIA_person[0-9]+)_(?:bacteria|virus)$", lambda m: m.group(1), regex=True)


def make_splits(index: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> pd.DataFrame:
    """Return the index with a `split` column (train / val / test)."""
    # Sort by image_id so the result does not depend on filesystem listing order.
    index = index.sort_values("image_id").reset_index(drop=True)
    if index.groupby("patient_id")["label"].nunique().max() > 1:
        raise ValueError("a patient appears in both classes; the class-wise split assumes it never does")

    split = pd.Series("train", index=index.index)
    split[index["official_folder"] == "test"] = "test"
    pool = index[split == "train"]

    groups = split_group(index["patient_id"])
    rng = np.random.RandomState(seed)
    for class_name in CLASSES:
        part = pool[pool["label"] == LABELS[class_name]]
        part_groups = groups[part.index]
        images_per_group = part_groups.value_counts().sort_index()
        order = rng.permutation(len(images_per_group))
        target = round(val_fraction * len(part))

        val_groups, n_val = [], 0
        for i in order:
            if n_val >= target:
                break
            val_groups.append(images_per_group.index[i])
            n_val += int(images_per_group.iloc[i])
        split[part.index[part_groups.isin(val_groups)]] = "val"

    index["split"] = split
    return index[SPLIT_COLUMNS]


def check_splits(df: pd.DataFrame) -> None:
    """Raise if a patient is in two splits, a person{N} is in train and val, or the test set was changed."""
    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} patients in both {a} and {b}, e.g. {sorted(overlap)[:3]}")
    groups = split_group(df["patient_id"])
    overlap = set(groups[df["split"] == "train"]) & set(groups[df["split"] == "val"])
    if overlap:
        raise AssertionError(f"{len(overlap)} split groups in both train and val, e.g. {sorted(overlap)[:3]}")
    if not (df.loc[df["split"] == "test", "official_folder"] == "test").all():
        raise AssertionError("test split contains images from outside the official test folder")
    if (df.loc[df["official_folder"] == "test", "split"] != "test").any():
        raise AssertionError("some official test images are not in the test split")
    if df["image_id"].duplicated().any():
        raise AssertionError("duplicate image_id")


def summarise(df: pd.DataFrame) -> dict:
    """Counts per split and the train-only pos_weight."""
    summary = {"seed": SEED, "val_fraction": VAL_FRACTION,
               "method": "class-wise group shuffle, numpy RandomState",
               "grouping": "PNEUMONIA: person{N} (bacterial and viral together); NORMAL: patient_id",
               "splits": {}}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        summary["splits"][s] = {
            "images": int(len(part)),
            "NORMAL": int((part["label"] == 0).sum()),
            "PNEUMONIA": int((part["label"] == 1).sum()),
            "pct_pneumonia": round(float(100 * part["label"].mean()), 1),
            "patients": int(part["patient_id"].nunique()),
            "patients_NORMAL": int(part.loc[part["label"] == 0, "patient_id"].nunique()),
            "patients_PNEUMONIA": int(part.loc[part["label"] == 1, "patient_id"].nunique()),
        }
    train = summary["splits"]["train"]
    summary["pos_weight"] = round(train["NORMAL"] / train["PNEUMONIA"], 6)
    return summary


def write_splits(df: pd.DataFrame, out_dir=SPLITS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(out_dir / f"{s}.csv", index=False, lineterminator="\n")
    summary = summarise(df)
    with open(out_dir / "split_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    return summary


def load_split(name: str, splits_dir=SPLITS_DIR) -> pd.DataFrame:
    """Read one split CSV (train / val / test)."""
    return pd.read_csv(splits_dir / f"{name}.csv")


def load_pos_weight(splits_dir=SPLITS_DIR) -> float:
    return json.loads((splits_dir / "split_summary.json").read_text())["pos_weight"]


def main() -> None:
    data_root = get_data_root()
    df = make_splits(build_index(data_root))
    check_splits(df)
    summary = write_splits(df)
    print("Data root:", data_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
