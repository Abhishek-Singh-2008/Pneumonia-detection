"""Stage 1.2: patient-level stratified train/val split and class weights.

Decisions (issue #21, rows 3, 7, 14):
- The official test/ folder is the test set, untouched.
- Pool = official train/ + the 16 official val/ images.
- Val = first fold of StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=42),
  i.e. about 12.5% of the pool, grouped by patient and stratified by class.
- pos_weight = #NORMAL / #PNEUMONIA in the train split only (for BCEWithLogitsLoss).

Run from the repo root:
    python -m src.data.splits            # writes data/splits/{train,val,test}.csv + split_summary.json
"""
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED, SPLITS_DIR, VAL_N_SPLITS, get_data_root
from src.data.index import build_index

SPLIT_COLUMNS = ["image_id", "filepath", "label", "label_name", "patient_id", "official_folder", "split"]


def make_splits(index: pd.DataFrame, n_splits: int = VAL_N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """Return the index with a `split` column (train / val / test)."""
    # Sort by image_id so the result does not depend on filesystem listing order.
    index = index.sort_values("image_id").reset_index(drop=True)

    is_test = index["official_folder"] == "test"
    pool = index[~is_test]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_pos = next(sgkf.split(np.zeros(len(pool)), pool["label"], groups=pool["patient_id"]))

    split = pd.Series("test", index=index.index)
    split[pool.index] = "train"
    split[pool.index[val_pos]] = "val"
    index["split"] = split
    return index[SPLIT_COLUMNS]


def check_splits(df: pd.DataFrame) -> None:
    """Raise if a patient is in two splits or the official test set was changed."""
    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} patients in both {a} and {b}, e.g. {sorted(overlap)[:3]}")
    if not (df.loc[df["split"] == "test", "official_folder"] == "test").all():
        raise AssertionError("test split contains images from outside the official test folder")
    if (df.loc[df["official_folder"] == "test", "split"] != "test").any():
        raise AssertionError("some official test images are not in the test split")
    if df["image_id"].duplicated().any():
        raise AssertionError("duplicate image_id")


def summarise(df: pd.DataFrame) -> dict:
    """Counts per split and the train-only pos_weight."""
    summary = {"seed": SEED, "n_splits": VAL_N_SPLITS, "scikit_learn": sklearn.__version__, "splits": {}}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        summary["splits"][s] = {
            "images": int(len(part)),
            "NORMAL": int((part["label"] == 0).sum()),
            "PNEUMONIA": int((part["label"] == 1).sum()),
            "pct_pneumonia": round(100 * part["label"].mean(), 1),
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
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
