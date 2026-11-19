"""Stage 1.2: patient-level, stratified train/val split and the class weight (owned by Abhishek).

Why a new split? The official val/ folder has only 16 images, far too few to choose a
model or an epoch. So we keep the official test/ folder untouched as the test set and
carve a new validation set out of the remaining images.

Decisions (issue #21, rows 3, 7, 14; PR #23 review):
- Test  = the official test/ folder, unchanged (624 images). It is never used to choose anything.
- Pool  = official train/ + the 16 official val/ images (5,232 images).
- Val   = 12.5% of each class of the pool. Train = the rest.
- Patient-level: all images of one patient go to the same side, otherwise the model could
  be validated on a child it was trained on and the val score would be too optimistic.
- Stricter grouping (PR #23, Ajit's review): the patient_id column keeps the #21 rule
  (PNEUMONIA_person{N}_{type}), but for the train/val split all pneumonia images of
  person{N} are kept together, bacterial and viral. The evidence says these are different
  children, but this way train and val share no person{N} under either reading.
  NORMAL images are grouped by patient_id. See split_group().
- pos_weight = #NORMAL / #PNEUMONIA in the TRAIN split only, used by BCEWithLogitsLoss to
  balance the classes (train is 74% pneumonia).

Method: every group belongs to exactly one class, so "stratified by class" and "grouped
by patient" can be done one class at a time. For each class (NORMAL first, then
PNEUMONIA) the groups are sorted, shuffled with NumPy's legacy RandomState(42), and
moved into val one by one until val holds 12.5% of that class's images.
RandomState's random stream is frozen by NumPy, so the split is identical on every
library version and operating system (verified: Kaggle Linux == Windows laptop).
scikit-learn's StratifiedGroupKFold was tried first, but with the same seed it gave a
different split on scikit-learn 1.6 and 1.8.

Run from the repo root:
    python -m src.data.splits     # writes data/splits/{train,val,test}.csv + split_summary.json
The CSVs are committed; every stage loads them with load_split() and never makes new splits.
"""
import json

import numpy as np
import pandas as pd

from src.config import CLASSES, LABELS, SEED, SPLITS_DIR, VAL_FRACTION, get_data_root
from src.data.index import build_index

# Column order of the split CSVs (agreed in #21 row 14).
SPLIT_COLUMNS = ["image_id", "filepath", "label", "label_name", "patient_id", "official_folder", "split"]


def split_group(patient_ids: pd.Series) -> pd.Series:
    """Map each patient_id to the group used for the train/val split.

    PNEUMONIA_person{N}_bacteria and PNEUMONIA_person{N}_virus both become PNEUMONIA_person{N},
    so a person's bacterial and viral images always land on the same side (PR #23).
    NORMAL patient IDs are returned unchanged.

    Args:
        patient_ids: Series of patient_id strings.

    Returns:
        Series of group IDs with the same index.
    """
    # [0-9] instead of \d and a lambda instead of "\1" keep this free of backslash escapes.
    return patient_ids.str.replace(r"^(PNEUMONIA_person[0-9]+)_(?:bacteria|virus)$", lambda m: m.group(1), regex=True)


def make_splits(index: pd.DataFrame, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> pd.DataFrame:
    """Assign every image to train, val or test.

    Args:
        index: output of build_index() (one row per image).
        val_fraction: share of each class of the pool that goes to val (default 0.125).
        seed: seed for the shuffle (default 42).

    Returns:
        DataFrame with the SPLIT_COLUMNS, sorted by image_id, where `split` is
        "train", "val" or "test".
    """
    # Sort by image_id (a plain string) so the result never depends on the order in which
    # the file system listed the files.
    index = index.sort_values("image_id").reset_index(drop=True)
    # The class-by-class method below assumes no patient has images in both classes.
    if index.groupby("patient_id")["label"].nunique().max() > 1:
        raise ValueError("a patient appears in both classes; the class-wise split assumes it never does")

    # Start with: official test folder -> "test", everything else (the pool) -> "train".
    split = pd.Series("train", index=index.index)
    split[index["official_folder"] == "test"] = "test"
    pool = index[split == "train"]

    groups = split_group(index["patient_id"])
    # One RandomState for both classes, used in a fixed order (NORMAL, then PNEUMONIA).
    rng = np.random.RandomState(seed)
    for class_name in CLASSES:
        part = pool[pool["label"] == LABELS[class_name]]          # pool images of this class
        part_groups = groups[part.index]                          # their group IDs
        images_per_group = part_groups.value_counts().sort_index()  # group -> number of images, sorted by name
        order = rng.permutation(len(images_per_group))           # random order of the groups
        target = round(val_fraction * len(part))                 # how many images of this class val should get

        # Walk through the groups in random order and move whole groups to val until the
        # target is reached. The last group can overshoot by a few images; this keeps
        # every patient on one side, which matters more than an exact count.
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
    """Fail loudly if the split breaks any of our rules.

    Checks that:
    - no patient_id appears in two splits;
    - no person{N} group appears in both train and val (stricter rule, PR #23);
    - the test split is exactly the official test folder;
    - every image_id is unique.

    Args:
        df: output of make_splits().

    Raises:
        AssertionError: describing the first rule that is broken.
    """
    patients = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise AssertionError(f"{len(overlap)} patients in both {a} and {b}, e.g. {sorted(overlap)[:3]}")
    # The person{N} check is only between train and val: we cannot change the official
    # test folder, and with person{N}_{type} it already shares no patients with the pool.
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
    """Count images and patients per split and compute the train-only pos_weight.

    Args:
        df: output of make_splits().

    Returns:
        dict that is saved as data/splits/split_summary.json. pos_weight is
        #NORMAL / #PNEUMONIA in train (about 0.347): BCEWithLogitsLoss multiplies the loss
        of pneumonia images by it, so the majority class does not dominate training.
        It is computed from train only, so no information from val or test leaks in (#21 row 7).
    """
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
    """Write train.csv, val.csv, test.csv and split_summary.json to out_dir.

    Files are written with "\\n" line endings on every OS, so a regenerated split is
    byte-for-byte comparable with the committed one (`git diff data/splits` stays empty).

    Args:
        df: output of make_splits().
        out_dir: target folder (default data/splits/).

    Returns:
        The summary dict (see summarise()).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(out_dir / f"{s}.csv", index=False, lineterminator="\n")
    summary = summarise(df)
    with open(out_dir / "split_summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    return summary


def load_split(name: str, splits_dir=SPLITS_DIR) -> pd.DataFrame:
    """Read one committed split CSV.

    Args:
        name: "train", "val" or "test".
        splits_dir: folder with the CSVs (default data/splits/).

    Returns:
        DataFrame with the SPLIT_COLUMNS. Every stage must use this instead of making its own split.
    """
    return pd.read_csv(splits_dir / f"{name}.csv")


def load_pos_weight(splits_dir=SPLITS_DIR) -> float:
    """Return the train-only class weight pos_weight from split_summary.json (about 0.347)."""
    return json.loads((splits_dir / "split_summary.json").read_text())["pos_weight"]


def main() -> None:
    """Build the index from the dataset, make and check the split, write the files, print the summary."""
    data_root = get_data_root()
    df = make_splits(build_index(data_root))
    check_splits(df)   # never write a split that breaks the rules
    summary = write_splits(df)
    print("Data root:", data_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
