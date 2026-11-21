"""Shared settings for every stage of the project (owned by Stage 1, Abhishek).

Every value that more than one file needs lives here: the random seed, paths, class
labels, image size, normalisation, augmentation strength, decision threshold, the
metrics CSV columns and the Grad-CAM layer names. Other files import from here and
must never hardcode these values, so that all four stages stay consistent.

The reasons behind the values are the final decisions in issue #21 (closing comment).
Rule (CLAUDE.md): other stages may only ADD entries here (e.g. Ajit fills
LAST_CONV_LAYER["densenet121"]); changing an existing value needs an issue first.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed for everything (Python, NumPy, PyTorch, data shuffling, the val split).
# src.utils.set_seed() applies it; call that at the start of every script (#21 row 9).
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# REPO_ROOT is the folder that contains src/, data/, results/ ...
# __file__ is src/config.py, so parents[1] is the repository root. Using it makes all
# paths work no matter which folder a script is started from.
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"   # committed split CSVs (Stage 1.2), used by every stage
RESULTS_DIR = REPO_ROOT / "results"           # committed metrics and figures; the report takes numbers only from here

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"   # the only dataset used (#21 row 2)
OFFICIAL_FOLDERS = ("train", "val", "test")                  # folder names inside chest_xray/
CLASSES = ("NORMAL", "PNEUMONIA")                            # sub-folder names = class names


def get_data_root() -> Path:
    """Find the folder that contains the official train/, val/ and test/ folders.

    The same code must run on Kaggle, Colab and a laptop without editing paths (#21 row 1),
    so we try several locations in a fixed order and return the first one that has all
    three official folders:

    1. the PNEUMONIA_DATA_ROOT environment variable (set this to use data stored anywhere);
    2. the Kaggle input folder (current mount path first, then the older one);
    3. any other chest_xray/ folder under /kaggle/input (in case Kaggle changes the path again);
    4. data/chest_xray/ inside the repo (where the README tells Colab/local users to unzip).

    Only the top-level chest_xray/ folder is used. The Kaggle download also contains a
    duplicate copy at chest_xray/chest_xray/ and a __MACOSX/ folder; reading those
    would count every image twice, so they are skipped.

    Returns:
        Path to the folder that contains train/, val/ and test/.

    Raises:
        FileNotFoundError: if none of the locations has the dataset (the message lists
        every place that was checked).
    """
    candidates = []
    # 1. Explicit override by the user always wins.
    if os.environ.get("PNEUMONIA_DATA_ROOT"):
        candidates.append(Path(os.environ["PNEUMONIA_DATA_ROOT"]))
    # 2. Kaggle mount paths: the current one (seen on our Kaggle runs), then the older one.
    candidates.append(Path("/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"))
    candidates.append(Path("/kaggle/input/chest-xray-pneumonia/chest_xray"))
    # 3. Fallback search on Kaggle. Skip __MACOSX and the nested duplicate
    #    (a chest_xray folder whose parent is also called chest_xray).
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates += sorted(p for p in kaggle_input.glob("**/chest_xray")
                             if "__MACOSX" not in p.parts and p.parent.name != "chest_xray")
    # 4. Local / Colab download inside the repo.
    candidates.append(REPO_ROOT / "data" / "chest_xray")

    # Return the first candidate that really contains all three official folders.
    for root in candidates:
        if all((root / f).is_dir() for f in OFFICIAL_FOLDERS):
            return root
    raise FileNotFoundError(
        "Dataset not found. Attach the Kaggle dataset, or set PNEUMONIA_DATA_ROOT "
        "to the folder that contains train/, val/ and test/. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


# ---------------------------------------------------------------------------
# Labels (#21 row 4)
# ---------------------------------------------------------------------------
# PNEUMONIA = 1 is the positive class: precision, recall, F1 and AUC are all computed for it,
# and every model outputs ONE logit whose sigmoid is P(pneumonia).
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}

# ---------------------------------------------------------------------------
# Splits (#21 row 3, PR #23 review)
# ---------------------------------------------------------------------------
# Fraction of each class (from official train/ + val/) that goes into our validation split.
# The split is patient-level and person-grouped; see src/data/splits.py.
VAL_FRACTION = 0.125

# ---------------------------------------------------------------------------
# Preprocessing, shared by ALL models (#21 rows 5 and 6)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224                         # plain resize to 224x224 (no pad, no crop)
IMAGENET_MEAN = (0.485, 0.456, 0.406)    # ImageNet statistics, needed by the pretrained DenseNet-121 (Stage 2);
IMAGENET_STD = (0.229, 0.224, 0.225)     # the baseline uses the same values so every stage shares one pipeline
ROTATION_DEGREES = 10                    # train-only random rotation of +/-10 degrees. No horizontal flip:
                                         # flipping mirrors the heart and the L/R markers we study as shortcuts.

# ---------------------------------------------------------------------------
# Evaluation (#21 rows 4 and 8)
# ---------------------------------------------------------------------------
# Probability above which an image is called PNEUMONIA. If it is ever tuned, tune it on
# the validation set only (never on test) and record the new value here.
THRESHOLD = 0.5
# Column order of every results/stageN_metrics.csv, so Stage 4 can merge all stages into
# one table. attention_in_lung stays empty until Stage 3 computes the Grad-CAM metric.
METRICS_COLUMNS = (
    "model", "data_setting", "split", "accuracy", "precision", "recall", "f1",
    "auc", "tn", "fp", "fn", "tp", "attention_in_lung",
)

# ---------------------------------------------------------------------------
# Grad-CAM target layers (#21 row 9)
# ---------------------------------------------------------------------------
# Name of the last convolutional layer of each model, as used by model.get_submodule(name).
# Each stage fills in its own model so Aman (Stage 3) can hook Grad-CAM without guessing.
LAST_CONV_LAYER = {
    "custom_cnn": "features.block5.conv2",  # Stage 1.4, output 256 x 14 x 14
    # Stage 2.1 (Ajit): output of the last feature map, 1024 x 7 x 7 (BatchNorm after denseblock4).
    # src/models/densenet121.py applies a non-in-place ReLU after it, so hooks see clean tensors.
    "densenet121": "features.norm5",
}
