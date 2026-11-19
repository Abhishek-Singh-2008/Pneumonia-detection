"""Shared settings for every stage. Change values here, never hardcode them elsewhere.

Final decisions behind these values: issue #21 (closing comment).
"""
import os
from pathlib import Path

# Reproducibility
SEED = 42

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
OFFICIAL_FOLDERS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def get_data_root() -> Path:
    """Folder that contains the official train/, val/ and test/ folders.

    Order: the PNEUMONIA_DATA_ROOT environment variable, then the Kaggle input
    folder, then data/chest_xray inside the repo (Colab or local download).
    Only the top-level chest_xray/ is used, never the duplicate
    chest_xray/chest_xray/ or __MACOSX/ folders.
    """
    candidates = []
    if os.environ.get("PNEUMONIA_DATA_ROOT"):
        candidates.append(Path(os.environ["PNEUMONIA_DATA_ROOT"]))
    # Current Kaggle mount path, then the older one
    candidates.append(Path("/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"))
    candidates.append(Path("/kaggle/input/chest-xray-pneumonia/chest_xray"))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # Newer Kaggle images can mount datasets under a deeper path.
        candidates += sorted(p for p in kaggle_input.glob("**/chest_xray")
                             if "__MACOSX" not in p.parts and p.parent.name != "chest_xray")
    candidates.append(REPO_ROOT / "data" / "chest_xray")

    for root in candidates:
        if all((root / f).is_dir() for f in OFFICIAL_FOLDERS):
            return root
    raise FileNotFoundError(
        "Dataset not found. Attach the Kaggle dataset, or set PNEUMONIA_DATA_ROOT "
        "to the folder that contains train/, val/ and test/. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


# Labels (PNEUMONIA is the positive class)
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}

# Splits: patient-level, stratified, 12.5% of the train+val pool as val
VAL_N_SPLITS = 8  # StratifiedGroupKFold(n_splits=8) -> first fold is val

# Preprocessing (shared by all models)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROTATION_DEGREES = 10  # train only; no horizontal flip

# Evaluation
THRESHOLD = 0.5  # if tuned, only on val, and record the new value here
METRICS_COLUMNS = (
    "model", "data_setting", "split", "accuracy", "precision", "recall", "f1",
    "auc", "tn", "fp", "fn", "tp", "attention_in_lung",
)

# Grad-CAM target layers, filled in by the stage that builds each model
LAST_CONV_LAYER = {
    "custom_cnn": None,    # Stage 1.4
    "densenet121": None,   # Stage 2
}
