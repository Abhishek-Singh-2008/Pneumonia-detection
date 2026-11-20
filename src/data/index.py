"""Stage 1.1: build a table with one row per image of the dataset (owned by Abhishek).

The Kaggle dataset is just folders of JPEG files:
    <data_root>/{train,val,test}/{NORMAL,PNEUMONIA}/<file>.jpeg
build_index() turns those folders into a pandas DataFrame that every later step uses
(the EDA notebook, the split, the data loader).

Two IDs are derived from each filename (decided in issue #21, rows 3 and 14):

patient_id: which child the image belongs to. Several images can come from one child,
    and they must never be spread over training and evaluation sets.
        PNEUMONIA  person{N}_{bacteria|virus}_{M}[_{K}].jpeg  ->  PNEUMONIA_person{N}_{type}
        NORMAL     IM-{XXXX}-{YYYY}[-{Z}].jpeg                ->  NORMAL_IM-{XXXX}
        NORMAL     NORMAL2-IM-{XXXX}-{YYYY}[-{Z}].jpeg        ->  NORMAL_NORMAL2-IM-{XXXX}
    The infection type is part of the pneumonia ID because with it the official train and
    test folders share zero patients (without it they would share 170). The val split is
    still grouped by person{N} alone as a safety margin (src/data/splits.py, PR #23).
    The class name is prefixed so that IDs from the two classes can never collide.

image_id: a unique name for every image = {class}_{official_folder}_{file_stem},
    e.g. NORMAL_train_IM-0115-0001. Lung masks (Stage 3) are saved as {image_id}.png and
    predictions are saved per image_id, so files from different stages can be joined.
"""
import re
from pathlib import Path

import pandas as pd

from src.config import CLASSES, LABELS, OFFICIAL_FOLDERS

# Only image files are indexed; anything else in the folders (e.g. .DS_Store) is ignored.
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

# Filename patterns (matched against the file name without its extension).
# Every one of the 5,856 files matches one of these; an unexpected name raises an error
# instead of silently getting a wrong patient ID.
_PATIENT_PATTERNS = {
    # person1_bacteria_1, person1_virus_6, and one odd file person447_virus_921_1
    # group(1) = "person{N}", group(2) = "bacteria" or "virus"
    "PNEUMONIA": re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)?$", re.IGNORECASE),
    # IM-0115-0001, IM-0115-0001-0002, NORMAL2-IM-1427-0001, NORMAL2-IM-0381-0001-0002
    # group(1) = "IM-{XXXX}" or "NORMAL2-IM-{XXXX}" (the patient part, before the image number)
    "NORMAL": re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)?$", re.IGNORECASE),
}


def parse_patient_id(class_name: str, file_stem: str) -> str:
    """Return the patient ID for one image file.

    Args:
        class_name: "NORMAL" or "PNEUMONIA" (the folder the file is in).
        file_stem: the file name without extension, e.g. "person1_virus_6".

    Returns:
        e.g. "PNEUMONIA_person1_virus", "NORMAL_IM-0115" or "NORMAL_NORMAL2-IM-1427".

    Raises:
        ValueError: if the file name does not follow the expected pattern.
    """
    match = _PATIENT_PATTERNS[class_name].match(file_stem)
    if match is None:
        raise ValueError(f"Unexpected {class_name} filename: {file_stem!r}")
    if class_name == "PNEUMONIA":
        # Lower-case the type so "Bacteria" and "bacteria" would give the same ID.
        return f"PNEUMONIA_{match.group(1)}_{match.group(2).lower()}"
    return f"NORMAL_{match.group(1)}"


def make_image_id(class_name: str, official_folder: str, file_stem: str) -> str:
    """Return the unique image ID, e.g. make_image_id("NORMAL", "train", "IM-0115-0001")
    -> "NORMAL_train_IM-0115-0001". The official folder (not our new split) is used, so the
    ID never changes even if the split changes."""
    return f"{class_name}_{official_folder}_{file_stem}"


def build_index(data_root) -> pd.DataFrame:
    """Scan data_root/{train,val,test}/{NORMAL,PNEUMONIA}/ and return one row per image.

    Args:
        data_root: folder that contains train/, val/ and test/ (see src.config.get_data_root()).

    Returns:
        DataFrame with 5,856 rows and the columns
            image_id         unique ID, e.g. NORMAL_train_IM-0115-0001
            filepath         path relative to data_root with forward slashes, e.g. train/NORMAL/IM-0115-0001.jpeg
                             (relative + "/" so the same CSV works on Windows, Linux and Kaggle)
            label            0 = NORMAL, 1 = PNEUMONIA
            label_name       "NORMAL" or "PNEUMONIA"
            patient_id       see the module docstring
            official_folder  "train", "val" or "test" (the Kaggle folder the image came from)

    Raises:
        FileNotFoundError: if one of the six class folders is missing.
        ValueError: if a filename is unexpected or two images get the same image_id.
    """
    data_root = Path(data_root)
    rows = []
    for folder in OFFICIAL_FOLDERS:
        for class_name in CLASSES:
            class_dir = data_root / folder / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            # sorted() gives a stable order; the split later re-sorts by image_id anyway,
            # so the result never depends on how the operating system lists files.
            for path in sorted(class_dir.iterdir()):
                # Skip non-images and hidden files such as ._IM-0115-0001.jpeg from macOS.
                if path.suffix.lower() not in IMAGE_EXTENSIONS or path.name.startswith("."):
                    continue
                rows.append({
                    "image_id": make_image_id(class_name, folder, path.stem),
                    "filepath": path.relative_to(data_root).as_posix(),
                    "label": LABELS[class_name],
                    "label_name": class_name,
                    "patient_id": parse_patient_id(class_name, path.stem),
                    "official_folder": folder,
                })
    index = pd.DataFrame(rows)
    # image_id is the join key for masks and predictions, so it must be unique.
    if index["image_id"].duplicated().any():
        dupes = index.loc[index["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"image_id is not unique, e.g. {dupes}")
    return index
