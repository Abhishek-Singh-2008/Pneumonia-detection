"""Build one table row per image from the official dataset folders.

Patient-ID rule (agreed in issue #21):
    PNEUMONIA  person{N}_{bacteria|virus}_{M}[_{K}].jpeg  ->  PNEUMONIA_person{N}_{type}
    NORMAL     IM-{XXXX}-{YYYY}[-{Z}].jpeg                ->  NORMAL_IM-{XXXX}
    NORMAL     NORMAL2-IM-{XXXX}-{YYYY}[-{Z}].jpeg        ->  NORMAL_NORMAL2-IM-{XXXX}

image_id = {class}_{official_folder}_{file_stem}, e.g. NORMAL_train_IM-0115-0001.
It is unique across the dataset and is the key for lung masks ({image_id}.png).
"""
import re
from pathlib import Path

import pandas as pd

from src.config import CLASSES, LABELS, OFFICIAL_FOLDERS

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

_PATIENT_PATTERNS = {
    "PNEUMONIA": re.compile(r"^(person\d+)_(bacteria|virus)_\d+(?:_\d+)?$", re.IGNORECASE),
    "NORMAL": re.compile(r"^((?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)?$", re.IGNORECASE),
}


def parse_patient_id(class_name: str, file_stem: str) -> str:
    """Return the patient ID for one file, or raise ValueError if the name is unexpected."""
    match = _PATIENT_PATTERNS[class_name].match(file_stem)
    if match is None:
        raise ValueError(f"Unexpected {class_name} filename: {file_stem!r}")
    if class_name == "PNEUMONIA":
        return f"PNEUMONIA_{match.group(1)}_{match.group(2).lower()}"
    return f"NORMAL_{match.group(1)}"


def make_image_id(class_name: str, official_folder: str, file_stem: str) -> str:
    return f"{class_name}_{official_folder}_{file_stem}"


def build_index(data_root) -> pd.DataFrame:
    """Scan data_root/{train,val,test}/{NORMAL,PNEUMONIA}/ and return one row per image.

    Columns: image_id, filepath (relative to data_root, forward slashes), label,
    label_name, patient_id, official_folder.
    """
    data_root = Path(data_root)
    rows = []
    for folder in OFFICIAL_FOLDERS:
        for class_name in CLASSES:
            class_dir = data_root / folder / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {class_dir}")
            for path in sorted(class_dir.iterdir()):
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
    if index["image_id"].duplicated().any():
        dupes = index.loc[index["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"image_id is not unique, e.g. {dupes}")
    return index
