"""Stage 3 helpers: load the two trained models from their checkpoints (owned by Aman, issue #11).

Both Stage 1 and Stage 2 save checkpoints as dicts (see src/train.py):
    {"model_state": state_dict, "epoch": int, "val_loss": float, "val_auc": float, "extra": dict of settings}
The weights are NOT in the repo; they are in the team Google Drive folder (README). On Kaggle, put them in
one folder (e.g. a private Kaggle dataset, or /kaggle/working/checkpoints) and pass that folder.

Usage:
    from src.explain.common import load_model, find_checkpoint
    model = load_model("custom_cnn", find_checkpoint("custom_cnn", "/kaggle/working/checkpoints"), device)
"""
from pathlib import Path

import torch
from torch import nn

from src.models.custom_cnn import CustomCNN
from src.models.densenet121 import DenseNet121Classifier

# The two unmasked models analysed in Stage 3, in report order (baseline first).
MODELS = ("custom_cnn", "densenet121")

# File name of each final checkpoint, as written by the Stage 1 and Stage 2 handover notes (#1, #6).
CHECKPOINT_FILES = {
    "custom_cnn": "custom_cnn_best.pth",                         # src/train_baseline.py
    "densenet121": "densenet121_phaseB_lr1e-4_bs16_best.pth",   # chosen Phase B run (val loss), src/evaluate_densenet.py
}


def find_checkpoint(model_name: str, checkpoint_dir) -> Path:
    """Locate a model's checkpoint file inside a folder (searched recursively).

    Recursive search because Kaggle datasets and Drive downloads often add an extra sub-folder.

    Args:
        model_name: "custom_cnn" or "densenet121".
        checkpoint_dir: folder that contains the .pth file somewhere below it.

    Returns:
        Path to the checkpoint.

    Raises:
        FileNotFoundError: if the file is not found (the message names the expected file).
    """
    name = CHECKPOINT_FILES[model_name]
    hits = sorted(Path(checkpoint_dir).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {checkpoint_dir}. Download it from the team Drive folder (README).")
    return hits[0]


def load_model(model_name: str, checkpoint_path, device) -> nn.Module:
    """Rebuild a trained model and load its weights, ready for evaluation and Grad-CAM.

    Args:
        model_name: "custom_cnn" or "densenet121".
        checkpoint_path: path to the .pth checkpoint dict.
        device: torch device to move the model to.

    Returns:
        The model in eval mode on `device`.

    Notes:
        load_state_dict(strict=True) fails loudly if the architecture does not match the weights, so a wrong
        file can never be analysed silently. The dropout value is read from the saved settings; it has no
        effect in eval mode but keeps the rebuilt model identical to the trained one.
        weights_only=False is required because the checkpoint also stores the settings dict (CLAUDE.md).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dropout = ckpt.get("extra", {}).get("dropout")
    if model_name == "custom_cnn":
        model = CustomCNN() if dropout is None else CustomCNN(dropout=dropout)
    elif model_name == "densenet121":
        # pretrained=False: all weights come from the checkpoint, so nothing is downloaded.
        model = DenseNet121Classifier(pretrained=False) if dropout is None else \
            DenseNet121Classifier(pretrained=False, dropout=dropout)
    else:
        raise ValueError(f"unknown model {model_name!r}; expected one of {MODELS}")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()
