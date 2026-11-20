"""Training and evaluation loop shared by Stage 1 (custom CNN) and Stage 2 (DenseNet-121).

Owned by Stage 1 (Abhishek). Stage 2 should reuse fit() and predict() so both models
are trained and evaluated in exactly the same way.

Rules this file implements (issue #21):
- Loss: BCEWithLogitsLoss on ONE logit, with pos_weight from the train split (rows 4 and 7).
- Model selection and early stopping use the VALIDATION loss only. The test set is never
  used to choose anything (row 8); evaluate it once, with the best checkpoint, after fit().
- Mixed precision (AMP) on GPU for speed; on CPU everything runs in float32.

Checkpoints are dicts saved with torch.save:
    {"model_state": state_dict, "epoch": int, "val_loss": float, "val_auc": float,
     "extra": dict of all training settings}
Load with torch.load(path, map_location="cpu", weights_only=False) (weights_only=False is
needed because the dict contains more than tensors).

Example (see src/train_baseline.py for the full script):
    criterion = make_criterion(load_pos_weight(), device)
    model, history, best = fit(model, loaders, criterion, optimizer, device, checkpoint_path=...)
    test = predict(model, loaders["test"], device)
"""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.metrics import compute_metrics


def get_device() -> torch.device:
    """Return the GPU if one is available (Kaggle T4), otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_criterion(pos_weight: float, device) -> nn.Module:
    """Build the loss: binary cross-entropy on the logit, with a weight for the positive class.

    Args:
        pos_weight: #NORMAL / #PNEUMONIA in train (about 0.347, from split_summary.json).
            The loss of each PNEUMONIA image is multiplied by it, so the majority class
            (74% of train) does not dominate training.
        device: where the loss is computed (must match the model's device).

    Returns:
        nn.BCEWithLogitsLoss. It applies the sigmoid itself (numerically stable), so models
        output raw logits.
    """
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


@torch.no_grad()   # no gradients needed for evaluation: saves memory and time
def predict(model: nn.Module, loader, device, criterion=None) -> dict:
    """Run a model over every batch of a loader and collect its predictions.

    Args:
        model: a model that outputs logits of shape (N, 1).
        loader: DataLoader yielding (images, labels, image_ids).
        device: torch device the model is on.
        criterion: optional loss; if given, the mean loss over the loader is returned too.

    Returns:
        dict with
            labels     numpy array (N,) of true labels
            probs      numpy array (N,) of P(pneumonia) = sigmoid(logit)
            image_ids  list of N image_id strings, in loader order
            loss       mean loss per image, or None if no criterion was given
    """
    model.eval()   # evaluation mode: dropout off, BatchNorm uses its running statistics
    labels, probs, ids, loss_sum, n = [], [], [], 0.0, 0
    for images, y, image_ids in loader:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(images).squeeze(1)   # (N, 1) -> (N,)
        if criterion is not None:
            # criterion returns the batch mean; multiply by the batch size so the final
            # average is per image even when the last batch is smaller.
            loss_sum += criterion(logits, y.float()).item() * len(y)
        n += len(y)
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ids.extend(image_ids)
    return {"labels": np.concatenate(labels), "probs": np.concatenate(probs), "image_ids": ids,
            "loss": loss_sum / n if criterion is not None else None}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    """Train the model for one pass over the training loader.

    Args:
        model, loader, criterion, optimizer, device: the usual training objects.
        scaler: a torch.amp.GradScaler to train with mixed precision on GPU, or None for float32.

    Returns:
        Mean training loss per image for this epoch.
    """
    model.train()   # training mode: dropout on, BatchNorm updates its statistics
    loss_sum, n = 0.0, 0
    for images, y, _ in loader:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)   # clear gradients from the previous step
        # autocast runs the forward pass in float16 where that is safe (only when a scaler is used).
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images).squeeze(1), y)
        if scaler is not None:
            # Scale the loss so small float16 gradients don't underflow to zero, then unscale and step.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / n


def fit(model, loaders, criterion, optimizer, device, max_epochs: int = 30, patience: int = 5,
        scheduler=None, checkpoint_path=None, use_amp: bool = True, extra: dict = None):
    """Train with early stopping on the validation loss and keep the best weights.

    After every epoch the model is evaluated on val. Whenever the val loss reaches a new
    minimum, a copy of the weights is kept (and saved to checkpoint_path). Training stops
    when the val loss has not improved for `patience` epochs, or after `max_epochs`.
    The test loader is never touched here.

    Args:
        model: the model to train (already on `device`).
        loaders: dict with at least "train" and "val" DataLoaders.
        criterion: loss from make_criterion().
        optimizer: e.g. torch.optim.Adam(model.parameters(), ...).
        device: torch device.
        max_epochs: upper limit on epochs.
        patience: epochs without val-loss improvement before stopping.
        scheduler: optional ReduceLROnPlateau; it is stepped with the val loss after every epoch.
        checkpoint_path: where to save the best checkpoint (None = don't save).
        use_amp: use mixed precision when on GPU.
        extra: dict of settings stored in the checkpoint (for reproducibility).

    Returns:
        (model, history, best):
            model    the same model with the BEST weights loaded
            history  DataFrame with one row per epoch (lr, train_loss, val_loss, val_auc, val_recall, val_f1, seconds)
            best     the best checkpoint dict (model_state, epoch, val_loss, val_auc, extra)
    """
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    best, best_loss, bad_epochs, history = None, float("inf"), 0, []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler)
        val = predict(model, loaders["val"], device, criterion)
        val_m = compute_metrics(val["labels"], val["probs"])
        lr = optimizer.param_groups[0]["lr"]   # learning rate used in this epoch (logged before the scheduler changes it)
        if scheduler is not None:
            scheduler.step(val["loss"])   # e.g. halve the learning rate when val loss stops improving
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_auc": val_m["auc"], "val_recall": val_m["recall"], "val_f1": val_m["f1"],
                        "seconds": round(time.time() - start, 1)})
        improved = val["loss"] < best_loss
        # One log line per epoch; "*" marks a new best epoch.
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val['loss']:.4f}  "
              f"val_auc {val_m['auc']:.4f}  val_f1 {val_m['f1']:.4f}  lr {lr:.1e}  "
              f"{history[-1]['seconds']}s{'  *' if improved else ''}", flush=True)
        if improved:
            best_loss, bad_epochs = val["loss"], 0
            # deepcopy: state_dict() returns references that would change as training continues.
            best = {"model_state": copy.deepcopy(model.state_dict()), "epoch": epoch,
                    "val_loss": val["loss"], "val_auc": val_m["auc"], "extra": extra or {}}
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no val-loss improvement for {patience} epochs.")
                break
    # Return the model with the weights of the best epoch, not the last one.
    model.load_state_dict(best["model_state"])
    return model, pd.DataFrame(history), best
