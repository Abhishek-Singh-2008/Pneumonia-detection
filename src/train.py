"""Training and evaluation loop shared by Stage 1 (custom CNN) and Stage 2 (DenseNet-121).

- Loss: BCEWithLogitsLoss with pos_weight from the train split (issue #21 rows 4 and 7).
- Model selection / early stopping on val loss only. The test set is never used to
  choose anything (row 8); evaluate it once with the best checkpoint.
- Checkpoints are dicts: {"model_state", "epoch", "val_loss", "val_auc", "extra"}.
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_criterion(pos_weight: float, device) -> nn.Module:
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))


@torch.no_grad()
def predict(model: nn.Module, loader, device, criterion=None) -> dict:
    """Run the model over a loader. Returns labels, probs, image_ids and mean loss."""
    model.eval()
    labels, probs, ids, loss_sum, n = [], [], [], 0.0, 0
    for images, y, image_ids in loader:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        if criterion is not None:
            loss_sum += criterion(logits, y.float()).item() * len(y)
        n += len(y)
        labels.append(y.cpu().numpy())
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ids.extend(image_ids)
    return {"labels": np.concatenate(labels), "probs": np.concatenate(probs), "image_ids": ids,
            "loss": loss_sum / n if criterion is not None else None}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for images, y, _ in loader:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images).squeeze(1), y)
        if scaler is not None:
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
    """Train with early stopping on val loss. Loads the best weights into `model` before returning.

    Returns (model, history DataFrame, best checkpoint dict).
    """
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    best, best_loss, bad_epochs, history = None, float("inf"), 0, []
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, scaler)
        val = predict(model, loaders["val"], device, criterion)
        val_m = compute_metrics(val["labels"], val["probs"])
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val["loss"])
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_auc": val_m["auc"], "val_recall": val_m["recall"], "val_f1": val_m["f1"],
                        "seconds": round(time.time() - start, 1)})
        improved = val["loss"] < best_loss
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val['loss']:.4f}  "
              f"val_auc {val_m['auc']:.4f}  val_f1 {val_m['f1']:.4f}  lr {lr:.1e}  "
              f"{history[-1]['seconds']}s{'  *' if improved else ''}", flush=True)
        if improved:
            best_loss, bad_epochs = val["loss"], 0
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
    model.load_state_dict(best["model_state"])
    return model, pd.DataFrame(history), best
