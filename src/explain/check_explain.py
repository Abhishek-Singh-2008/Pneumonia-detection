"""Stage 3 self-checks: Grad-CAM, the attention metric and the mask files, on synthetic data (owned by Aman, #11-#13).

Nothing here needs the dataset, the trained weights or a GPU, so it runs anywhere in about a minute:
    python -m src.explain.check_explain

What is checked (each check prints PASS or raises AssertionError):
1. attention metric maths: full mask -> 1, empty mask -> 0, an even heatmap -> the lung area, scaling a
   heatmap changes nothing, an all-zero heatmap -> NaN, wrong shapes are refused; border band of an even heatmap = 0.36.
2. Grad-CAM finds a known answer: a toy model whose "pneumonia" evidence is a bright square in the top-left
   and whose "normal" evidence is a dark square in the bottom-right. Predicted PNEUMONIA must light up the
   top-left; predicted NORMAL must light up the bottom-right (this tests the +logit / -logit rule of #21).
3. Grad-CAM on the real architectures (random weights): output shape (N, 224, 224), values in [0, 1], the
   feature-map sizes 14 x 14 (custom CNN) and 7 x 7 (DenseNet-121), works inside torch.no_grad(),
   and does not change the model.
4. Checkpoint loading: a saved checkpoint is found in a sub-folder and loads with strict=True.
5. Lung-mask files: save -> load returns the same binary 224 x 224 mask (the format agreed in #21 row 14).
"""
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import IMAGE_SIZE
from src.explain.attention import attention_in_lung, border_fraction, lung_area_fraction
from src.explain.common import find_checkpoint, load_model
from src.explain.gradcam import GRADCAM_LAYER, grad_cam
from src.models.custom_cnn import CustomCNN
from src.models.densenet121 import DenseNet121Classifier
from src.utils import set_seed


def check_attention_metric() -> None:
    """Check 1: the metric formulas on hand-made heatmaps and masks."""
    rng = np.random.RandomState(0)
    cam = rng.rand(IMAGE_SIZE, IMAGE_SIZE)              # random positive heatmap
    full = np.ones_like(cam, dtype=bool)
    empty = np.zeros_like(cam, dtype=bool)
    half = np.zeros_like(cam, dtype=bool)
    half[:, : IMAGE_SIZE // 2] = True                   # left half of the image is "lung"
    assert abs(attention_in_lung(cam, full) - 1.0) < 1e-12
    assert attention_in_lung(cam, empty) == 0.0
    even = np.ones_like(cam)
    assert abs(attention_in_lung(even, half) - 0.5) < 1e-12, "even heatmap must score the lung area"
    assert abs(lung_area_fraction(half) - 0.5) < 1e-12
    assert abs(attention_in_lung(cam, half) - attention_in_lung(7.3 * cam, half)) < 1e-12, "must be scale-invariant"
    assert np.isnan(attention_in_lung(np.zeros_like(cam), half)), "all-zero heatmap must give NaN"
    batch = attention_in_lung(np.stack([cam, even]), np.stack([full, half]))
    assert batch.shape == (2,) and abs(batch[1] - 0.5) < 1e-12
    for bad in (np.ones((10, 10)), -cam):
        try:
            attention_in_lung(bad, half)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong shape / negative heatmap must raise ValueError")
    assert abs(border_fraction(even) - (1 - 0.8 ** 2)) < 0.01, "outer 10% band of an even heatmap is ~36%"
    print("PASS 1: attention metric, lung area and border fraction")


class ToyModel(nn.Module):
    """A tiny model with a known answer for Grad-CAM (check 2).

    features: a fixed 1x1 convolution makes channel 0 = +mean(RGB) and channel 1 = -mean(RGB), then a ReLU
    named "relu" (hooked like DenseNet's). So channel 0 is active where the image is bright and channel 1
    where it is dark. logit = mean(channel 0) - mean(channel 1) + bias: brightness is evidence for
    pneumonia, darkness evidence for normal. The bias decides which class is predicted.
    """

    def __init__(self, bias: float):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[1, 1, 1], [-1, -1, -1]], dtype=torch.float32).view(2, 3, 1, 1) / 3)
        self.relu = nn.ReLU()
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> logits (N, 1)."""
        f = self.relu(self.conv(x))                       # (N, 2, H, W)
        return (f[:, 0].mean(dim=(1, 2)) - f[:, 1].mean(dim=(1, 2)) + self.bias).unsqueeze(1)


def check_gradcam_known_answer() -> None:
    """Check 2: Grad-CAM points at the right square for each predicted class."""
    x = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    q = IMAGE_SIZE // 4
    x[:, :, :q, :q] = 1.0                                 # bright square, top-left: pneumonia evidence
    x[:, :, -q:, -q:] = -1.0                              # dark square, bottom-right: normal evidence
    top_left = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
    top_left[:q, :q] = True
    bottom_right = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
    bottom_right[-q:, -q:] = True

    pneu = grad_cam(ToyModel(bias=+5.0), x, "relu")       # large positive bias -> predicted PNEUMONIA
    assert pneu["preds"][0] == 1 and pneu["signs"][0] == 1
    assert attention_in_lung(pneu["cams"][0], top_left) > 0.95, "PNEUMONIA heatmap must sit on the bright square"

    norm = grad_cam(ToyModel(bias=-5.0), x, "relu")       # large negative bias -> predicted NORMAL
    assert norm["preds"][0] == 0 and norm["signs"][0] == -1
    assert attention_in_lung(norm["cams"][0], bottom_right) > 0.95, "NORMAL heatmap must sit on the dark square"

    # target="pneumonia" always explains +logit, even for a NORMAL prediction.
    forced = grad_cam(ToyModel(bias=-5.0), x, "relu", target="pneumonia")
    assert attention_in_lung(forced["cams"][0], top_left) > 0.95
    print("PASS 2: Grad-CAM finds the known evidence for both predicted classes (+logit / -logit rule)")


def check_real_architectures() -> None:
    """Check 3: Grad-CAM runs on both project models with random weights."""
    set_seed(42)
    images = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
    for name, model, fmap in (("custom_cnn", CustomCNN(), (14, 14)),
                              ("densenet121", DenseNet121Classifier(pretrained=False), (7, 7))):
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():                             # grad_cam must still work inside no_grad
            out = grad_cam(model, images, GRADCAM_LAYER[name])
            logits = model(images).squeeze(1)
        assert out["cams"].shape == (3, IMAGE_SIZE, IMAGE_SIZE), out["cams"].shape
        assert out["cams"].min() >= 0 and out["cams"].max() <= 1 + 1e-6
        assert out["fmap_shape"] == fmap, (name, out["fmap_shape"])
        assert np.allclose(out["probs"], torch.sigmoid(logits).numpy(), atol=1e-5), "Grad-CAM must not change outputs"
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "weights must not change"
        assert all(p.grad is None for p in model.parameters()), "gradients must be cleared"
        print(f"PASS 3: {name}: layer {GRADCAM_LAYER[name]} -> feature map {fmap}, heatmaps (3, 224, 224)")


def check_checkpoint_loading() -> None:
    """Check 4: find_checkpoint searches sub-folders and load_model rebuilds the exact weights."""
    set_seed(42)
    model = CustomCNN()
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "drive_download" / "weights"
        sub.mkdir(parents=True)
        # Same dict layout as src/train.py writes.
        torch.save({"model_state": model.state_dict(), "epoch": 1, "val_loss": 0.1, "val_auc": 0.9,
                    "extra": {"dropout": 0.3}}, sub / "custom_cnn_best.pth")
        loaded = load_model("custom_cnn", find_checkpoint("custom_cnn", tmp), torch.device("cpu"))
    assert not loaded.training, "loaded model must be in eval mode"
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
    print("PASS 4: checkpoint found in a sub-folder and loaded with strict=True")


def check_mask_files() -> None:
    """Check 5: lung masks survive a save/load round trip in the agreed format (#21 row 14)."""
    from src.masking.lung_masks import load_mask, save_mask
    rng = np.random.RandomState(1)
    mask = rng.rand(IMAGE_SIZE, IMAGE_SIZE) > 0.6
    with tempfile.TemporaryDirectory() as tmp:
        save_mask(mask, Path(tmp), "NORMAL_test_IM-0001-0001")
        path = Path(tmp) / "NORMAL_test_IM-0001-0001.png"
        assert path.exists()
        back = load_mask(Path(tmp), "NORMAL_test_IM-0001-0001")
    assert back.dtype == bool and back.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert np.array_equal(back, mask)
    print("PASS 5: mask PNG round trip ({image_id}.png, binary, 224 x 224)")


def main() -> None:
    """Run all checks in order."""
    check_attention_metric()
    check_gradcam_known_answer()
    check_real_architectures()
    check_checkpoint_loading()
    check_mask_files()
    print("All Stage 3 checks passed.")


if __name__ == "__main__":
    main()
