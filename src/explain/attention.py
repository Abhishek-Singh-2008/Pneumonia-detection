"""Stage 3.3: the lung-attention metric and two helper measures (owned by Aman, issue #13).

This file is the reusable function Himanshu calls in Stage 4.3 on the masked model (#10 "Done when").

attention_in_lung (the project's research metric, team guide / issue #13):
    attention_in_lung = sum(heatmap inside the lung mask) / sum(heatmap over the whole image)
It is the share of the Grad-CAM "evidence" that lies on the lungs: 1.0 = all inside, 0.0 = all outside.

Why also lung_area_fraction? A heatmap that is spread evenly over the image would still score
attention_in_lung = (lung area / image area), roughly 0.25 to 0.35 on these X-rays. So the metric only means
something compared with that chance level: a model that looks at the lungs must score clearly ABOVE its
lung_area_fraction. Every per-image table reports both, plus their ratio (lift = attention / area).

border_fraction: share of the heatmap in the outer band of the image (default 10% on each side).
Text markers ("R", "L"), cropped edges and scanner borders sit there, so a high value is a shortcut warning
(Stage 3.4). An even heatmap scores 1 - 0.8^2 = 0.36 for a 10% band.

All functions take numpy arrays and work on one image (H, W) or a batch (N, H, W).
Heatmaps must be >= 0 (Grad-CAM output) and the same size as the mask (224 x 224, Stage 3.2).
Checked by: python -m src.explain.check_explain
"""
import numpy as np


def _as_batch(a: np.ndarray) -> np.ndarray:
    """Return a float64 array of shape (N, H, W); a single (H, W) image becomes N = 1."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 2:
        return a[None]
    if a.ndim != 3:
        raise ValueError(f"expected shape (H, W) or (N, H, W), got {a.shape}")
    return a


def attention_in_lung(cams, masks):
    """Share of each Grad-CAM heatmap that falls inside the lung mask.

    Args:
        cams: heatmaps, numpy (H, W) or (N, H, W), values >= 0 (e.g. grad_cam(...)["cams"]).
            Any positive scaling of a heatmap gives the same result, because the metric is a ratio.
        masks: lung masks with the same shape, bool or 0/1 (e.g. from src.masking.lung_masks.load_mask);
            any value > 0 counts as lung.

    Returns:
        float for a single image, or numpy (N,) for a batch, each in [0, 1]. NaN when a heatmap is all zero
        (the model found no positive evidence, so "where it looked" is undefined; such images are
        left out of averages with np.nanmean and counted separately).

    Raises:
        ValueError: if the shapes differ or a heatmap has negative values.
    """
    single = np.ndim(cams) == 2
    c, m = _as_batch(cams), _as_batch(masks) > 0
    if c.shape != m.shape:
        raise ValueError(f"heatmap shape {c.shape} != mask shape {m.shape}; resize both to 224 x 224 first")
    if (c < 0).any():
        raise ValueError("heatmaps must be >= 0 (apply the Grad-CAM ReLU first)")
    total = c.sum(axis=(1, 2))
    inside = (c * m).sum(axis=(1, 2))
    # errstate: an all-zero heatmap gives 0/0 -> NaN on purpose (see Returns), without a warning.
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, inside / total, np.nan)
    return float(out[0]) if single else out


def lung_area_fraction(masks):
    """Share of the image covered by the lung mask: the chance level of attention_in_lung.

    Args:
        masks: numpy (H, W) or (N, H, W), bool or 0/1.

    Returns:
        float (single image) or numpy (N,), in [0, 1].
    """
    single = np.ndim(masks) == 2
    out = (_as_batch(masks) > 0).mean(axis=(1, 2))
    return float(out[0]) if single else out


def border_fraction(cams, width: float = 0.10):
    """Share of each heatmap that lies in the outer border band of the image.

    Args:
        cams: heatmaps, numpy (H, W) or (N, H, W), values >= 0.
        width: band width on EACH side, as a fraction of the image side (0.10 = outer 10%).

    Returns:
        float (single image) or numpy (N,); NaN for an all-zero heatmap.
    """
    single = np.ndim(cams) == 2
    c = _as_batch(cams)
    _, h, w = c.shape
    bh, bw = int(round(h * width)), int(round(w * width))
    # Build a mask of the border band: everything except the central rectangle.
    band = np.ones((h, w), dtype=bool)
    band[bh:h - bh, bw:w - bw] = False
    total = c.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, (c * band).sum(axis=(1, 2)) / total, np.nan)
    return float(out[0]) if single else out
