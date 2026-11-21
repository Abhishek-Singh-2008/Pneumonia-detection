"""Stage 3.1: one Grad-CAM function for both models (owned by Aman, issue #11).

Grad-CAM (Selvaraju et al., ICCV 2017) shows which image regions a CNN used for a decision:
    1. run the image forward and keep the feature map A of one late layer, shape (K, h, w);
    2. back-propagate the score of the class we want to explain down to A, giving dScore/dA;
    3. weight each of the K channels by its average gradient: alpha_k = mean over (h, w) of dScore/dA_k;
    4. heatmap = ReLU( sum_k alpha_k * A_k ), shape (h, w), then upsample to 224 x 224.
The ReLU keeps only the regions that push the score UP.

Which score do we explain? (#21 row 4 and Aman's reply in #21)
Both models output ONE logit z, with sigmoid(z) = P(pneumonia). There is no separate NORMAL output, so:
    - image predicted PNEUMONIA (sigmoid(z) >= THRESHOLD): explain  +z  (what supports "pneumonia");
    - image predicted NORMAL:                              explain  -z  (what supports "normal").
So every heatmap shows the evidence for the class the model actually predicted, for both models alike.
target="pneumonia" is also offered: it always explains +z (useful to ask "where does it see pneumonia?").

Which layer? GRADCAM_LAYER below. Grad-CAM is normally computed on the activations AFTER the last ReLU
of the network, because those are the features the classifier really receives (all >= 0):
    - custom_cnn:  "features.block5.relu2", the ReLU after the last convolution, 256 x 14 x 14.
      Note: src.config.LAST_CONV_LAYER["custom_cnn"] is "features.block5.conv2", the same convolution
      BEFORE its BatchNorm and ReLU. Shared config values must not be changed without an issue (CLAUDE.md),
      so the post-ReLU choice is kept here in Stage 3's own file, and the report says which layer was used.
    - densenet121: "relu" from src.config.LAST_CONV_LAYER, the named post-ReLU module, 1024 x 7 x 7 (PR #24).

Usage:
    from src.explain.gradcam import grad_cam, GRADCAM_LAYER
    from src.explain.common import load_model
    model = load_model("densenet121", checkpoint_path, device)
    out = grad_cam(model, images, GRADCAM_LAYER["densenet121"])
    out["cams"]   # numpy (N, 224, 224), each heatmap scaled to max 1

Checked by: python -m src.explain.check_explain (synthetic tests, no dataset needed).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import IMAGE_SIZE, LAST_CONV_LAYER, THRESHOLD

# Layer hooked for Grad-CAM, per model (see the module docstring for why custom_cnn uses relu2).
GRADCAM_LAYER = {
    "custom_cnn": "features.block5.relu2",          # post-ReLU, 256 x 14 x 14
    "densenet121": LAST_CONV_LAYER["densenet121"],  # "relu": post-ReLU, 1024 x 7 x 7 (set by Ajit, PR #24)
}


def grad_cam(model: nn.Module, images: torch.Tensor, layer_name: str, target: str = "predicted",
             out_size: int = IMAGE_SIZE) -> dict:
    """Compute Grad-CAM heatmaps for a batch of images.

    Args:
        model: a model that outputs logits of shape (N, 1) (custom CNN or DenseNet-121). It is put in
            eval mode here: dropout off and BatchNorm on its running statistics, so every image in the batch
            is explained independently of the others.
        images: float tensor (N, 3, H, W), normalised exactly like the training pipeline
            (src.data.dataset.build_transforms(train=False)), on the same device as the model.
        layer_name: name of the layer to hook, as accepted by model.get_submodule(), e.g. GRADCAM_LAYER[...].
        target: "predicted" explains the predicted class (+logit if predicted PNEUMONIA, -logit if NORMAL);
            "pneumonia" always explains +logit.
        out_size: side length of the returned heatmaps (default 224, the model input size, so they line up
            pixel by pixel with the 224 x 224 lung masks of Stage 3.2).

    Returns:
        dict with numpy arrays
            cams        (N, out_size, out_size) float32, >= 0, each heatmap divided by its own maximum
                        (all-zero if the model found no positive evidence; the metric then returns NaN)
            probs       (N,) P(pneumonia) = sigmoid(logit)
            preds       (N,) 0/1 prediction at src.config.THRESHOLD
            signs       (N,) +1 or -1, the sign applied to the logit before back-propagation
            fmap_shape  (h, w) of the hooked feature map (14 x 14 for custom_cnn, 7 x 7 for densenet121)

    Side effects:
        Temporarily registers a forward hook on the layer (always removed, even on errors) and clears the
        model's gradients afterwards. Model weights are not changed.
    """
    if target not in ("predicted", "pneumonia"):
        raise ValueError(f"target must be 'predicted' or 'pneumonia', got {target!r}")
    model.eval()
    layer = model.get_submodule(layer_name)   # raises AttributeError if the name is wrong
    store = {}

    def forward_hook(_module, _inputs, output):
        """Keep the layer output and ask autograd to hand us its gradient during backward()."""
        store["acts"] = output
        # register_hook on the TENSOR (not a module backward hook) gives exactly dScore/dA for this output.
        output.register_hook(lambda grad: store.__setitem__("grads", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        # Gradients are needed even if the caller is inside torch.no_grad() (e.g. an evaluation loop).
        with torch.enable_grad():
            logits = model(images).squeeze(1)                     # (N, 1) -> (N,)
            probs = torch.sigmoid(logits)
            preds = (probs >= THRESHOLD).long()
            if target == "predicted":
                signs = preds.float() * 2 - 1                     # 1 -> +1 (explain pneumonia), 0 -> -1 (explain normal)
            else:
                signs = torch.ones_like(logits)
            # Summing over the batch is safe: in eval mode each image's score depends only on its own
            # activations, so d(sum)/dA_i = d(score_i)/dA_i for every image i.
            model.zero_grad(set_to_none=True)
            (signs * logits).sum().backward()
    finally:
        handle.remove()

    acts, grads = store["acts"].detach(), store["grads"].detach()  # both (N, K, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)                 # alpha_k: (N, K, 1, 1)
    cams = F.relu((weights * acts).sum(dim=1, keepdim=True))       # (N, 1, h, w), keep positive evidence only
    # Bilinear upsampling from the coarse feature map to the input size, so the heatmap can be
    # compared with the 224 x 224 lung mask pixel by pixel.
    cams = F.interpolate(cams.float(), size=(out_size, out_size), mode="bilinear", align_corners=False)
    cams = cams.squeeze(1)                                         # (N, out_size, out_size)
    # Scale each heatmap to max 1 for plotting. The attention metric is a ratio, so this scaling
    # does not change it. clamp avoids dividing by zero for an all-zero heatmap.
    cams = cams / cams.flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1)
    model.zero_grad(set_to_none=True)
    return {
        "cams": cams.cpu().numpy().astype(np.float32),
        "probs": probs.detach().cpu().numpy(),
        "preds": preds.cpu().numpy(),
        "signs": signs.detach().cpu().numpy(),
        "fmap_shape": tuple(acts.shape[2:]),
    }


def grad_cam_loader(model: nn.Module, loader, layer_name: str, device, target: str = "predicted") -> dict:
    """Run grad_cam() over every batch of a DataLoader and stack the results.

    Args:
        model: model on `device` (see grad_cam).
        loader: DataLoader yielding (images, labels, image_ids), e.g. the test loader of
            src.data.dataset.get_dataloaders (not shuffled, so the order follows the split CSV).
        layer_name, target: passed to grad_cam().
        device: torch device of the model.

    Returns:
        dict with numpy arrays cams (N, 224, 224), probs (N,), preds (N,), signs (N,), labels (N,),
        and image_ids (list of N str), in loader order.
    """
    parts = {k: [] for k in ("cams", "probs", "preds", "signs", "labels")}
    ids = []
    for images, labels, image_ids in loader:
        out = grad_cam(model, images.to(device), layer_name, target=target)
        for k in ("cams", "probs", "preds", "signs"):
            parts[k].append(out[k])
        parts["labels"].append(labels.numpy())
        ids.extend(image_ids)
    result = {k: np.concatenate(v) for k, v in parts.items()}
    result["image_ids"] = ids
    return result
