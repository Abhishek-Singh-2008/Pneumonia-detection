"""Stage 1.4: the custom CNN baseline, trained from scratch (owned by Abhishek).

A deliberately simple model, so we can compare it with the much larger pretrained
DenseNet-121 in Stage 2 and see whether both look at the lungs (Stage 3).

Architecture (input: a batch of normalised images, shape (N, 3, 224, 224)):

    5 conv blocks, each = [3x3 conv -> BatchNorm -> ReLU] x 2 -> 2x2 max pool
        channels:     32      64      128     256     256
        output size: 112  ->  56  ->  28  ->  14  ->   7   (pooling halves height and width)
    global average pooling   (N, 256, 7, 7) -> (N, 256)
    dropout (p = 0.3)
    linear layer             (N, 256) -> (N, 1)   = the PNEUMONIA logit (#21 row 4)

2.35 M parameters. The sigmoid of the logit is P(pneumonia); the loss is
BCEWithLogitsLoss, which applies the sigmoid itself, so the model has no sigmoid layer.

Grad-CAM (Stage 3, #21 rows 9 and 10): the target layer is `features.block5.conv2`,
the last convolution, whose output is (N, 256, 14, 14). Its name is stored in
src.config.LAST_CONV_LAYER["custom_cnn"]; get the layer with
model.get_submodule(LAST_CONV_LAYER["custom_cnn"]). The ReLUs are NOT in-place, so
forward/backward hooks see the unmodified activations and gradients.
"""
from collections import OrderedDict

import torch
from torch import nn

# Output channels of the five conv blocks.
CHANNELS = (32, 64, 128, 256, 256)


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    """One conv block: two 3x3 convolutions with BatchNorm and ReLU, then 2x2 max pooling.

    Args:
        in_ch: input channels.
        out_ch: output channels.

    Returns:
        nn.Sequential with named layers conv1, bn1, relu1, conv2, bn2, relu2, pool
        (named so that e.g. "features.block5.conv2" can be addressed for Grad-CAM).
    """
    return nn.Sequential(OrderedDict([
        # padding=1 keeps height and width the same; bias=False because BatchNorm adds its own bias.
        ("conv1", nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)),
        ("bn1", nn.BatchNorm2d(out_ch)),
        ("relu1", nn.ReLU()),       # not in-place, so Grad-CAM hooks see the original tensor
        ("conv2", nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
        ("bn2", nn.BatchNorm2d(out_ch)),
        ("relu2", nn.ReLU()),
        ("pool", nn.MaxPool2d(2)),  # halves height and width
    ]))


class CustomCNN(nn.Module):
    """The baseline classifier.

    Args:
        channels: output channels of each conv block (default CHANNELS: five blocks).
        dropout: dropout probability before the final linear layer (default 0.3), to reduce overfitting.

    Forward input:  float tensor (N, 3, 224, 224), normalised as in src/data/dataset.py.
    Forward output: float tensor (N, 1) of logits; sigmoid(logit) = P(pneumonia).
    """

    def __init__(self, channels=CHANNELS, dropout: float = 0.3):
        super().__init__()
        # Build block1 ... block5; each block's input channels = previous block's output channels.
        blocks, in_ch = OrderedDict(), 3   # 3 input channels (grayscale copied to RGB)
        for i, out_ch in enumerate(channels, start=1):
            blocks[f"block{i}"] = _block(in_ch, out_ch)
            in_ch = out_ch
        self.features = nn.Sequential(blocks)  # 224 -> 7 after five poolings
        self.pool = nn.AdaptiveAvgPool2d(1)    # global average pooling: one number per channel
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, 1)  # single PNEUMONIA logit
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise the weights.

        Kaiming (He) initialisation for the convolutions is designed for ReLU networks and keeps
        activations from shrinking or exploding through the layers; BatchNorm starts as the
        identity (scale 1, shift 0). The linear layer keeps PyTorch's default initialisation.
        All of this runs after set_seed(), so the initial weights are the same in every run.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Args:
            x: images, shape (N, 3, 224, 224).

        Returns:
            logits, shape (N, 1). Use .squeeze(1) to get shape (N,) for the loss.
        """
        x = self.pool(self.features(x)).flatten(1)   # (N, 256, 7, 7) -> (N, 256, 1, 1) -> (N, 256)
        return self.classifier(self.dropout(x))       # (N, 256) -> (N, 1)
