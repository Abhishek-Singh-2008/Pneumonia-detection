"""Stage 1.4: custom CNN baseline trained from scratch.

Five conv blocks (two 3x3 conv + BatchNorm + ReLU each, then 2x2 max pooling),
global average pooling, dropout and one output logit (PNEUMONIA = 1, issue #21 row 4).

Grad-CAM target (issue #21 rows 9 and 10): the last conv layer is
`features.block5.conv2`, stored in src.config.LAST_CONV_LAYER["custom_cnn"].
Get it with model.get_submodule(LAST_CONV_LAYER["custom_cnn"]).
ReLUs are not in-place, so forward/backward hooks see unmodified tensors.
"""
from collections import OrderedDict

import torch
from torch import nn

CHANNELS = (32, 64, 128, 256, 256)


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict([
        ("conv1", nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)),
        ("bn1", nn.BatchNorm2d(out_ch)),
        ("relu1", nn.ReLU()),
        ("conv2", nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
        ("bn2", nn.BatchNorm2d(out_ch)),
        ("relu2", nn.ReLU()),
        ("pool", nn.MaxPool2d(2)),
    ]))


class CustomCNN(nn.Module):
    def __init__(self, channels=CHANNELS, dropout: float = 0.3):
        super().__init__()
        blocks, in_ch = OrderedDict(), 3
        for i, out_ch in enumerate(channels, start=1):
            blocks[f"block{i}"] = _block(in_ch, out_ch)
            in_ch = out_ch
        self.features = nn.Sequential(blocks)  # 224 -> 7 after five poolings
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (N, 1)."""
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))
