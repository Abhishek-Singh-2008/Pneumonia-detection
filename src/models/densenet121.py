"""Stage 2.1: DenseNet-121 (ImageNet-pretrained) with a one-logit head (owned by Ajit, issue #7).

CheXNet-style transfer learning: take DenseNet-121 pretrained on ImageNet, replace the
1000-class layer by ONE output logit (PNEUMONIA = 1, #21 row 4) and train in two phases:
    Phase A (this issue, #7): backbone frozen, only the new final layer is trained.
    Phase B (issue #8):       backbone unfrozen and fine-tuned with a small learning rate.

Why we wrap torchvision's model instead of using it directly:
- torchvision's DenseNet applies an IN-PLACE functional ReLU after the last feature map, which
  cannot be hooked and would modify the tensor a hook sees. Our model has a named, non-in-place
  `relu` module after `features`, so Aman (Stage 3) can hook it: the post-ReLU 1024 x 7 x 7 map
  is src.config.LAST_CONV_LAYER["densenet121"] = "relu" (requested in the PR #24 review).
- When the backbone is frozen we must also keep its BatchNorm layers in eval mode. The
  shared training loop calls model.train() every epoch, which would otherwise keep
  updating the BatchNorm running statistics of a "frozen" backbone.

Shapes: input (N, 3, 224, 224) -> features (N, 1024, 7, 7) -> pooled (N, 1024) -> logits (N, 1).
"""
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 backbone + dropout + one linear output logit."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        # IMAGENET1K_V1 weights; this is why the shared pipeline uses ImageNet mean/std (#21 row 5).
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        # `features` ends with norm5, a BatchNorm that outputs the final 1024 x 7 x 7 feature map
        # BEFORE the ReLU.
        self.features = backbone.features
        # Named, NOT in-place ReLU that follows norm5. Its output is the post-ReLU 1024 x 7 x 7 map that
        # Grad-CAM should use (hook `relu`, see src.config.LAST_CONV_LAYER["densenet121"]; PR #24 review).
        # It has no parameters, so checkpoints saved before it was added still load unchanged.
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        # The only new layer: 1024 pooled features -> 1 logit (sigmoid gives P(PNEUMONIA)).
        self.classifier = nn.Linear(1024, 1)
        self.backbone_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (N, 1)."""
        x = self.features(x)
        x = self.relu(x)  # a module (not torch.relu) so Grad-CAM can hook it; not in-place, output stays intact
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))

    def freeze_backbone(self) -> None:
        """Phase A: only the classifier gets gradients."""
        for p in self.features.parameters():
            p.requires_grad = False
        self.backbone_frozen = True
        self.features.eval()

    def unfreeze_backbone(self) -> None:
        """Phase B: train the whole network."""
        for p in self.features.parameters():
            p.requires_grad = True
        self.backbone_frozen = False

    def train(self, mode: bool = True):
        """Like nn.Module.train(), but a frozen backbone always stays in eval mode
        (fixed BatchNorm statistics, no dropout-like behaviour changes)."""
        super().train(mode)
        if self.backbone_frozen:
            self.features.eval()
        return self

    def trainable_parameters(self):
        """Parameters the optimizer should update (empty backbone part while frozen)."""
        return [p for p in self.parameters() if p.requires_grad]
