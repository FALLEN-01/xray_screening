"""
model.py -- ResNet-50 fine-tuned for X-ray baggage classification.
Two-phase strategy:
    Phase 1 (freeze_epochs): train only the new classifier head
    Phase 2 (remaining epochs): unfreeze all layers, lower LR
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class XRayClassifier(nn.Module):
    """
    ResNet-50 with a custom 2-layer classification head.
    Input:  (B, 3, 224, 224)  -- ImageNet-normalised
    Output: (B, num_classes)  -- raw logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4,
                 pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original FC head; keep everything up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features   # 2048

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Reference to last conv block for Grad-CAM
        self.gradcam_layer = backbone.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)          # (B, 2048, 1, 1)
        feat = feat.flatten(1)           # (B, 2048)
        return self.classifier(feat)     # (B, num_classes)

    # -- Two-phase fine-tuning helpers ----------------------------------------

    def freeze_backbone(self):
        """Phase 1: only train the classifier head."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("[Model] Backbone FROZEN -- training head only.")

    def unfreeze_backbone(self):
        """Phase 2: full fine-tuning with lower LR."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] Backbone UNFROZEN -- full fine-tuning.")

    def get_gradcam_target_layers(self):
        """Returns target layer list for pytorch-grad-cam."""
        # layer4 is the last residual block of ResNet-50
        return [self.features[-1][-1]]   # layer4 -> last BasicBlock/Bottleneck

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(cfg: dict) -> XRayClassifier:
    model = XRayClassifier(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    )
    params = model.count_params()
    print(f"[Model] ResNet-50  |  total={params['total']:,}  "
          f"trainable={params['trainable']:,}")
    return model
