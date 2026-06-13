"""EfficientNet-B0 transfer-learning model (M4).

We take the ImageNet-pretrained EfficientNet-B0 **backbone** and replace its
1000-class ImageNet head with our own 4-class linear head. We then train the head
(Phase A) and fine-tune the whole network (Phase B) ourselves.

CLAUDE.md invariant #3: we do NOT load any pre-finetuned PlantVillage classifier.
The only thing pretrained here is the generic ImageNet feature extractor; every
disease-specific weight is learned by our own pipeline.
"""

from typing import cast

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .labels import CANONICAL_LABELS

WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


def build_model(num_classes: int = len(CANONICAL_LABELS)) -> nn.Module:
    """EfficientNet-B0 with ImageNet weights and a fresh num_classes head."""
    model = efficientnet_b0(weights=WEIGHTS)
    # torchvision's classifier is Sequential(Dropout(p=0.2), Linear(1280, 1000)).
    # Keep the dropout; swap only the final Linear for our 4-class output.
    # cast: nn.Module.__getattr__ is typed as `Tensor | Module`, so we tell the
    # type checker these submodules are the concrete types we know them to be.
    classifier = cast(nn.Sequential, model.classifier)
    in_features = cast(nn.Linear, classifier[1]).in_features
    classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase A: freeze the conv backbone; only the new head trains.

    `model.features` is the convolutional stack. Freezing it means its weights get
    no gradient, so Phase A only adapts the head to our 4 classes — fast and stable
    before we touch the pretrained features.
    """
    for p in cast(nn.Module, model.features).parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Phase B: unfreeze everything for end-to-end fine-tuning at a low LR."""
    for p in model.parameters():
        p.requires_grad = True
