"""
Freezes early layers of a torchvision ResNet backbone, keeping only the
later layers (and the classification head) trainable.

Why: a pretrained ResNet's early layers already encode general-purpose
visual features (edges, textures, simple shapes) learned from ImageNet's
millions of images -- these don't need to change for mammography. Only
the later layers, which encode more task-specific/abstract features, need
to adapt. Freezing the early layers cuts the model's effective trainable
capacity substantially, which directly reduces its ability to memorize a
small training set (the root cause of the overfitting measured in this
project: train ROC-AUC 0.82-0.96 vs test ROC-AUC 0.69-0.76).

Usage -- add these two lines right after building the model and moving it
to device, and BEFORE creating the optimizer (the optimizer must only see
parameters that still require gradients):

    model = build_classifier(config).to(device)
    from src.utils.regularization import freeze_resnet_early_layers
    n_frozen, n_trainable = freeze_resnet_early_layers(model, unfreeze_from="layer3")
    print(f"Froze {n_frozen:,} params, {n_trainable:,} remain trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # <-- IMPORTANT: filter here too
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

Both changes matter: freezing alone does nothing if the optimizer still
tries to update frozen parameters (Adam will error or waste memory
tracking momentum for parameters that never move) -- always pass a
filtered parameter list to the optimizer after freezing.
"""

from typing import Tuple

import torch.nn as nn

# Freeze everything up to (not including) this layer by default. ResNet's
# stages, in order: conv1/bn1 (stem) -> layer1 -> layer2 -> layer3 -> layer4 -> fc.
# "layer3" is a common, safe default: freezes the stem + layer1 + layer2
# (the most generic, least task-specific features), leaves layer3, layer4,
# and the classification head trainable.
_RESNET_STAGE_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def freeze_resnet_early_layers(model: nn.Module, unfreeze_from: str = "layer3") -> Tuple[int, int]:
    """
    Freezes every ResNet stage before `unfreeze_from` (exclusive), plus
    always leaves the final classification head (whatever comes after the
    last conv stage -- typically named 'fc') trainable regardless.

    Works whether `model` IS a torchvision ResNet directly, or wraps one
    as `model.backbone` / `model.resnet` (checked automatically) --
    matching how build_classifier() may have structured things.

    Returns (n_frozen_params, n_trainable_params) so the caller can print
    a sanity-check confirmation.
    """
    if unfreeze_from not in _RESNET_STAGE_ORDER:
        raise ValueError(f"unfreeze_from must be one of {_RESNET_STAGE_ORDER}, got {unfreeze_from!r}")

    # Find the actual resnet module, whether build_classifier() returns it
    # directly or wraps it under a named attribute (build_multimodal_model()
    # uses 'image_backbone').
    resnet = model
    for attr in ("backbone", "resnet", "encoder", "model", "image_backbone"):
        if hasattr(model, attr) and hasattr(getattr(model, attr), "layer1"):
            resnet = getattr(model, attr)
            break

    if not hasattr(resnet, "layer1"):
        raise ValueError(
            "Could not find a torchvision-ResNet-shaped module (no 'layer1' attribute found "
            "on the model or on model.backbone/.resnet/.encoder/.model). If build_classifier() "
            "wraps the resnet under a different attribute name, freeze it manually instead:\n"
            "    for name, param in model.named_parameters():\n"
            "        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:\n"
            "            param.requires_grad = False"
        )

    freeze_stage_names = _RESNET_STAGE_ORDER[:_RESNET_STAGE_ORDER.index(unfreeze_from)]

    n_frozen, n_trainable = 0, 0
    for name, param in resnet.named_parameters():
        stage = name.split(".")[0]  # e.g. "layer2.0.conv1.weight" -> "layer2"
        if stage in freeze_stage_names:
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            param.requires_grad = True
            n_trainable += param.numel()

    # Always keep the classification head trainable, wherever it lives
    # (on `model` itself if build_classifier() replaced resnet.fc with a
    # custom head attached to the outer wrapper).
    for name, param in model.named_parameters():
        if "fc" in name or "classifier" in name or "head" in name:
            if not param.requires_grad:
                param.requires_grad = True
                n_frozen -= param.numel()
                n_trainable += param.numel()

    return n_frozen, n_trainable