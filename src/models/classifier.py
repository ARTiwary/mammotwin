"""
Phase 6: transfer-learning classifier, backbone selected via config.yaml
(model.backbone: resnet50 | densenet121 | efficientnet_b0).
"""

import torch.nn as nn
import torchvision.models as models


def build_classifier(config: dict) -> nn.Module:
    model_cfg = config["model"]
    backbone_name = model_cfg["backbone"]
    pretrained = model_cfg.get("pretrained", True)
    num_classes = model_cfg["num_classes"]
    dropout = model_cfg.get("dropout", 0.3)

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    elif backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    elif backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    elif backbone_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                          f"Supported: resnet18, resnet50, densenet121, efficientnet_b0")

    return model