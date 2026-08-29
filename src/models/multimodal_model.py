"""
Phase 13: multimodal fusion model — a CNN image embedding fused with an
MLP-processed tabular embedding, per the plan's architecture:
  "CNN extracts an image embedding. An MLP ... processes structured
  variables. Fuse the representations. Train a final prediction head."
"""

import torch
import torch.nn as nn

from src.models.classifier import build_classifier


def get_image_embedding_backbone(config: dict):
    """
    Builds the same backbone as build_classifier(), then strips off its
    final classification layer so it outputs a raw embedding instead of
    class logits. Returns (backbone_module, embedding_dim).
    """
    model = build_classifier(config)
    backbone_name = config["model"]["backbone"]

    if backbone_name in ("resnet18", "resnet50"):
        embedding_dim = model.fc[1].in_features  # Sequential(Dropout, Linear) from build_classifier
        model.fc = nn.Identity()
    elif backbone_name == "densenet121":
        embedding_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        embedding_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone for embedding extraction: {backbone_name}")

    return model, embedding_dim


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.mlp(x)


class MultimodalFusionModel(nn.Module):
    def __init__(self, image_backbone, image_embedding_dim, tabular_input_dim,
                 tabular_hidden_dims=(64, 32), num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.image_backbone = image_backbone
        self.tabular_mlp = TabularMLP(tabular_input_dim, tabular_hidden_dims, dropout=dropout)

        fused_dim = image_embedding_dim + self.tabular_mlp.output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, tabular):
        image_embedding = self.image_backbone(image)
        tabular_embedding = self.tabular_mlp(tabular)
        fused = torch.cat([image_embedding, tabular_embedding], dim=1)
        return self.fusion_head(fused)


def build_multimodal_model(config: dict, tabular_input_dim: int) -> MultimodalFusionModel:
    backbone, embedding_dim = get_image_embedding_backbone(config)
    return MultimodalFusionModel(
        image_backbone=backbone,
        image_embedding_dim=embedding_dim,
        tabular_input_dim=tabular_input_dim,
        num_classes=config["model"]["num_classes"],
        dropout=config["model"].get("dropout", 0.3),
    )