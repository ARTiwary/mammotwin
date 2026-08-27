"""
Phase 8: segmentation loss and metrics, per the project plan's guidance to
try Dice loss or a Dice/BCE combination, evaluated with Dice and IoU.
"""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss. BCE gives per-pixel gradient signal even
    when overlap is currently zero (Dice alone can struggle to get started
    when the initial prediction doesn't overlap the target at all); Dice
    directly optimizes the overlap metric we actually care about.

    Lesion pixels are typically under 1% of a mammogram — without
    correcting for this, plain BCE lets the model achieve deceptively low
    loss by predicting "background everywhere," and every pixel probability
    can sit below the 0.5 threshold indefinitely (Dice stuck at exactly
    0.0 even as loss keeps falling). pos_weight is computed dynamically
    per-batch (mirrors the inverse-frequency class weighting used for the
    classifier in Phase 6) to counteract this.
    """
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5, max_pos_weight: float = 100.0):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.max_pos_weight = max_pos_weight

    def forward(self, logits, targets):
        with torch.no_grad():
            num_pos = targets.sum()
            num_neg = targets.numel() - num_pos
            pos_weight = torch.clamp(num_neg / (num_pos + 1e-6),
                                      max=self.max_pos_weight).to(logits.device)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_coefficient(pred_binary: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """pred_binary and target are both {0,1} tensors of the same shape."""
    pred_flat = pred_binary.reshape(-1).float()
    target_flat = target.reshape(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def iou_score(pred_binary: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred_flat = pred_binary.reshape(-1).float()
    target_flat = target.reshape(-1).float()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor,
                                  threshold: float = 0.5) -> dict:
    """logits: raw model output (pre-sigmoid), targets: {0,1} ground truth.
    Computes per-sample Dice/IoU then averages across the batch."""
    probs = torch.sigmoid(logits)
    preds_binary = (probs >= threshold).float()

    dices, ious = [], []
    for i in range(logits.size(0)):
        dices.append(dice_coefficient(preds_binary[i], targets[i]))
        ious.append(iou_score(preds_binary[i], targets[i]))

    return {"dice": sum(dices) / len(dices), "iou": sum(ious) / len(ious)}