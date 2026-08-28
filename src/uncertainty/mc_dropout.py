"""
Phase 11: Monte Carlo Dropout for uncertainty estimation (Gal & Ghahramani,
2016) — an experimental, lightweight uncertainty method, per the project
plan's own framing.

Standard inference disables Dropout (model.eval()). MC-Dropout instead
keeps Dropout layers ACTIVE at inference time and runs the same input
through the model many times; because Dropout randomly zeroes different
units each pass, the predictions vary — the spread (std) of predicted
probabilities across passes is used as an uncertainty estimate. BatchNorm
layers are deliberately left in eval() mode throughout (their running
statistics should NOT fluctuate per-pass), which is why we set only
nn.Dropout submodules to train() rather than the whole model.
"""

import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Call model.eval() FIRST, then this — it selectively re-enables just
    the Dropout layers, leaving BatchNorm and everything else in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, input_tensor: torch.Tensor, n_passes: int = 20):
    """
    input_tensor: (1, C, H, W), a single image.
    Returns (mean_prob, std_prob) for the POSITIVE class (index 1,
    "malignant"), aggregated over n_passes stochastic forward passes.
    """
    model.eval()
    enable_mc_dropout(model)

    probs = []
    for _ in range(n_passes):
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
        probs.append(prob)

    probs_tensor = torch.tensor(probs)
    return probs_tensor.mean().item(), probs_tensor.std().item()