"""
Phase 10: Grad-CAM, implemented directly rather than via the `grad-cam`
package — following the lesson from Phases 7-8 (fewer third-party
dependencies means fewer ways an environment-specific issue can derail
things), and it's simple enough to describe exactly in a methods section.

Grad-CAM (Selvaraju et al. 2017): for a target class, backpropagate from
that class's score to a chosen convolutional layer's activations. Each
channel's gradient, globally average-pooled, becomes that channel's
"importance weight." A weighted sum of the activation maps, ReLU'd (we only
care about features that POSITIVELY support the target class) and resized
to the input's spatial size, is the heatmap.

IMPORTANT: a Grad-CAM heatmap shows where the model's evidence for its
prediction was concentrated — it is a model explanation, not proof of
cancer, and does not by itself validate that the prediction is correct.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Reasonable default target layer per backbone — the last spatial feature
# map before global pooling. If you use a backbone not listed here, pass
# target_layer explicitly to GradCAM().
DEFAULT_TARGET_LAYERS = {
    "resnet18": lambda model: model.layer4[-1],
    "resnet50": lambda model: model.layer4[-1],
    "densenet121": lambda model: model.features[-1],
    "efficientnet_b0": lambda model: model.features[-1],
}


class GradCAM:
    def __init__(self, model, target_layer=None, backbone_name: str = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            if backbone_name is None or backbone_name not in DEFAULT_TARGET_LAYERS:
                raise ValueError(
                    "Pass target_layer explicitly, or a recognized backbone_name "
                    f"(one of {list(DEFAULT_TARGET_LAYERS.keys())})."
                )
            target_layer = DEFAULT_TARGET_LAYERS[backbone_name](model)

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        input_tensor: a SINGLE image, shape (1, C, H, W), on the same device
                      as the model.
        target_class: class index to explain. If None, uses the model's own
                      top prediction.

        Returns (heatmap, predicted_class, predicted_prob) — heatmap is a
        (H, W) float32 array in [0, 1], resized to input_tensor's H, W.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weight.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        heatmap = F.relu(weighted_activations).squeeze().cpu().numpy()

        # Normalize to [0, 1]. A near-zero max means the model found almost
        # nothing supporting this class anywhere — return a blank heatmap
        # rather than dividing by ~zero and amplifying noise.
        if heatmap.max() > 1e-8:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros_like(heatmap)

        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(heatmap, (target_w, target_h))

        predicted_prob = float(probs[0, target_class].item())
        return heatmap_resized, target_class, predicted_prob


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    image: grayscale, float32 [0,1] or uint8, shape (H, W)
    heatmap: float32 [0,1], shape (H, W), same size as image
    Returns an RGB uint8 image with the heatmap blended on top using a
    standard "jet" colormap (red = high relevance, blue = low).
    """
    if image.dtype != np.uint8:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)