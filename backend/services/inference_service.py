"""
Phase 15: the dashboard's inference engine.

Loads every available trained checkpoint ONCE at process startup (this is
a research demo running as a single process, not a production model
server -- one set of weights in memory is intentional and sufficient) and
exposes a single run_full_pipeline() call that mirrors the project's own
described workflow:

    Upload -> Quality Check -> Preprocessing -> Localization ->
    Segmentation -> Classification -> Uncertainty -> Explainability

Every stage that depends on a checkpoint that isn't available degrades
gracefully (available=False + an explanatory message) rather than
crashing the whole request -- e.g. if no detector was trained, you still
get classification + uncertainty + explainability, just no localization
box, and segmentation reports why it couldn't run (it needs a
localization box to know where to crop the patch it was trained on).
"""

import os
import io
import json
import base64
import uuid
from typing import Optional

import numpy as np
import cv2
import torch

from src.utils.config import load_config, set_global_seed
from src.preprocessing.basic_preprocess import preprocess_image
from src.data.image_io import load_image
from src.models.classifier import build_classifier
from src.models.detector import build_detector
from src.models.segmentation_model import build_segmentation_model
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.uncertainty.mc_dropout import mc_dropout_predict

MALIGNANT_IDX = 1
LABELS = ["benign", "malignant"]


def _to_tensor_3ch(processed: np.ndarray, device) -> torch.Tensor:
    """(H, W) float32 [0,1] -> (1, 3, H, W) on device, matching every
    training script's tensor prep exactly (single channel repeated x3)."""
    t = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
    t = t.repeat(3, 1, 1).float().unsqueeze(0)
    return t.to(device)


def _encode_png_base64(rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode overlay image as PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# Public aliases -- these two helpers are reused by other route modules
# (e.g. routes_longitudinal.py) that need the exact same tensor prep /
# encoding the main pipeline uses, so they're intentionally exported
# rather than treated as private to this module.
to_tensor_3ch = _to_tensor_3ch
encode_png_base64 = _encode_png_base64


def _read_registry(models_dir: str) -> dict:
    path = os.path.join(models_dir, "registry.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path) as f:
        return json.load(f)


def _best_entry(registry: dict, phase_name: str = None, fallback_prefix: str = None,
                 metric_key: str = "val_auc") -> Optional[dict]:
    matching = {k: v for k, v in registry.items() if phase_name is not None and v.get("phase") == phase_name}
    if not matching and fallback_prefix:
        matching = {k: v for k, v in registry.items() if k.startswith(fallback_prefix)}
    if not matching:
        return None
    best_key = max(matching, key=lambda k: matching[k].get(metric_key, -1))
    return matching[best_key]


class MammoTwinService:
    def __init__(self):
        self.config = load_config()
        set_global_seed(self.config["project"]["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models_dir = self.config["paths"]["models_dir"]
        metadata_dir = self.config["paths"]["data_metadata"]
        registry = _read_registry(models_dir)

        self.uncertainty_cfg = self.config.get("uncertainty", {"mc_dropout_passes": 20, "review_threshold": 0.6})

        # --- Classifier (required for almost everything the dashboard does) ---
        self.classifier = None
        self.classifier_config = None
        self.operating_threshold = 0.5
        self.target_sensitivity = None
        self.gradcam = None
        self._load_classifier(models_dir, metadata_dir, registry)

        # --- Detector (optional) ---
        self.detector = None
        self.detector_config = None
        self._load_detector(models_dir, registry)

        # --- Segmentation (optional, needs a localization box at inference time) ---
        self.segmentation_model = None
        self.segmentation_config = None
        self.segmentation_patch_size = None
        self._load_segmentation(models_dir, registry)

        # In-memory cache so /explain, /uncertainty etc. can be re-fetched for
        # an already-uploaded image without re-uploading + re-preprocessing it.
        self._cache = {}

        print(f"[MammoTwinService] device={self.device}  "
              f"classifier={'loaded' if self.classifier else 'MISSING'}  "
              f"detector={'loaded' if self.detector else 'not available'}  "
              f"segmentation={'loaded' if self.segmentation_model else 'not available'}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_classifier(self, models_dir, metadata_dir, registry):
        entry = _best_entry(registry, fallback_prefix="baseline_", metric_key="val_auc")
        if entry is None:
            print("[MammoTwinService] WARNING: no baseline classifier found in registry.json. "
                  "The dashboard cannot classify anything until Phase 6 has been trained.")
            return

        checkpoint = torch.load(entry["checkpoint_path"], map_location=self.device, weights_only=False)
        self.classifier_config = checkpoint["config"]
        self.classifier = build_classifier(self.classifier_config).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()
        self.gradcam = GradCAM(self.classifier, backbone_name=self.classifier_config["model"]["backbone"])

        thresholds_path = os.path.join(metadata_dir, "operating_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            info = thresholds.get("whole_image_baseline")
            if info:
                self.operating_threshold = info["operating_threshold"]
                self.target_sensitivity = info["target_sensitivity"]
                return
        print("[MammoTwinService] No operating_thresholds.json entry for whole_image_baseline -- "
              "falling back to the default 0.5 cutoff. Run select_operating_thresholds.py for a "
              "sensitivity-appropriate operating point.")

    def _load_detector(self, models_dir, registry):
        entry = _best_entry(registry, phase_name="7_localization", metric_key="val_map")
        if entry is None:
            return
        checkpoint = torch.load(entry["checkpoint_path"], map_location=self.device, weights_only=False)
        self.detector_config = checkpoint["config"]
        self.detector = build_detector(self.detector_config).to(self.device)
        self.detector.load_state_dict(checkpoint["model_state_dict"])
        self.detector.eval()

    def _load_segmentation(self, models_dir, registry):
        entry = _best_entry(registry, phase_name="8_segmentation", metric_key="val_dice")
        if entry is None:
            return
        checkpoint = torch.load(entry["checkpoint_path"], map_location=self.device, weights_only=False)
        self.segmentation_config = checkpoint["config"]
        self.segmentation_patch_size = checkpoint.get("patch_size")
        self.segmentation_model = build_segmentation_model(self.segmentation_config).to(self.device)
        self.segmentation_model.load_state_dict(checkpoint["model_state_dict"])
        self.segmentation_model.eval()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _run_localization(self, img: np.ndarray, score_threshold: float = 0.3, max_boxes: int = 3):
        if self.detector is None:
            return {"available": False, "boxes": [], "overlay_png_base64": None,
                    "message": "No detector checkpoint available."}

        result = preprocess_image(img, self.detector_config, run_quality_gate=False)
        processed = result["processed"]
        tensor = _to_tensor_3ch(processed, self.device).squeeze(0)  # detector wants a LIST of (3,H,W)
        with torch.no_grad():
            pred = self.detector([tensor])[0]

        boxes_out = []
        scores = pred["scores"].cpu().numpy()
        boxes = pred["boxes"].cpu().numpy()
        order = np.argsort(-scores)[:max_boxes]
        for i in order:
            if scores[i] < score_threshold:
                continue
            x1, y1, x2, y2 = boxes[i]
            boxes_out.append({"x": float(x1), "y": float(y1), "w": float(x2 - x1),
                               "h": float(y2 - y1), "score": float(scores[i])})

        # Render boxes onto the preprocessed grayscale image server-side, so
        # every pipeline stage (localization, segmentation, explainability)
        # hands the frontend a ready-to-display image in the same format,
        # rather than making the frontend redo coordinate math per stage.
        base_uint8 = (np.clip(processed, 0, 1) * 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)
        for b in boxes_out:
            x0, y0 = int(b["x"]), int(b["y"])
            x1, y1 = int(b["x"] + b["w"]), int(b["y"] + b["h"])
            cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        message = None if boxes_out else "No region scored above the detection threshold."
        return {"available": True, "boxes": boxes_out, "overlay_png_base64": _encode_png_base64(overlay_rgb),
                "message": message}

    def _run_segmentation(self, img: np.ndarray, localization_boxes: list):
        if self.segmentation_model is None:
            return {"available": False, "overlay_png_base64": None, "used_localization_box": False,
                    "message": "No segmentation checkpoint available."}
        if not localization_boxes:
            return {"available": False, "overlay_png_base64": None, "used_localization_box": False,
                    "message": "Segmentation needs a localization box to know where to crop the "
                               "patch it was trained on -- no region was detected for this image."}

        result = preprocess_image(img, self.segmentation_config, run_quality_gate=False)
        processed = result["processed"]
        image_size = tuple(self.segmentation_config["preprocessing"]["image_size"])

        ps = self.segmentation_patch_size
        if ps is None:
            patch_img = processed
        else:
            # Center the patch on the TOP detected box, since at real
            # inference time (unlike training/eval) there is no ground-
            # truth mask to center on -- this is the realistic integration
            # of localization -> segmentation the offline eval couldn't
            # exercise, because it always had the GT mask available.
            box = localization_boxes[0]
            cy = int(box["y"] + box["h"] / 2)
            cx = int(box["x"] + box["w"] / 2)
            H, W = processed.shape
            ps_eff = min(ps, H, W)
            half = ps_eff // 2
            y0 = max(0, min(cy - half, H - ps_eff))
            x0 = max(0, min(cx - half, W - ps_eff))
            patch_img = processed[y0:y0 + ps_eff, x0:x0 + ps_eff]

        tensor = torch.from_numpy(np.ascontiguousarray(patch_img)).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.segmentation_model(tensor)
            prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

        overlay = overlay_heatmap(patch_img, prob_mask, alpha=0.4)
        return {"available": True, "overlay_png_base64": _encode_png_base64(overlay),
                "used_localization_box": ps is not None, "message": None}

    def run_full_pipeline(self, image_path: str) -> dict:
        if self.classifier is None:
            raise RuntimeError("No classifier checkpoint is loaded -- train Phase 6 before using the dashboard.")

        img = load_image(image_path)

        # --- Quality + preprocessing (classifier's own config governs this) ---
        result = preprocess_image(img, self.classifier_config, run_quality_gate=True)
        quality = result["quality"]
        tensor = _to_tensor_3ch(result["processed"], self.device)

        # --- Classification ---
        with torch.no_grad():
            probs = torch.softmax(self.classifier(tensor), dim=1)[0]
        prob_malignant = float(probs[MALIGNANT_IDX].item())
        pred_label = LABELS[1] if prob_malignant >= self.operating_threshold else LABELS[0]
        classification = {
            "predicted_label": pred_label, "probability_malignant": prob_malignant,
            "operating_threshold": self.operating_threshold, "target_sensitivity": self.target_sensitivity,
            "model_name": "whole_image_baseline",
        }

        # --- Uncertainty (MC-Dropout) ---
        n_passes = self.uncertainty_cfg.get("mc_dropout_passes", 20)
        mean_prob, std_prob = mc_dropout_predict(self.classifier, tensor, n_passes=n_passes)
        confidence = max(mean_prob, 1 - mean_prob)
        review_threshold = self.uncertainty_cfg.get("review_threshold", 0.6)
        uncertainty = {
            "mean_prob_malignant": mean_prob, "std_prob_malignant": std_prob,
            "confidence": confidence, "needs_review": confidence < review_threshold,
            "review_threshold": review_threshold, "n_passes": n_passes,
        }

        # --- Explainability (always the malignant-class Grad-CAM -- see docs/RESULTS_AND_LIMITATIONS.md) ---
        heatmap, _, _ = self.gradcam.generate(tensor, target_class=MALIGNANT_IDX)
        overlay = overlay_heatmap(result["processed"], heatmap)
        explainability = {"overlay_png_base64": _encode_png_base64(overlay)}

        # --- Localization + Segmentation ---
        localization = self._run_localization(img)
        segmentation = self._run_segmentation(img, localization["boxes"])

        image_id = str(uuid.uuid4())
        response = {
            "image_id": image_id, "quality": quality, "localization": localization,
            "classification": classification, "uncertainty": uncertainty,
            "explainability": explainability, "segmentation": segmentation,
        }
        self._cache[image_id] = response
        return response

    def get_cached(self, image_id: str) -> Optional[dict]:
        return self._cache.get(image_id)


# Single shared instance for the FastAPI process (see backend/main.py).
_service_instance: Optional[MammoTwinService] = None


def get_service() -> MammoTwinService:
    global _service_instance
    if _service_instance is None:
        _service_instance = MammoTwinService()
    return _service_instance
