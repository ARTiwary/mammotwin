"""
Phase 15 / Phase 12: POST /longitudinal/compare.

The project plan is explicit that CBIS-DDSM (and, by extension, any two
images a dashboard user happens to upload) is NOT a genuine longitudinal
pairing -- a real prior/current comparison requires the same patient,
matched laterality/view, and proper image registration. Since the
dashboard has no way to verify any of that for two arbitrary uploads,
every response from this endpoint is unconditionally marked
is_simulated=True with an explanation, per the project's own rule:
"If no genuine longitudinal data are available, use simulated pairs only
to demonstrate the software and explicitly label them simulated."

This demonstrates the SOFTWARE MECHANICS of a longitudinal comparison
(score change over time, a naive visual diff) -- it is not, and must
never be presented as, a real change-detection result.
"""

import os
import tempfile

import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.inference_service import get_service, to_tensor_3ch, encode_png_base64
from backend.schemas.responses import LongitudinalComparisonResponse
from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import preprocess_image

router = APIRouter()


def _classify_one(service, tmp_path: str):
    import torch
    img = load_image(tmp_path)
    result = preprocess_image(img, service.classifier_config, run_quality_gate=False)
    tensor = to_tensor_3ch(result["processed"], service.device)
    with torch.no_grad():
        probs = torch.softmax(service.classifier(tensor), dim=1)[0]
    return float(probs[1].item()), result["processed"]


@router.post("/longitudinal/compare", response_model=LongitudinalComparisonResponse)
async def compare_longitudinal(prior: UploadFile = File(...), current: UploadFile = File(...)):
    service = get_service()
    if service.classifier is None:
        raise HTTPException(status_code=503, detail="No classifier checkpoint is loaded.")

    tmp_paths = []
    try:
        for upload in (prior, current):
            ext = os.path.splitext(upload.filename or "")[1] or ".png"
            contents = await upload.read()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                tmp_paths.append(tmp.name)

        prior_prob, prior_img = _classify_one(service, tmp_paths[0])
        current_prob, current_img = _classify_one(service, tmp_paths[1])
        score_change = current_prob - prior_prob

        # A NAIVE, unregistered pixel-difference visualization only -- this
        # is not a real change-detection result (no registration/alignment
        # is performed), which is exactly why every field here is qualified
        # as simulated.
        diff = np.abs(current_img.astype(np.float32) - prior_img.astype(np.float32))
        if diff.max() > 0:
            diff = diff / diff.max()
        diff_uint8 = (diff * 255).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
        diff_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

        return LongitudinalComparisonResponse(
            is_simulated=True,
            message=(
                "This is a SIMULATED demonstration of the longitudinal comparison "
                "mechanism, using two independently uploaded images with no "
                "verified patient identity, laterality match, or image "
                "registration. It illustrates the software's mechanics only and "
                "is not a real change-detection result."
            ),
            prior_probability_malignant=prior_prob,
            current_probability_malignant=current_prob,
            score_change=score_change,
            overlay_png_base64=encode_png_base64(diff_rgb),
        )
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)
