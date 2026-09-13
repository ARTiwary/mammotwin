"""
Phase 15: POST /predict/upload -- runs the ENTIRE pipeline (quality check,
localization, classification, uncertainty, explainability, segmentation)
on one uploaded mammogram in a single request, mirroring the project
plan's own "Recommended minimum final demo" flow. Returns everything the
dashboard needs to render at once, plus an image_id the frontend can use
to re-fetch individual pieces later (see routes_explain.py /
routes_uncertainty.py) without re-uploading.
"""

import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.inference_service import get_service
from backend.schemas.responses import FullPredictionResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".dicom"}


@router.post("/predict/upload", response_model=FullPredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    service = get_service()
    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = service.run_full_pipeline(tmp_path)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
