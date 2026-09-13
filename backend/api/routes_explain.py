"""Phase 15: GET /explain/{image_id} -- re-fetch the Grad-CAM overlay for an
already-uploaded image (see routes_predict.py) without re-uploading it."""

from fastapi import APIRouter, HTTPException

from backend.services.inference_service import get_service
from backend.schemas.responses import ExplainabilityResult

router = APIRouter()


@router.get("/explain/{image_id}", response_model=ExplainabilityResult)
async def get_explanation(image_id: str):
    cached = get_service().get_cached(image_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Unknown image_id -- run /predict/upload first.")
    return cached["explainability"]
