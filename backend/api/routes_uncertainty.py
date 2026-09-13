"""Phase 15: GET /uncertainty/{image_id} -- re-fetch the MC-Dropout
uncertainty result for an already-uploaded image without re-uploading it."""

from fastapi import APIRouter, HTTPException

from backend.services.inference_service import get_service
from backend.schemas.responses import UncertaintyResult

router = APIRouter()


@router.get("/uncertainty/{image_id}", response_model=UncertaintyResult)
async def get_uncertainty(image_id: str):
    cached = get_service().get_cached(image_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="Unknown image_id -- run /predict/upload first.")
    return cached["uncertainty"]
