"""
Phase 15: FastAPI application entry point.

Run from the project root with:
    uvicorn backend.main:app --reload --port 8000

The React dev server (frontend/, default port 5173) talks to this over
CORS -- see origins below. In a real deployment you'd restrict this to
the actual frontend origin; for a local research demo, localhost only.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes_predict, routes_explain, routes_uncertainty, routes_longitudinal
from backend.services.inference_service import get_service

app = FastAPI(
    title="MammoTwin API",
    description="Research prototype API for mammogram lesion analysis. "
                 "Not a diagnostic system -- see /docs for the full schema.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_predict.router, tags=["predict"])
app.include_router(routes_explain.router, tags=["explain"])
app.include_router(routes_uncertainty.router, tags=["uncertainty"])
app.include_router(routes_longitudinal.router, tags=["longitudinal"])


@app.on_event("startup")
def load_models_on_startup():
    # Loads every checkpoint once, at process start, instead of on the
    # first request -- so the first user doesn't pay a multi-second
    # cold-start cost, and so a missing/broken checkpoint fails loudly at
    # startup rather than deep inside a request.
    get_service()


@app.get("/health")
def health():
    service = get_service()
    return {
        "status": "ok",
        "device": str(service.device),
        "classifier_loaded": service.classifier is not None,
        "detector_loaded": service.detector is not None,
        "segmentation_loaded": service.segmentation_model is not None,
        "disclaimer": "MammoTwin is a research prototype, not a diagnostic system.",
    }
