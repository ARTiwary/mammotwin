"""
Phase 15: Pydantic response schemas for the FastAPI dashboard backend.

Every schema here mirrors a stage of the pipeline described in the project
plan (Quality Check -> Preprocessing -> Localization -> Segmentation ->
Classification -> Uncertainty -> Explainability -> Dashboard), and every
schema that carries a model-derived claim also carries the disclaimer
context needed to present it responsibly -- this is a research prototype,
not a diagnostic system, and the API contract should make that hard to
lose track of on the frontend.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


RESEARCH_DISCLAIMER = (
    "MammoTwin is an academic research prototype. Its outputs are not a "
    "medical diagnosis and must not be used to make clinical decisions."
)


class QualityResult(BaseModel):
    passed: bool
    is_blank: bool
    is_low_contrast: bool
    breast_area_too_small: Optional[bool] = None


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    score: float


class LocalizationResult(BaseModel):
    available: bool
    boxes: List[BoundingBox] = Field(default_factory=list)
    overlay_png_base64: Optional[str] = None
    message: Optional[str] = None


class ClassificationResult(BaseModel):
    predicted_label: str  # "benign" | "malignant"
    probability_malignant: float
    operating_threshold: float
    target_sensitivity: Optional[float] = None
    model_name: str


class UncertaintyResult(BaseModel):
    mean_prob_malignant: float
    std_prob_malignant: float
    confidence: float
    needs_review: bool
    review_threshold: float
    method: str = "mc_dropout"
    n_passes: int


class ExplainabilityResult(BaseModel):
    overlay_png_base64: str
    target_class: str = "malignant"
    disclaimer: str = (
        "This heatmap shows where the model's evidence for malignancy was "
        "concentrated. It is a model explanation, not proof of cancer, and "
        "does not by itself validate that the prediction is correct."
    )


class SegmentationResult(BaseModel):
    available: bool
    overlay_png_base64: Optional[str] = None
    used_localization_box: bool = False
    message: Optional[str] = None


class FullPredictionResponse(BaseModel):
    image_id: str
    quality: QualityResult
    localization: LocalizationResult
    classification: ClassificationResult
    uncertainty: UncertaintyResult
    explainability: ExplainabilityResult
    segmentation: SegmentationResult
    disclaimer: str = RESEARCH_DISCLAIMER


class LongitudinalComparisonResponse(BaseModel):
    is_simulated: bool
    message: str
    prior_probability_malignant: Optional[float] = None
    current_probability_malignant: Optional[float] = None
    score_change: Optional[float] = None
    overlay_png_base64: Optional[str] = None
    disclaimer: str = RESEARCH_DISCLAIMER
