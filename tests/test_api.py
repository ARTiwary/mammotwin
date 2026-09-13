"""
Phase 16 / Phase 15: HTTP-level tests for the FastAPI dashboard backend
(backend/main.py). Per the Phase 16 checklist -- "Test invalid/corrupt
uploads" and "Test the complete upload -> prediction -> explanation
workflow" -- these exercise real request/response/validation behavior
against a FAKE MammoTwinService (no real checkpoints or torch compute
needed), so they test the API CONTRACT: status codes, schema shapes,
error handling -- not model correctness (that's test_inference.py's job).

NOTE: backend.main transitively imports torch (via inference_service.py
-> classifier.py/detector.py/segmentation_model.py), so this whole file
is skipped in any environment without torch installed, same as the
torch-dependent tests in test_inference.py.
"""
import base64

import numpy as np
import pytest

pytest.importorskip("torch", reason="backend.main transitively requires torch to import")

import cv2
from fastapi.testclient import TestClient

import backend.services.inference_service as svc_module
from backend.main import app


def _fake_png_base64():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class FakeService:
    """A stand-in for MammoTwinService that returns canned, schema-valid
    results instantly -- no checkpoints, no GPU, no real inference."""

    def __init__(self, classifier_loaded=True, detector_loaded=True, segmentation_loaded=True):
        self.classifier = object() if classifier_loaded else None
        self.detector = object() if detector_loaded else None
        self.segmentation_model = object() if segmentation_loaded else None
        self.device = "cpu"
        self._cache = {}

    def run_full_pipeline(self, image_path: str) -> dict:
        if self.classifier is None:
            raise RuntimeError("No classifier checkpoint is loaded.")
        response = {
            "image_id": "fake-image-id-0001",
            "quality": {"passed": True, "is_blank": False, "is_low_contrast": False,
                        "breast_area_too_small": False},
            "localization": {"available": True,
                              "boxes": [{"x": 10.0, "y": 10.0, "w": 40.0, "h": 40.0, "score": 0.87}],
                              "overlay_png_base64": _fake_png_base64(), "message": None},
            "classification": {"predicted_label": "benign", "probability_malignant": 0.12,
                                "operating_threshold": 0.16, "target_sensitivity": 0.9,
                                "model_name": "whole_image_baseline"},
            "uncertainty": {"mean_prob_malignant": 0.13, "std_prob_malignant": 0.02,
                             "confidence": 0.87, "needs_review": False,
                             "review_threshold": 0.6, "n_passes": 20},
            "explainability": {"overlay_png_base64": _fake_png_base64(), "target_class": "malignant",
                                "disclaimer": "This heatmap is a model explanation, not proof of cancer."},
            "segmentation": {"available": True, "overlay_png_base64": _fake_png_base64(),
                              "used_localization_box": True, "message": None},
        }
        self._cache[response["image_id"]] = response
        return response

    def get_cached(self, image_id: str):
        return self._cache.get(image_id)


@pytest.fixture
def client():
    svc_module._service_instance = FakeService()
    with TestClient(app) as c:
        yield c
    svc_module._service_instance = None


@pytest.fixture
def client_no_classifier():
    svc_module._service_instance = FakeService(classifier_loaded=False)
    with TestClient(app) as c:
        yield c
    svc_module._service_instance = None


# ---------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------

def test_health_reports_loaded_models(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["classifier_loaded"] is True
    assert "diagnos" in body["disclaimer"].lower()  # the disclaimer must always be present


def test_health_reports_missing_classifier(client_no_classifier):
    res = client_no_classifier.get("/health")
    assert res.status_code == 200
    assert res.json()["classifier_loaded"] is False


# ---------------------------------------------------------------------
# POST /predict/upload -- the full workflow
# ---------------------------------------------------------------------

def test_predict_upload_returns_full_pipeline_response(client):
    fake_image_bytes = cv2.imencode(".png", np.zeros((32, 32), dtype=np.uint8))[1].tobytes()
    res = client.post("/predict/upload", files={"file": ("test.png", fake_image_bytes, "image/png")})

    assert res.status_code == 200
    body = res.json()
    for key in ["image_id", "quality", "localization", "classification",
                "uncertainty", "explainability", "segmentation", "disclaimer"]:
        assert key in body
    assert body["classification"]["predicted_label"] in ("benign", "malignant")
    assert "diagnos" in body["disclaimer"].lower()


def test_predict_upload_rejects_unsupported_file_extension(client):
    res = client.post("/predict/upload", files={"file": ("test.exe", b"not an image", "application/octet-stream")})
    assert res.status_code == 400
    assert "unsupported" in res.json()["detail"].lower()


def test_predict_upload_returns_503_when_no_classifier_loaded(client_no_classifier):
    fake_image_bytes = cv2.imencode(".png", np.zeros((32, 32), dtype=np.uint8))[1].tobytes()
    res = client_no_classifier.post("/predict/upload", files={"file": ("test.png", fake_image_bytes, "image/png")})
    assert res.status_code == 503


def test_predict_upload_requires_a_file(client):
    res = client.post("/predict/upload")
    assert res.status_code == 422  # FastAPI's own validation for a missing required field


# ---------------------------------------------------------------------
# GET /explain/{image_id}, /uncertainty/{image_id} -- cache re-fetch
# ---------------------------------------------------------------------

def test_explain_and_uncertainty_reuse_cached_prediction(client):
    fake_image_bytes = cv2.imencode(".png", np.zeros((32, 32), dtype=np.uint8))[1].tobytes()
    predict_res = client.post("/predict/upload", files={"file": ("test.png", fake_image_bytes, "image/png")})
    image_id = predict_res.json()["image_id"]

    explain_res = client.get(f"/explain/{image_id}")
    assert explain_res.status_code == 200
    assert "overlay_png_base64" in explain_res.json()

    uncertainty_res = client.get(f"/uncertainty/{image_id}")
    assert uncertainty_res.status_code == 200
    assert "needs_review" in uncertainty_res.json()


def test_explain_returns_404_for_unknown_image_id(client):
    res = client.get("/explain/this-id-does-not-exist")
    assert res.status_code == 404


def test_uncertainty_returns_404_for_unknown_image_id(client):
    res = client.get("/uncertainty/this-id-does-not-exist")
    assert res.status_code == 404
