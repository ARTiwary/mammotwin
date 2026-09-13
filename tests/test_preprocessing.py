"""
Phase 16: unit tests for the Phase 4 preprocessing pipeline.

Covers: normalize_image, resize_image, quality checks (blank/low-contrast/
too-small), the full preprocess_image() pipeline, and invalid/corrupt
input handling -- per the Phase 16 checklist ("Unit-test preprocessing",
"Test invalid/corrupt uploads").
"""
import numpy as np
import pytest

from src.data.image_io import generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import normalize_image, resize_image, preprocess_image
from src.preprocessing.quality_checks import is_blank, is_low_contrast, breast_area_too_small, run_quality_checks
from src.utils.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def synthetic_mammogram():
    return generate_synthetic_mammogram(size=256, seed=0)


# ---------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------

def test_normalize_image_output_range():
    img = np.random.default_rng(0).normal(100, 20, size=(64, 64)).astype(np.float32)
    out = normalize_image(img)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_image_handles_constant_image_without_dividing_by_zero():
    """A perfectly flat image has zero percentile range -- must not raise
    or produce NaN/inf (the denom-guard in normalize_image exists for
    exactly this case)."""
    img = np.full((32, 32), 50.0, dtype=np.float32)
    out = normalize_image(img)
    assert np.isfinite(out).all()


def test_normalize_image_clips_outliers():
    img = np.zeros((50, 50), dtype=np.float32)
    img[0, 0] = 100000.0  # single extreme outlier pixel
    img[1:, 1:] = np.random.default_rng(1).uniform(0, 10, size=(49, 49))
    out = normalize_image(img)
    # If the outlier were NOT clipped, it would dominate the min-max scale
    # and squash the rest of the (0-10) data into a tiny sliver near 0.
    # With percentile clipping working correctly, the real data instead
    # spreads across most of the [0, 1] range -- verify that directly
    # rather than asserting a specific median (which depends on the
    # uniform sample and isn't the actual property being tested).
    assert out[1:, 1:].max() > 0.8
    assert out[1:, 1:].min() < 0.2


# ---------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------

def test_resize_image_output_shape():
    img = np.random.default_rng(0).uniform(0, 1, size=(300, 400)).astype(np.float32)
    out = resize_image(img, (128, 96))  # (height, width)
    assert out.shape == (128, 96)


# ---------------------------------------------------------------------
# quality checks
# ---------------------------------------------------------------------

def test_is_blank_flags_uniform_image():
    flat = np.full((64, 64), 30.0, dtype=np.float32)
    assert is_blank(flat) is True


def test_is_blank_does_not_flag_normal_image(synthetic_mammogram):
    assert is_blank(synthetic_mammogram) is False


def test_is_low_contrast_flags_narrow_range_image():
    img = np.random.default_rng(0).uniform(100, 102, size=(64, 64)).astype(np.float32)
    assert is_low_contrast(img) is True


def test_is_low_contrast_does_not_flag_normal_image(synthetic_mammogram):
    assert is_low_contrast(synthetic_mammogram) is False


def test_breast_area_too_small():
    assert breast_area_too_small(0.01) is True
    assert breast_area_too_small(0.5) is False


def test_run_quality_checks_passes_for_normal_image(synthetic_mammogram):
    result = run_quality_checks(synthetic_mammogram, breast_area_fraction=0.4)
    assert result["passed"] is True
    assert result["is_blank"] is False
    assert result["is_low_contrast"] is False
    assert result["breast_area_too_small"] is False


def test_run_quality_checks_fails_for_blank_image():
    flat = np.full((64, 64), 10.0, dtype=np.float32)
    result = run_quality_checks(flat)
    assert result["passed"] is False
    assert result["is_blank"] is True


# ---------------------------------------------------------------------
# background removal: marker-artifact zeroing (the shortcut-learning fix)
# ---------------------------------------------------------------------

def test_remove_background_zeros_out_disconnected_marker_artifact():
    """Regression test for the shortcut-learning root cause documented in
    docs/RESULTS_AND_LIMITATIONS.md: a burned-in laterality/view marker
    positioned inside the breast's bounding RECTANGLE, but not part of
    the breast's own connected component, must be zeroed out -- not
    passed through to the model untouched."""
    from src.preprocessing.background_removal import remove_background

    img = np.zeros((300, 300), dtype=np.float32)
    img[50:250, 20:280] = 180.0   # main breast blob
    img[55:90, 25:80] = 0.0       # notch carved out of one corner
    img[62:78, 32:72] = 220.0     # isolated marker inside the notch, disconnected from the blob

    result = remove_background(img)
    cropped = result["cropped"]
    x0, y0, _, _ = result["bbox"]

    marker_in_crop = cropped[62 - y0:78 - y0, 32 - x0:72 - x0]
    blob_in_crop = cropped[150 - y0:160 - y0, 150 - x0:160 - x0]

    assert marker_in_crop.max() == 0, "Marker artifact was not zeroed out"
    assert blob_in_crop.max() == 180.0, "Real breast tissue was incorrectly zeroed out"


# ---------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------

def test_preprocess_image_end_to_end(config, synthetic_mammogram):
    result = preprocess_image(synthetic_mammogram, config, run_quality_gate=True)
    expected_size = tuple(config["preprocessing"]["image_size"])

    assert result["processed"].shape == expected_size
    assert result["processed"].dtype == np.float32
    assert result["processed"].min() >= 0.0
    assert result["processed"].max() <= 1.0
    assert result["quality"] is not None
    assert result["quality"]["passed"] is True
    assert result["bbox"] is not None


def test_preprocess_image_is_deterministic(config, synthetic_mammogram):
    """Same input + same config must give byte-identical output -- any
    hidden randomness here would silently break train/inference matching."""
    r1 = preprocess_image(synthetic_mammogram, config, run_quality_gate=False)
    r2 = preprocess_image(synthetic_mammogram, config, run_quality_gate=False)
    np.testing.assert_array_equal(r1["processed"], r2["processed"])


def test_preprocess_image_skips_quality_gate_when_disabled(config, synthetic_mammogram):
    result = preprocess_image(synthetic_mammogram, config, run_quality_gate=False)
    assert result["quality"] is None


# ---------------------------------------------------------------------
# invalid / corrupt inputs (Phase 16: "Test invalid/corrupt uploads")
# ---------------------------------------------------------------------

def test_preprocess_image_rejects_all_zero_image_via_quality_gate(config):
    """A corrupt/blank upload shouldn't crash the pipeline -- it should
    still produce a (resized) output, but be clearly flagged as failing
    quality, so the caller (e.g. the dashboard) can act on it safely."""
    corrupt = np.zeros((512, 512), dtype=np.float32)
    result = preprocess_image(corrupt, config, run_quality_gate=True)
    assert result["quality"]["passed"] is False
    assert result["processed"].shape == tuple(config["preprocessing"]["image_size"])


def test_preprocess_image_handles_tiny_image_without_crashing(config):
    """An unexpectedly tiny image (e.g. a corrupted/truncated upload)
    should not crash background removal or resizing."""
    tiny = np.random.default_rng(0).uniform(0, 255, size=(8, 8)).astype(np.float32)
    result = preprocess_image(tiny, config, run_quality_gate=True)
    assert result["processed"].shape == tuple(config["preprocessing"]["image_size"])


def test_preprocess_image_handles_nan_pixels_without_crashing(config):
    """Corrupt image files can decode with NaN/inf pixels. This should not
    raise -- it's fine for the OUTPUT to also be degenerate (NaN in,
    NaN-ish out) as long as the pipeline doesn't crash; the quality gate
    is what's responsible for catching this case downstream."""
    corrupt = np.random.default_rng(0).uniform(0, 255, size=(256, 256)).astype(np.float32)
    corrupt[10:20, 10:20] = np.nan
    try:
        result = preprocess_image(corrupt, config, run_quality_gate=True)
    except Exception as e:
        pytest.fail(f"preprocess_image raised on NaN input instead of degrading gracefully: {e}")
    assert result["processed"].shape == tuple(config["preprocessing"]["image_size"])
