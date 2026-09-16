"""
Phase 16: "Test inference against saved reference outputs", "Test that
training and inference preprocessing match", and inference-path tests for
uncertainty (MC-Dropout) and explainability (Grad-CAM).

The torch-dependent tests build a tiny, UNTRAINED resnet18
(pretrained=False) rather than loading a real checkpoint or downloading
ImageNet weights -- they test that the MECHANICS are correct (shapes,
value ranges, determinism, dropout actually introducing variance), not
that any particular model performs well, so no real weights are needed
and these run offline.
"""
import hashlib
import copy

import numpy as np
import pytest

from src.data.image_io import generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image
from src.utils.config import load_config


# ---------------------------------------------------------------------
# Golden-output regression test (no torch needed)
# ---------------------------------------------------------------------

# Computed once from the current preprocessing pipeline + config.yaml
# (see the comment below) -- if this ever changes, it means a preprocessing
# step changed behavior, which is exactly the kind of silent drift this
# test exists to catch. If the change was intentional, recompute and
# update this constant deliberately, don't just delete the test.
_EXPECTED_CHECKSUM = "712fe6d1762461618165c32ccb85ea667b770253dfd70c95090bc3dd44982ad5"


def test_preprocessing_output_matches_golden_reference():
    """Regenerating this checksum: run
        python -c "from src.data.image_io import generate_synthetic_mammogram as g; \\
                    from src.preprocessing.basic_preprocess import preprocess_image as p; \\
                    from src.utils.config import load_config as c; import hashlib; \\
                    r = p(g(size=512, seed=42), c(), run_quality_gate=False); \\
                    print(hashlib.sha256(r['processed'].tobytes()).hexdigest())"
    """
    config = load_config()
    img = generate_synthetic_mammogram(size=512, seed=42)
    result = preprocess_image(img, config, run_quality_gate=False)
    checksum = hashlib.sha256(result["processed"].tobytes()).hexdigest()
    assert checksum == _EXPECTED_CHECKSUM, (
        "Preprocessing output changed! If this is an intentional change to "
        "preprocessing (e.g. a bug fix), regenerate _EXPECTED_CHECKSUM using "
        "the command in this test's docstring. If it's NOT intentional, this "
        "is exactly the kind of silent train/inference drift Phase 16 exists "
        "to catch -- do not update the constant without understanding why it changed."
    )


# ---------------------------------------------------------------------
# Training vs. inference preprocessing must match
# ---------------------------------------------------------------------

def test_quality_gate_flag_does_not_affect_the_processed_image():
    """run_quality_gate=True (used during EDA/training-time inspection) vs
    False (used at inference in most scripts) must produce an IDENTICAL
    processed image -- only the `quality` dict should differ. If these
    ever diverged, a model's inputs at eval time could differ from what
    it saw during quality-checked training, which is the exact class of
    bug this project has already hit once (see docs/OVERFITTING_FIXES.md)."""
    config = load_config()
    img = generate_synthetic_mammogram(size=400, seed=1)

    with_gate = preprocess_image(img, config, run_quality_gate=True)
    without_gate = preprocess_image(img, config, run_quality_gate=False)

    np.testing.assert_array_equal(with_gate["processed"], without_gate["processed"])
    assert with_gate["quality"] is not None
    assert without_gate["quality"] is None


def test_checkpoint_saved_config_preprocessing_matches_current_config(tmp_path):
    """Simulates the real failure mode this test guards against: a model
    checkpoint saves its OWN config (see every training script in this
    project), and inference must use THAT saved config, not whatever
    config.yaml says today (which may have changed since training). This
    test proves preprocess_image is fully config-driven -- swapping in a
    different image_size actually changes the output shape -- so using
    the checkpoint's saved config at inference is meaningful, not a no-op."""
    config = load_config()
    img = generate_synthetic_mammogram(size=400, seed=2)

    # A checkpoint "from the past" trained with a different image_size.
    old_config = copy.deepcopy(config)
    old_config["preprocessing"]["image_size"] = [128, 128]

    result_current = preprocess_image(img, config, run_quality_gate=False)
    result_old = preprocess_image(img, old_config, run_quality_gate=False)

    assert result_current["processed"].shape == tuple(config["preprocessing"]["image_size"])
    assert result_old["processed"].shape == (128, 128)
    assert result_current["processed"].shape != result_old["processed"].shape


# ---------------------------------------------------------------------
# MC-Dropout (uncertainty) -- mechanics only, untrained model
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_classifier():
    torch = pytest.importorskip("torch", reason="torch not installed in this environment")
    from src.models.classifier import build_classifier
    config = {"model": {"backbone": "resnet18", "pretrained": False, "num_classes": 2, "dropout": 0.5}}
    model = build_classifier(config)
    model.eval()
    return model


def test_mc_dropout_predict_returns_valid_probability_range(tiny_classifier):
    torch = pytest.importorskip("torch")
    from src.uncertainty.mc_dropout import mc_dropout_predict
    tensor = torch.rand(1, 3, 64, 64)
    mean_prob, std_prob = mc_dropout_predict(tiny_classifier, tensor, n_passes=10)
    assert 0.0 <= mean_prob <= 1.0
    assert std_prob >= 0.0


def test_mc_dropout_introduces_variance_that_plain_eval_does_not(tiny_classifier):
    """The whole point of MC-Dropout: repeated forward passes with dropout
    ACTIVE should disagree with each other (std > 0). A plain model.eval()
    forward pass, called repeatedly, must be perfectly deterministic
    (std == 0) since dropout is off -- this is the contrast that proves
    enable_mc_dropout() is actually doing something."""
    torch = pytest.importorskip("torch")
    from src.uncertainty.mc_dropout import mc_dropout_predict
    tensor = torch.rand(1, 3, 64, 64)

    _, std_with_dropout = mc_dropout_predict(tiny_classifier, tensor, n_passes=30)

    tiny_classifier.eval()  # dropout OFF, no MC
    with torch.no_grad():
        probs = [torch.softmax(tiny_classifier(tensor), dim=1)[0, 1].item() for _ in range(30)]
    std_without_dropout = float(np.std(probs))

    assert std_without_dropout == pytest.approx(0.0, abs=1e-6)
    assert std_with_dropout > std_without_dropout


# ---------------------------------------------------------------------
# Grad-CAM -- mechanics only, untrained model
# ---------------------------------------------------------------------

def test_gradcam_output_shape_matches_input(tiny_classifier):
    torch = pytest.importorskip("torch")
    from src.explainability.gradcam import GradCAM
    gradcam = GradCAM(tiny_classifier, backbone_name="resnet18")
    tensor = torch.rand(1, 3, 96, 80)  # deliberately non-square, catches H/W swap bugs
    heatmap, pred_class, prob = gradcam.generate(tensor, target_class=1)

    assert heatmap.shape == (96, 80)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert pred_class == 1
    assert 0.0 <= prob <= 1.0


def test_gradcam_defaults_to_model_top_prediction_when_target_class_is_none(tiny_classifier):
    torch = pytest.importorskip("torch")
    from src.explainability.gradcam import GradCAM
    gradcam = GradCAM(tiny_classifier, backbone_name="resnet18")
    tensor = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        expected_class = int(tiny_classifier(tensor).argmax(dim=1).item())

    _, pred_class, _ = gradcam.generate(tensor, target_class=None)
    assert pred_class == expected_class


def test_gradcam_rejects_unknown_backbone_without_explicit_target_layer(tiny_classifier):
    from src.explainability.gradcam import GradCAM
    with pytest.raises(ValueError):
        GradCAM(tiny_classifier, backbone_name="not_a_real_backbone")


# ---------------------------------------------------------------------
# Pretrained-encoder U-Net segmentation model (accuracy-improvement follow-up)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("size", [(224, 224), (256, 256), (250, 250), (192, 320)])
def test_pretrained_unet_output_shape_matches_input(size):
    """pretrained=False so this runs offline without downloading ImageNet
    weights -- tests the architecture's shape mechanics only, at both
    square/non-square and even/odd resolutions (odd sizes catch decoder
    upsample/skip-connection size-mismatch bugs that only show up when
    dimensions don't divide evenly by the encoder's downsampling factor)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.models.segmentation_pretrained import PretrainedUNet

    model = PretrainedUNet(encoder_name="resnet18", pretrained=False)
    model.eval()
    x = torch.randn(2, 1, *size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, *size)


def test_pretrained_unet_gradients_flow_without_nan():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.models.segmentation_pretrained import PretrainedUNet

    model = PretrainedUNet(encoder_name="resnet18", pretrained=False)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_build_segmentation_model_dispatches_on_encoder_config():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.models.segmentation_model import build_segmentation_model, UNet
    from src.models.segmentation_pretrained import PretrainedUNet

    scratch = build_segmentation_model({"segmentation": {"encoder": "scratch", "base_filters": 8}})
    assert isinstance(scratch, UNet)

    pretrained = build_segmentation_model({"segmentation": {"encoder": "resnet18", "pretrained": False}})
    assert isinstance(pretrained, PretrainedUNet)

    with pytest.raises(ValueError):
        build_segmentation_model({"segmentation": {"encoder": "not_a_real_encoder"}})
