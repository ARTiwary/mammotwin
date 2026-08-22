"""
Phase 2 deliverable: prove the environment and pipeline plumbing work
end-to-end by loading one image, preprocessing it, and displaying the result.

Usage:
    python scripts/verify_environment.py
        -> runs on a synthetic placeholder image (no real data needed yet)

    python scripts/verify_environment.py --image path/to/image.png
    python scripts/verify_environment.py --image path/to/image.dcm
        -> runs on a real image once you have one

Run from the repo root (mammotwin/).
"""

import os
import sys
import argparse

# Make src/ importable when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # safe for headless / no-display environments
import matplotlib.pyplot as plt

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image


def check_imports():
    """Confirm the core Phase 2 dependencies are installed and print versions."""
    print("=== Checking core dependencies ===")
    import numpy, cv2, yaml, matplotlib
    print(f"  numpy      {numpy.__version__}")
    print(f"  opencv     {cv2.__version__}")
    print(f"  matplotlib {matplotlib.__version__}")
    print(f"  pyyaml     {yaml.__version__}")
    try:
        import pydicom
        print(f"  pydicom    {pydicom.__version__}")
    except ImportError:
        print("  pydicom    NOT INSTALLED (needed once you have .dcm files)")
    try:
        import torch
        print(f"  torch      {torch.__version__} (cuda available: {torch.cuda.is_available()})")
    except ImportError:
        print("  torch      NOT INSTALLED YET (needed from Phase 6 onward)")
    print()


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 2 environment check")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a real image (.png/.jpg/.dcm). "
                              "If omitted, a synthetic placeholder is used.")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to config.yaml (defaults to config/config.yaml)")
    args = parser.parse_args()

    check_imports()

    config = load_config(args.config) if args.config else load_config()
    set_global_seed(config["project"]["seed"])
    print(f"Project: {config['project']['name']} | seed: {config['project']['seed']}")

    if args.image:
        print(f"Loading real image: {args.image}")
        raw_img = load_image(args.image)
        source_label = os.path.basename(args.image)
    else:
        print("No --image passed. Using a SYNTHETIC placeholder image.")
        print("(This has no diagnostic meaning — it only verifies the pipeline "
              "mechanics before real data is downloaded in Phase 3.)")
        raw_img = generate_synthetic_mammogram()
        source_label = "synthetic placeholder"

    print(f"Raw image shape: {raw_img.shape}, dtype: {raw_img.dtype}, "
          f"min: {raw_img.min():.2f}, max: {raw_img.max():.2f}")

    processed_img = preprocess_image(raw_img, config)
    print(f"Processed image shape: {processed_img.shape}, dtype: {processed_img.dtype}, "
          f"min: {processed_img.min():.3f}, max: {processed_img.max():.3f}")

    # Display before / after
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title(f"Raw ({source_label})")
    axes[0].axis("off")

    axes[1].imshow(processed_img, cmap="gray")
    axes[1].set_title(f"Preprocessed {tuple(config['preprocessing']['image_size'])}")
    axes[1].axis("off")

    plt.tight_layout()

    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "phase2_preprocessing_check.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved before/after figure to: {out_path}")
    print("Phase 2 environment check PASSED.")


if __name__ == "__main__":
    main()