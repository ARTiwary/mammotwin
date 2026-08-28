"""
Phase 12 deliverable: the MammoTwin longitudinal (prior-vs-current)
comparison module.

============================================================================
IMPORTANT — READ BEFORE USING ANY OUTPUT FROM THIS SCRIPT:
CBIS-DDSM contains NO genuine prior/current exam pairs for the same
patient (confirmed in the Phase 1 project plan). Every "baseline" and
"follow-up" timepoint this script produces is SIMULATED from a single
real exam by synthetically shrinking/removing the real lesion and adding
a small random misalignment. This demonstrates the SOFTWARE'S capability
to register, compare, and quantify change between two mammograms — it
does NOT demonstrate anything about real disease progression, and must
never be described or shown as if it were real longitudinal data.
============================================================================

Real data:
    python scripts/run_phase12_longitudinal.py --checkpoint models/baseline_resnet50_<timestamp>.pt

Demo mode (fully synthetic image, no real CBIS-DDSM data needed):
    python scripts/run_phase12_longitudinal.py --demo
"""

import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image
from src.models.classifier import build_classifier
from src.longitudinal.simulate_pairs import simulate_timeline
from src.longitudinal.registration import register_images
from src.longitudinal.change_detection import compute_change_features

SIMULATION_BANNER = (
    "SIMULATED LONGITUDINAL DEMONSTRATION — NOT REAL PATIENT DATA. "
    "CBIS-DDSM contains no genuine prior/current exam pairs; this timeline "
    "is synthesized from a single real exam to demonstrate the pipeline."
)


def get_model_score(model, image, config, device):
    if model is None:
        return None
    result = preprocess_image(image, config, run_quality_gate=False)
    processed = result["processed"]
    tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob


def run_one_patient(image, mask, config, model, device, seed):
    timeline = simulate_timeline(image, mask, seed=seed, add_misalignment=True)

    # Register baseline and follow-up onto CURRENT's coordinate frame.
    current_img = timeline["current"]["image"]
    aligned_baseline_img, M_b, n_inliers_b = register_images(timeline["baseline"]["image"], current_img)
    aligned_followup_img, M_f, n_inliers_f = register_images(timeline["followup"]["image"], current_img)

    # Apply the SAME estimated transform to each timepoint's mask, so mask
    # and image stay consistent after registration.
    def warp_mask(mask_img, M):
        if M is None:
            return mask_img
        h, w = current_img.shape[:2]
        return cv2.warpAffine(mask_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    aligned_baseline_mask = warp_mask(timeline["baseline"]["mask"], M_b)
    aligned_followup_mask = warp_mask(timeline["followup"]["mask"], M_f)
    current_mask = timeline["current"]["mask"]

    score_baseline = get_model_score(model, aligned_baseline_img, config, device)
    score_followup = get_model_score(model, aligned_followup_img, config, device)
    score_current = get_model_score(model, current_img, config, device)

    change_b_to_f = compute_change_features(
        aligned_baseline_mask, aligned_followup_mask, aligned_baseline_img, aligned_followup_img,
        score_baseline, score_followup,
    )
    change_f_to_c = compute_change_features(
        aligned_followup_mask, current_mask, aligned_followup_img, current_img,
        score_followup, score_current,
    )

    return {
        "images": {"baseline": aligned_baseline_img, "followup": aligned_followup_img, "current": current_img},
        "masks": {"baseline": aligned_baseline_mask, "followup": aligned_followup_mask, "current": current_mask},
        "registration_quality": {"baseline_inliers": n_inliers_b, "followup_inliers": n_inliers_f},
        "scores": {"baseline": score_baseline, "followup": score_followup, "current": score_current},
        "change_baseline_to_followup": change_b_to_f,
        "change_followup_to_current": change_f_to_c,
    }


def plot_timeline(result, out_path, patient_label=""):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    stages = ["baseline", "followup", "current"]
    titles = ["Baseline\n(simulated)", "Follow-up\n(simulated)", "Current\n(REAL)"]

    for i, stage in enumerate(stages):
        img = result["images"][stage]
        mask = result["masks"][stage]
        score = result["scores"][stage]

        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].imshow(mask, cmap="Reds", alpha=0.4)
        score_str = f"P(malignant)={score:.3f}" if score is not None else "no classifier provided"
        axes[1, i].set_title(score_str, fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle(f"{SIMULATION_BANNER}\n{patient_label}", fontsize=9, color="darkred", wrap=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def print_change_summary(result):
    print(f"  Registration quality: baseline {result['registration_quality']['baseline_inliers']} "
          f"inlier matches, follow-up {result['registration_quality']['followup_inliers']} inlier matches")

    for transition, changes in [("Baseline -> Follow-up", result["change_baseline_to_followup"]),
                                 ("Follow-up -> Current", result["change_followup_to_current"])]:
        print(f"\n  {transition}:")
        print(f"    Lesion area: {changes['area_a']:.0f} -> {changes['area_b']:.0f} px "
              f"({changes['relative_change']:+.1%} relative change)")
        print(f"    Region overlap (IoU): {changes['region_overlap_iou']:.3f}")
        if "model_score_change" in changes:
            print(f"    Model score change: {changes['model_score_a']:.3f} -> "
                  f"{changes['model_score_b']:.3f} ({changes['model_score_change']:+.3f})")


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 12: longitudinal module")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Optional trained classifier checkpoint, to also report model-score change")
    parser.add_argument("--bbox-val-csv", type=str, default=None)
    parser.add_argument("--n-patients", type=int, default=3)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("\n" + "=" * 70)
    print(SIMULATION_BANNER)
    print("=" * 70 + "\n")

    model = None
    if args.demo:
        print("Running Phase 12 in DEMO mode on a fully synthetic image.\n")
        config = load_config(args.config) if args.config else load_config()
        set_global_seed(config["project"]["seed"])
        image = generate_synthetic_mammogram(size=400, seed=1)
        # Must match generate_synthetic_mammogram's own lesion geometry exactly
        # (cy, cx, r formula) — a mismatched mask would leave real lesion
        # pixels outside it, which inpainting then can't remove.
        size = 400
        cy, cx, r = int(size * 0.45), int(size * 0.4), size // 18
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2).astype(np.uint8) * 255
        examples = [(image, mask, "demo_patient_0")]
    else:
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            config = checkpoint["config"]
            model = build_classifier(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            print(f"Loaded classifier from {args.checkpoint} — will report model-score change too.\n")
        else:
            config = load_config(args.config) if args.config else load_config()
            print("No --checkpoint provided — will report area/overlap changes only "
                  "(no model-score change).\n")

        metadata_dir = config["paths"]["data_metadata"]
        bbox_csv = args.bbox_val_csv or os.path.join(metadata_dir, "bbox_metadata_val.csv")
        if not os.path.exists(bbox_csv):
            print(f"No bbox metadata at {bbox_csv}. Run Phase 7's build_bbox_metadata.py first, or use --demo.")
            return
        bbox_df = pd.read_csv(bbox_csv)
        bbox_df = bbox_df[bbox_df["has_bbox"] == True]  # noqa: E712
        bbox_df = bbox_df.dropna(subset=["mask_path_used", "image_file_path_resolved"])
        sample = bbox_df.sample(n=min(args.n_patients, len(bbox_df)), random_state=config["project"]["seed"])

        examples = []
        for _, row in sample.iterrows():
            img = load_image(row["image_file_path_resolved"])
            mask = cv2.imread(row["mask_path_used"], cv2.IMREAD_GRAYSCALE)
            examples.append((img, mask, row.get("image_id", "unknown")))

    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    for i, (image, mask, patient_label) in enumerate(examples):
        print(f"\n{'=' * 70}\nPatient {i+1}/{len(examples)}: {patient_label} (SIMULATED timeline)\n{'=' * 70}")
        result = run_one_patient(image, mask, config, model, device, seed=42 + i)
        print_change_summary(result)

        fig_path = os.path.join(figures_dir, f"phase12_timeline_{i}.png")
        plot_timeline(result, fig_path, patient_label=f"Patient: {patient_label}")
        print(f"\n  Saved timeline figure: {fig_path}")

    print("\n" + "=" * 70)
    print(SIMULATION_BANNER)
    print("=" * 70)
    print("\nPhase 12 longitudinal module COMPLETE.")


if __name__ == "__main__":
    main()