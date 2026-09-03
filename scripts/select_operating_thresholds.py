"""
Fixes the low-sensitivity problem found in Phase 14.

Every classifier (whole-image baseline, lesion-crop, multimodal) was being
scored with the default probability cutoff of 0.5 — an arbitrary choice
with no connection to the actual cost of an error on this task. On an
imbalanced dataset, 0.5 systematically favors the majority (benign) class,
which is exactly why Phase 14 measured only ~52-53% sensitivity: the
models were missing roughly half of the malignant cases at that cutoff.

This script does NOT retrain anything. It only chooses, per classifier,
WHERE on the model's existing probability output "positive" should start,
using the target sensitivity in config.yaml (training.target_sensitivity,
default 0.90) as the requirement, and picking the most specific threshold
that still meets it.

Critically, this selection is done on the VALIDATION set ONLY. The test
set stays locked — Phase 14 (run_phase14_final_evaluation.py) then
*applies* the threshold chosen here, it does not choose it. Choosing a
threshold on the test set would be exactly the kind of test-set tuning
Phase 14's "touch it once" rule exists to prevent.

Usage:
    python scripts/select_operating_thresholds.py
    python scripts/select_operating_thresholds.py --target-sensitivity 0.95

Reads:
    models/registry.json                          (best checkpoint per model)
    data/metadata/val_split.csv                    (whole-image + multimodal)
    data/metadata/bbox_metadata_val.csv            (lesion-crop, if present)
    data/metadata/train_split.csv                  (refit tabular preprocessor)

Writes:
    data/metadata/operating_thresholds.json
    reports/figures/threshold_selection_<model>.png (sensitivity/specificity vs. threshold)
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_global_seed
from src.utils.stats import roc_points, select_threshold_for_target_sensitivity, youden_optimal_threshold
from src.data.dataset import MammogramDataset
from src.data.crop_dataset import LesionCropDataset
from src.data.multimodal_dataset import MultimodalDataset
from src.data.tabular_preprocessing import TabularPreprocessor
from src.models.classifier import build_classifier
from src.models.multimodal_model import build_multimodal_model

# Reuse the exact checkpoint-discovery and inference logic Phase 14 uses,
# so "the model being tuned" and "the model being evaluated" can never
# silently drift apart.
from run_phase14_final_evaluation import find_best_checkpoint, run_classifier_inference


def plot_threshold_sweep(y_true, y_prob, chosen, figures_dir, name):
    thresholds, tprs, tnrs = roc_points(y_true, y_prob)
    order = np.argsort(thresholds)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds[order], tprs[order], label="Sensitivity (validation)")
    ax.plot(thresholds[order], tnrs[order], label="Specificity (validation)")
    ax.axvline(chosen["threshold"], color="black", linestyle="--",
               label=f"Chosen threshold ({chosen['threshold']:.3f})")
    ax.axhline(chosen["target_sensitivity"], color="gray", linestyle=":",
               label=f"Target sensitivity ({chosen['target_sensitivity']:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    ax.set_title(f"{name} — operating threshold selected on VALIDATION data")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"threshold_selection_{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def select_for_model(name, y_true, y_prob, target_sensitivity, figures_dir):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        print(f"  Skipping {name}: validation predictions contain only one class.")
        return None

    chosen = select_threshold_for_target_sensitivity(y_true, y_prob, target_sensitivity)
    youden = youden_optimal_threshold(y_true, y_prob)

    if not chosen["target_met"]:
        print(f"  WARNING: {name} cannot reach {target_sensitivity:.0%} sensitivity on the "
              f"validation set even by flagging every case positive. Best achievable: "
              f"{chosen['sensitivity']:.1%} sensitivity at threshold {chosen['threshold']:.3f}. "
              f"This is a genuine model-quality limitation, not a threshold-selection bug — "
              f"report it as such rather than silently lowering the target.")

    fig_path = plot_threshold_sweep(y_true, y_prob, chosen, figures_dir, name)

    print(f"  {name}: threshold={chosen['threshold']:.4f}  "
          f"sensitivity={chosen['sensitivity']:.3f}  specificity={chosen['specificity']:.3f}  "
          f"(target={target_sensitivity:.2f}, met={chosen['target_met']})")
    print(f"  {name}: for reference, Youden-optimal threshold={youden['threshold']:.4f} "
          f"(sensitivity={youden['sensitivity']:.3f}, specificity={youden['specificity']:.3f}) "
          f"— NOT used as the operating point, since it weighs a missed malignant case the "
          f"same as an unnecessary review, which is not appropriate here.")

    return {
        "operating_threshold": chosen["threshold"],
        "val_sensitivity_at_threshold": chosen["sensitivity"],
        "val_specificity_at_threshold": chosen["specificity"],
        "target_sensitivity": target_sensitivity,
        "target_met_on_validation": chosen["target_met"],
        "youden_reference_threshold": youden["threshold"],
        "n_validation_examples": int(len(y_true)),
        "selected_on": "validation_split_only",
        "figure": fig_path,
        "selected_at": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-sensitivity", type=float, default=None,
                         help="Overrides config training.target_sensitivity.")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--lesion-crop-checkpoint", type=str, default=None)
    parser.add_argument("--multimodal-checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sensitivity = args.target_sensitivity or config["training"].get("target_sensitivity", 0.90)
    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]
    models_dir = config["paths"]["models_dir"]

    print("=" * 70)
    print(f"SELECTING OPERATING THRESHOLDS (target sensitivity = {target_sensitivity:.0%})")
    print("Selection uses the VALIDATION split only. The test set is not touched here.")
    print("=" * 70)

    val_csv = os.path.join(metadata_dir, "val_split.csv")
    if not os.path.exists(val_csv):
        print(f"No validation split found at {val_csv}. Run Phase 3 first.")
        return
    val_df = pd.read_csv(val_csv)

    out = {}

    # --- Whole-image baseline ---
    baseline_ckpt = args.baseline_checkpoint or find_best_checkpoint(
        "6_baseline", models_dir, fallback_prefix="baseline_")
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        checkpoint = torch.load(baseline_ckpt, map_location=device, weights_only=False)
        model = build_classifier(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = DataLoader(MammogramDataset(val_df, checkpoint["config"]), batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=False)
        result = select_for_model("whole_image_baseline", y_true, y_prob, target_sensitivity, figures_dir)
        if result:
            result["checkpoint"] = baseline_ckpt
            out["whole_image_baseline"] = result
    else:
        print("  No baseline checkpoint found — skipping.")

    # --- Lesion-crop classifier ---
    bbox_val_csv = os.path.join(metadata_dir, "bbox_metadata_val.csv")
    lesion_crop_ckpt = args.lesion_crop_checkpoint or find_best_checkpoint("9_lesion_crop", models_dir)
    if lesion_crop_ckpt and os.path.exists(lesion_crop_ckpt) and os.path.exists(bbox_val_csv):
        bbox_val_df = pd.read_csv(bbox_val_csv)
        bbox_val_df = bbox_val_df[bbox_val_df["has_bbox"] == True]  # noqa: E712
        if "pathology_binary" not in bbox_val_df.columns:
            bbox_val_df = bbox_val_df.merge(val_df[["image_id", "pathology_binary"]], on="image_id", how="left")
        checkpoint = torch.load(lesion_crop_ckpt, map_location=device, weights_only=False)
        model = build_classifier(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = DataLoader(LesionCropDataset(bbox_val_df, checkpoint["config"]), batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=False)
        result = select_for_model("lesion_crop", y_true, y_prob, target_sensitivity, figures_dir)
        if result:
            result["checkpoint"] = lesion_crop_ckpt
            out["lesion_crop"] = result
    else:
        print("  No lesion-crop checkpoint/validation-bbox data found — skipping.")

    # --- Multimodal ---
    multimodal_ckpt = args.multimodal_checkpoint or find_best_checkpoint("13_multimodal", models_dir)
    if multimodal_ckpt and os.path.exists(multimodal_ckpt):
        checkpoint = torch.load(multimodal_ckpt, map_location=device, weights_only=False)
        cat_cols = checkpoint.get("tabular_categorical_cols", [])
        num_cols = checkpoint.get("tabular_numeric_cols", [])
        train_df = pd.read_csv(os.path.join(metadata_dir, "train_split.csv"))
        tabular_pp = TabularPreprocessor(cat_cols, num_cols)
        tabular_pp.fit(train_df)  # refit on TRAIN only, exactly matching training/Phase 14
        model = build_multimodal_model(checkpoint["config"], tabular_input_dim=tabular_pp.output_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        loader = DataLoader(MultimodalDataset(val_df, checkpoint["config"], tabular_pp), batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=True)
        result = select_for_model("multimodal", y_true, y_prob, target_sensitivity, figures_dir)
        if result:
            result["checkpoint"] = multimodal_ckpt
            out["multimodal"] = result
    else:
        print("  No multimodal checkpoint found — skipping.")

    out_path = os.path.join(metadata_dir, "operating_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved operating thresholds: {out_path}")
    print("Next: re-run scripts/run_phase14_final_evaluation.py — it will automatically pick these up "
          "and report tuned-threshold metrics next to the untuned 0.5 baseline.")


if __name__ == "__main__":
    main()
