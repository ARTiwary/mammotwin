"""
Evaluates an ensemble of lesion_crop + multimodal (now your two strongest
classifiers) on the locked test set.

This is a STANDALONE script, not a patch to run_phase14_final_evaluation.py,
because the two models don't cover the same test images: multimodal scores
all 526 test rows, but lesion_crop only covers the 512 that have a valid
bounding box. The existing ensemble code in run_phase14_final_evaluation.py
assumes identical row order/count (true for baseline+multimodal, since both
use the full test_df) and would silently misalign if reused here. This
script instead merges the two models' predictions on `image_id`, so only
the correctly-matched, genuinely shared 512 images are combined.

Threshold selection follows the same rule as everywhere else in this
project: chosen on the VALIDATION set only, then applied once here to the
test set.

Usage:
    python scripts/evaluate_ensemble.py \\
        --lesion-crop-checkpoint models\\lesion_crop_resnet50_20260919_095210.pt \\
        --multimodal-checkpoint models\\multimodal_resnet50_20260913_192536.pt \\
        --raw-images-dir "C:\\...\\Datasets\\jpeg" \\
        --i-understand-this-locks-the-test-set
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.config import load_config
from src.utils.metrics import compute_classification_metrics, format_metrics_report, plot_confusion_matrix
from src.utils.stats import select_threshold_for_target_sensitivity, roc_auc_score
from src.data.crop_dataset import LesionCropDataset
from src.data.multimodal_dataset import MultimodalDataset
from src.data.tabular_preprocessing import TabularPreprocessor
from src.models.classifier import build_classifier
from src.models.multimodal_model import build_multimodal_model
from run_phase14_final_evaluation import find_best_checkpoint, run_classifier_inference, bootstrap_ci


def get_lesion_crop_predictions(checkpoint_path, bbox_df, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_classifier(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = LesionCropDataset(bbox_df, checkpoint["config"])
    # LesionCropDataset filters internally -- get the image_ids it ACTUALLY
    # kept, in the same order, so predictions can be merged correctly.
    image_ids = dataset.df["image_id"].values
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=False)
    return pd.DataFrame({"image_id": image_ids, "y_true": y_true, "lesion_crop_prob": y_prob})


def get_multimodal_predictions(checkpoint_path, df, train_df, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cat_cols = checkpoint.get("tabular_categorical_cols", [])
    num_cols = checkpoint.get("tabular_numeric_cols", [])
    tabular_pp = TabularPreprocessor(cat_cols, num_cols)
    tabular_pp.fit(train_df)
    model = build_multimodal_model(checkpoint["config"], tabular_input_dim=tabular_pp.output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = MultimodalDataset(df, checkpoint["config"], tabular_pp)
    image_ids = dataset.df["image_id"].values
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=True)
    return pd.DataFrame({"image_id": image_ids, "y_true_check": y_true, "multimodal_prob": y_prob})


def build_ensemble_predictions(lesion_crop_ckpt, multimodal_ckpt, split_df, train_df, bbox_df, device):
    """Returns (image_ids, y_true, ensemble_prob) merged and aligned by image_id."""
    lc = get_lesion_crop_predictions(lesion_crop_ckpt, bbox_df, device)
    mm = get_multimodal_predictions(multimodal_ckpt, split_df, train_df, device)

    merged = lc.merge(mm, on="image_id", how="inner")
    # Sanity check: the two datasets' own label encodings must agree on
    # the images they share -- if not, something upstream disagrees about
    # ground truth and averaging their probabilities would be meaningless.
    mismatches = (merged["y_true"] != merged["y_true_check"]).sum()
    if mismatches > 0:
        raise ValueError(f"{mismatches} images have disagreeing ground-truth labels between "
                          f"lesion_crop and multimodal data -- refusing to ensemble until "
                          f"that's resolved.")

    ensemble_prob = (merged["lesion_crop_prob"] + merged["multimodal_prob"]) / 2.0
    return merged["image_id"].values, merged["y_true"].values, ensemble_prob.values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesion-crop-checkpoint", type=str, required=True)
    parser.add_argument("--multimodal-checkpoint", type=str, required=True)
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--i-understand-this-locks-the-test-set", action="store_true",
                         help="Required to run the TEST-set portion of this script.")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]

    train_df = pd.read_csv(os.path.join(metadata_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val_split.csv"))

    val_bbox_df = pd.read_csv(os.path.join(metadata_dir, "bbox_metadata_val.csv"))
    if "pathology_binary" not in val_bbox_df.columns:
        val_bbox_df = val_bbox_df.merge(val_df[["image_id", "pathology_binary"]], on="image_id", how="left")

    print("=" * 70)
    print("STEP 1: selecting the ensemble's operating threshold on VALIDATION data")
    print("=" * 70)
    _, val_y_true, val_ens_prob = build_ensemble_predictions(
        args.lesion_crop_checkpoint, args.multimodal_checkpoint, val_df, train_df, val_bbox_df, device)
    print(f"Ensemble validation set size (images both models scored): {len(val_y_true)}")

    threshold_result = select_threshold_for_target_sensitivity(val_y_true, val_ens_prob, args.target_sensitivity)
    threshold = threshold_result["threshold"]
    print(f"Selected threshold: {threshold:.4f}  "
          f"(val sensitivity={threshold_result['sensitivity']:.3f}, "
          f"val specificity={threshold_result['specificity']:.3f}, "
          f"target_met={threshold_result['target_met']})")

    thresholds_path = os.path.join(metadata_dir, "operating_thresholds.json")
    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    thresholds["ensemble_lesion_crop_multimodal"] = {
        "operating_threshold": threshold, "target_sensitivity": args.target_sensitivity,
        "target_met_on_validation": threshold_result["target_met"],
        "val_sensitivity_at_threshold": threshold_result["sensitivity"],
        "val_specificity_at_threshold": threshold_result["specificity"],
        "selected_on": "validation_split_only (lesion_crop + multimodal merged on image_id)",
    }
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved to {thresholds_path}\n")

    if not args.i_understand_this_locks_the_test_set:
        print("Stopping here -- pass --i-understand-this-locks-the-test-set to also "
              "evaluate on the locked test set (do this only once).")
        return

    print("=" * 70)
    print("STEP 2: evaluating on the LOCKED TEST SET (one-time)")
    print("=" * 70)
    test_df = pd.read_csv(os.path.join(metadata_dir, "test_split.csv"))
    test_bbox_df = pd.read_csv(os.path.join(metadata_dir, "bbox_metadata_test.csv"))
    if "pathology_binary" not in test_bbox_df.columns:
        test_bbox_df = test_bbox_df.merge(test_df[["image_id", "pathology_binary"]], on="image_id", how="left")

    image_ids, y_true, y_prob = build_ensemble_predictions(
        args.lesion_crop_checkpoint, args.multimodal_checkpoint, test_df, train_df, test_bbox_df, device)
    print(f"Test images both models scored (intersection): {len(y_true)} "
          f"(out of {len(test_df)} total test rows)")

    metrics_default = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    metrics_tuned = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    print("\n--- Default 0.5 threshold (reference only) ---")
    print(format_metrics_report(metrics_default))
    print(f"\n--- Operating threshold {threshold:.4f} (selected on VALIDATION set) ---")
    print(format_metrics_report(metrics_tuned))

    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=args.n_bootstrap)
    print(f"\nROC-AUC: {auc_point:.4f}  95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]")
    print("(compare against lesion_crop alone: 0.7989, multimodal alone: 0.7557, "
          "baseline+multimodal ensemble: 0.7598)")

    cm_path = os.path.join(figures_dir, "phase14_confusion_matrix_ensemble_lesion_crop_multimodal.png")
    plot_confusion_matrix(metrics_tuned["confusion_matrix"], cm_path,
                           title=f"ensemble (lesion_crop+multimodal) — Test Set (threshold={threshold:.3f})")
    print(f"Saved confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()