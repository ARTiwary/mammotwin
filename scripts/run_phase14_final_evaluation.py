"""
Phase 14 deliverable: THE FINAL EVALUATION. The test set is touched here
for the first and only time in this project.

Golden rule (from the project plan): choose/tune models using validation
data only; use the final test set ONCE for the final unbiased estimate.
Every model evaluated here was already trained and selected using
validation data in earlier phases — nothing is retrained or tuned here.

Usage:
    python scripts/run_phase14_final_evaluation.py --i-understand-this-locks-the-test-set \
        --raw-images-dir "C:\\...\\Datasets\\jpeg"

Checkpoints are auto-discovered from models/registry.json (best val_auc per
phase) unless explicitly overridden with --baseline-checkpoint,
--lesion-crop-checkpoint, --multimodal-checkpoint, --detector-checkpoint,
--segmentation-checkpoint.
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_global_seed, REPO_ROOT
from src.utils.metrics import compute_classification_metrics, format_metrics_report
from src.utils.stats import bootstrap_ci, roc_auc_score, average_precision_score
from src.utils.calibration import plot_reliability_diagram
from src.data.dataset import MammogramDataset
from src.data.crop_dataset import LesionCropDataset
from src.data.multimodal_dataset import MultimodalDataset
from src.data.tabular_preprocessing import TabularPreprocessor
from src.models.classifier import build_classifier
from src.models.multimodal_model import build_multimodal_model


def find_best_checkpoint(phase_name: str, models_dir: str, fallback_prefix: str = None):
    """
    Auto-discover the best checkpoint for a given phase from registry.json.

    Falls back to matching by run_id PREFIX if no entry has a matching
    'phase' tag — Phase 6's baseline classifier script (an earlier script
    in this project than Phase 9/13's) never wrote a 'phase' field into its
    registry entries, so phase-tag lookup alone misses it. Prefix matching
    (e.g. run_ids starting with "baseline_") recovers it without requiring
    you to pass --baseline-checkpoint manually every time.
    """
    registry_path = os.path.join(models_dir, "registry.json")
    if not os.path.exists(registry_path) or os.path.getsize(registry_path) == 0:
        return None
    with open(registry_path) as f:
        registry = json.load(f)

    matching = {k: v for k, v in registry.items() if v.get("phase") == phase_name}

    if not matching and fallback_prefix:
        matching = {k: v for k, v in registry.items() if k.startswith(fallback_prefix)}

    if not matching:
        return None
    best_key = max(matching, key=lambda k: matching[k].get("val_auc", -1))
    return matching[best_key]["checkpoint_path"]


@torch.no_grad()
def run_classifier_inference(model, loader, device, multimodal=False):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        if multimodal:
            images, tabular, labels = batch
            images, tabular = images.to(device), tabular.to(device)
            outputs = model(images, tabular)
        else:
            images, labels = batch
            images = images.to(device)
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def evaluate_classifier(name, checkpoint_path, test_df, config_override, device,
                         dataset_type: str, figures_dir: str, n_bootstrap: int,
                         tabular_pp=None, operating_threshold_info=None):
    print(f"\n{'=' * 70}\nEvaluating classifier: {name}\n{'=' * 70}")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"  No checkpoint found for '{name}' — skipping.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    if dataset_type == "whole_image":
        model = build_classifier(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = MammogramDataset(test_df, config)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=False)

    elif dataset_type == "lesion_crop":
        model = build_classifier(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = LesionCropDataset(test_df, config)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=False)

    elif dataset_type == "multimodal":
        tabular_pp.fit(config_override["train_df_for_refit"])  # refit on TRAIN only, matches training exactly
        model = build_multimodal_model(config, tabular_input_dim=tabular_pp.output_dim).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = MultimodalDataset(test_df, config, tabular_pp)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=True)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    print(f"  Test set size: {len(y_true)}, checkpoint: {checkpoint_path}")

    # Reference only — 0.5 is an arbitrary cutoff with no connection to the
    # actual cost of a missed malignant case. Kept here purely so the
    # improvement from proper threshold selection is visible, not hidden.
    metrics_default = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    print("  --- Default 0.5 threshold (reference only) ---")
    print("  " + format_metrics_report(metrics_default).replace("\n", "\n  "))

    metrics = metrics_default
    operating_threshold = 0.5
    if operating_threshold_info is not None:
        operating_threshold = operating_threshold_info["operating_threshold"]
        metrics = compute_classification_metrics(y_true, y_prob, threshold=operating_threshold)
        print(f"\n  --- Operating threshold {operating_threshold:.4f} "
              f"(selected on VALIDATION set for target sensitivity "
              f"{operating_threshold_info['target_sensitivity']:.0%}) ---")
        print("  " + format_metrics_report(metrics).replace("\n", "\n  "))
        if not operating_threshold_info.get("target_met_on_validation", True):
            print("  NOTE: this model could not reach the target sensitivity even on the "
                  "validation set — the gap below is a genuine model-quality limitation, "
                  "not a thresholding artifact.")
    else:
        print("\n  No tuned operating threshold available for this model — run "
              "scripts/select_operating_thresholds.py first. Reporting the 0.5 "
              "reference metrics as the only numbers, clearly labeled as such.")

    auc_point, auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=n_bootstrap)
    pr_point, pr_lo, pr_hi = bootstrap_ci(y_true, y_prob, average_precision_score, n_bootstrap=n_bootstrap)
    print(f"\n  ROC-AUC: {auc_point:.4f}  95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]" if auc_lo is not None
          else f"\n  ROC-AUC: {auc_point:.4f}  (CI unavailable)")
    print(f"  PR-AUC:  {pr_point:.4f}  95% CI: [{pr_lo:.4f}, {pr_hi:.4f}]" if pr_lo is not None
          else f"  PR-AUC:  {pr_point:.4f}  (CI unavailable)")

    calib_path = os.path.join(figures_dir, f"phase14_calibration_{name}.png")
    calib = plot_reliability_diagram(y_true, y_prob, calib_path, title=f"{name} — Test Set Calibration")
    print(f"  Brier score: {calib['brier_score']:.4f}")
    print(f"  Saved calibration figure: {calib_path}")

    return {
        "name": name, "n_test": len(y_true),
        "metrics": metrics, "metrics_default_threshold": metrics_default,
        "operating_threshold": operating_threshold,
        "operating_threshold_info": operating_threshold_info,
        "roc_auc_ci": (auc_point, auc_lo, auc_hi), "pr_auc_ci": (pr_point, pr_lo, pr_hi),
        "brier_score": calib["brier_score"],
    }


def evaluate_detector(checkpoint_path, bbox_test_df, device, figures_dir):
    print(f"\n{'=' * 70}\nEvaluating LOCALIZATION (detector) on TEST set\n{'=' * 70}")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("  No detector checkpoint found — skipping.")
        return None

    from src.data.detection_dataset import LesionDetectionDataset, detection_collate_fn
    from src.models.detector import build_detector
    from src.utils.detection_metrics import compute_detection_metrics

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_detector(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = LesionDetectionDataset(bbox_test_df, config)
    if len(dataset) == 0:
        print("  No usable test images with bounding boxes — skipping.")
        return None
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)

    all_predictions, all_ground_truths = [], []
    with torch.no_grad():
        for images, targets in loader:
            images_device = [img.to(device) for img in images]
            predictions = model(images_device)
            for pred, tgt in zip(predictions, targets):
                all_predictions.append({"boxes": pred["boxes"].cpu(), "scores": pred["scores"].cpu()})
                all_ground_truths.append({"boxes": tgt["boxes"].cpu()})

    metrics = compute_detection_metrics(all_predictions, all_ground_truths)
    print(f"  Test set size: {len(dataset)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def evaluate_segmentation(checkpoint_path, bbox_test_df, device, figures_dir):
    print(f"\n{'=' * 70}\nEvaluating SEGMENTATION on TEST set\n{'=' * 70}")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("  No segmentation checkpoint found — skipping.")
        return None

    from src.data.segmentation_dataset import LesionSegmentationDataset
    from src.models.segmentation_model import build_segmentation_model
    from src.utils.segmentation_metrics import compute_segmentation_metrics

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_segmentation_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = LesionSegmentationDataset(bbox_test_df, config)
    if len(dataset) == 0:
        print("  No usable test images with masks — skipping.")
        return None
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_dice, all_iou = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            batch_metrics = compute_segmentation_metrics(logits, masks)
            all_dice.append(batch_metrics["dice"])
            all_iou.append(batch_metrics["iou"])

    print(f"  Test set size: {len(dataset)}")
    print(f"  Dice: {np.mean(all_dice):.4f}")
    print(f"  IoU:  {np.mean(all_iou):.4f}")
    return {"dice": float(np.mean(all_dice)), "iou": float(np.mean(all_iou))}


def evaluate_explainability(checkpoint_path, test_df, bbox_lookup, device, figures_dir, n_examples=6):
    print(f"\n{'=' * 70}\nEvaluating EXPLAINABILITY (qualitative) on TEST set\n{'=' * 70}")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("  No classifier checkpoint found — skipping.")
        return None

    from src.data.image_io import load_image
    from src.preprocessing.basic_preprocess import preprocess_image
    from src.data.bbox_utils import transform_bbox_for_preprocessing
    from src.explainability.gradcam import GradCAM, overlay_heatmap

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_classifier(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gradcam = GradCAM(model, backbone_name=config["model"]["backbone"])
    sample = test_df.dropna(subset=["image_file_path_resolved", "pathology_binary"]).sample(
        n=min(n_examples, len(test_df)), random_state=config["project"]["seed"])

    overlap_scores = []
    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]

    for i, (_, row) in enumerate(sample.iterrows()):
        img = load_image(row["image_file_path_resolved"])
        result = preprocess_image(img, config, run_quality_gate=False)
        processed = result["processed"]
        crop_bbox = result["bbox"] or (0, 0, img.shape[1], img.shape[0])
        image_size = tuple(config["preprocessing"]["image_size"])

        tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1).float().unsqueeze(0).to(device)
        heatmap, pred_class, prob = gradcam.generate(tensor)
        overlay = overlay_heatmap(processed, heatmap)

        axes[i].imshow(overlay)
        gt_box = None
        image_id = row.get("image_id")
        if bbox_lookup is not None and image_id in bbox_lookup:
            b = bbox_lookup[image_id]
            gt_box = transform_bbox_for_preprocessing(
                (b["bbox_x"], b["bbox_y"], b["bbox_w"], b["bbox_h"]), crop_bbox, image_size)
            x, y, w, h = gt_box
            axes[i].add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))

            # Simple quantitative overlap: fraction of the GT box area where
            # the heatmap is "hot" (top 25% of its own range).
            hot_mask = heatmap > (0.75 * heatmap.max()) if heatmap.max() > 0 else np.zeros_like(heatmap)
            gx0, gy0, gw, gh = int(max(x, 0)), int(max(y, 0)), int(w), int(h)
            gx1, gy1 = min(gx0 + gw, hot_mask.shape[1]), min(gy0 + gh, hot_mask.shape[0])
            if gx1 > gx0 and gy1 > gy0:
                box_region = hot_mask[gy0:gy1, gx0:gx1]
                overlap_frac = float(box_region.mean())
                overlap_scores.append(overlap_frac)

        axes[i].set_title(f"Pred: {['benign','malignant'][pred_class]} ({prob:.2f})", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "phase14_explainability_test_examples.png")
    plt.savefig(fig_path, dpi=150)
    print(f"  Saved qualitative figure: {fig_path}")
    if overlap_scores:
        print(f"  Mean fraction of ground-truth lesion box covered by high-attention "
              f"heatmap pixels: {np.mean(overlap_scores):.2%} (n={len(overlap_scores)})")
    print("  REMINDER: Grad-CAM overlays are a model explanation, not proof of correctness.")
    return {"overlap_scores": overlap_scores, "figure_path": fig_path}


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 14: FINAL evaluation (locked test set)")
    parser.add_argument("--i-understand-this-locks-the-test-set", action="store_true", required=True,
                         help="Required acknowledgment — this script evaluates on the test set, "
                              "which should only happen once, at the very end.")
    parser.add_argument("--raw-images-dir", type=str, default=None,
                         help="Needed to build bbox_metadata_test.csv if it doesn't exist yet "
                              "(required for localization/segmentation/explainability-overlap evaluation).")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--lesion-crop-checkpoint", type=str, default=None)
    parser.add_argument("--multimodal-checkpoint", type=str, default=None)
    parser.add_argument("--detector-checkpoint", type=str, default=None)
    parser.add_argument("--segmentation-checkpoint", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-localization", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-explainability", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed = config["project"]["seed"]
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_dir = config["paths"]["data_metadata"]
    figures_dir = config["paths"]["figures_dir"]
    reports_dir = config["paths"]["reports_dir"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE 14 — FINAL EVALUATION ON THE LOCKED TEST SET")
    print("This is the ONLY time the test set should be touched in this project.")
    print("=" * 70)

    test_csv = os.path.join(metadata_dir, "test_split.csv")
    if not os.path.exists(test_csv):
        print(f"No test split found at {test_csv}. Run Phase 3 first.")
        return
    test_df = pd.read_csv(test_csv)
    print(f"\nLoaded locked test set: {len(test_df)} rows, "
          f"{test_df['patient_id'].nunique()} patients.\n")

    thresholds_path = os.path.join(metadata_dir, "operating_thresholds.json")
    operating_thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            operating_thresholds = json.load(f)
        print(f"Loaded operating thresholds from {thresholds_path} "
              f"(selected on validation data by select_operating_thresholds.py).\n")
    else:
        print(f"No {thresholds_path} found — classifiers below will be reported at the "
              f"default 0.5 threshold ONLY. Run scripts/select_operating_thresholds.py "
              f"first for a sensitivity-appropriate operating point.\n")

    results = {}

    # --- Classification: baseline, lesion-crop, multimodal ---
    baseline_ckpt = args.baseline_checkpoint or find_best_checkpoint("6_baseline", models_dir, fallback_prefix="baseline_")
    results["baseline"] = evaluate_classifier(
        "whole_image_baseline", baseline_ckpt, test_df, {}, device,
        "whole_image", figures_dir, args.n_bootstrap,
        operating_threshold_info=operating_thresholds.get("whole_image_baseline"))

    bbox_test_df = None
    bbox_test_csv = os.path.join(metadata_dir, "bbox_metadata_test.csv")
    if os.path.exists(bbox_test_csv):
        bbox_test_df = pd.read_csv(bbox_test_csv)
        bbox_test_df = bbox_test_df[bbox_test_df["has_bbox"] == True]  # noqa: E712
    elif args.raw_images_dir:
        print("\nbbox_metadata_test.csv not found — building it now (FIRST TIME touching test-set masks)...")
        sys.path.insert(0, os.path.dirname(__file__))
        from build_bbox_metadata import process_row
        rows = []
        for _, row in test_df.iterrows():
            result = process_row(row, args.raw_images_dir)
            result["image_id"] = row.get("image_id")
            result["patient_id"] = row.get("patient_id")
            rows.append(result)
        bbox_test_df = pd.DataFrame(rows)
        bbox_test_df.to_csv(bbox_test_csv, index=False)
        n_ok = int(bbox_test_df["has_bbox"].sum())
        print(f"Built {bbox_test_csv}: {n_ok}/{len(bbox_test_df)} boxes extracted.")
        bbox_test_df = bbox_test_df[bbox_test_df["has_bbox"] == True]  # noqa: E712
    else:
        print("\nNo bbox_metadata_test.csv and no --raw-images-dir given — "
              "lesion-crop/localization/segmentation test evaluation will be skipped.")

    if bbox_test_df is not None and "pathology_binary" not in bbox_test_df.columns:
        bbox_test_df = bbox_test_df.merge(test_df[["image_id", "pathology_binary"]], on="image_id", how="left")

    if bbox_test_df is not None:
        lesion_crop_ckpt = args.lesion_crop_checkpoint or find_best_checkpoint("9_lesion_crop", models_dir)
        results["lesion_crop"] = evaluate_classifier(
            "lesion_crop", lesion_crop_ckpt, bbox_test_df, {}, device,
            "lesion_crop", figures_dir, args.n_bootstrap,
            operating_threshold_info=operating_thresholds.get("lesion_crop"))

    multimodal_ckpt = args.multimodal_checkpoint or find_best_checkpoint("13_multimodal", models_dir)
    if multimodal_ckpt and os.path.exists(multimodal_ckpt):
        checkpoint = torch.load(multimodal_ckpt, map_location=device, weights_only=False)
        cat_cols = checkpoint.get("tabular_categorical_cols", [])
        num_cols = checkpoint.get("tabular_numeric_cols", [])
        train_csv = os.path.join(metadata_dir, "train_split.csv")
        train_df_for_refit = pd.read_csv(train_csv)
        tabular_pp = TabularPreprocessor(cat_cols, num_cols)
        results["multimodal"] = evaluate_classifier(
            "multimodal", multimodal_ckpt, test_df, {"train_df_for_refit": train_df_for_refit}, device,
            "multimodal", figures_dir, args.n_bootstrap, tabular_pp=tabular_pp,
            operating_threshold_info=operating_thresholds.get("multimodal"))

    # --- Localization ---
    if not args.skip_localization and bbox_test_df is not None:
        detector_ckpt = args.detector_checkpoint or find_best_checkpoint("7_localization", models_dir)
        if detector_ckpt is None:
            # detector runs weren't logged with a 'phase' field in the registry by Phase 7's script;
            # fall back to letting the user pass it explicitly.
            print("\nNo detector checkpoint auto-found in registry — pass --detector-checkpoint explicitly to include it.")
        else:
            results["localization"] = evaluate_detector(detector_ckpt, bbox_test_df, device, figures_dir)
    elif args.detector_checkpoint:
        results["localization"] = evaluate_detector(args.detector_checkpoint, bbox_test_df, device, figures_dir)

    # --- Segmentation ---
    if not args.skip_segmentation and bbox_test_df is not None and args.segmentation_checkpoint:
        results["segmentation"] = evaluate_segmentation(args.segmentation_checkpoint, bbox_test_df, device, figures_dir)

    # --- Explainability ---
    if not args.skip_explainability and baseline_ckpt:
        bbox_lookup = None
        if bbox_test_df is not None:
            bbox_lookup = bbox_test_df.set_index("image_id")[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_dict("index")
        results["explainability"] = evaluate_explainability(
            baseline_ckpt, test_df, bbox_lookup, device, figures_dir)

    # --- Robustness (explicitly noted as skipped, not silently omitted) ---
    print(f"\n{'=' * 70}\nROBUSTNESS EVALUATION\n{'=' * 70}")
    print("Skipped — no external dataset (e.g. INbreast) was downloaded for this project.")
    print("This is a genuine, honest limitation to state in the report, not a fabricated result.")

    # --- Final written report ---
    report_path = os.path.join(reports_dir, "FINAL_EVALUATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# MammoTwin — Phase 14 Final Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Locked test set: {len(test_df)} rows, {test_df['patient_id'].nunique()} patients. ")
        f.write("This test set was untouched by any training or model-selection decision "
                "prior to this evaluation.\n\n")

        f.write("## Classification\n\n")
        f.write("ROC-AUC/PR-AUC are threshold-independent (rank-based) and unaffected by the "
                "cutoff below. Sensitivity/specificity/balanced accuracy ARE threshold-dependent "
                "and are reported at TWO cutoffs so the effect of proper threshold selection is "
                "visible rather than hidden: the untuned default (0.5) that earlier reports used, "
                "and the operating threshold selected on the VALIDATION set only "
                "(see `select_operating_thresholds.py`).\n\n")
        f.write("| Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier "
                "| Sens. @0.5 | Spec. @0.5 "
                "| Operating thr. | Sens. @thr. | Spec. @thr. | Bal.Acc. @thr. | Thr. selected on |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for key in ["baseline", "lesion_crop", "multimodal"]:
            r = results.get(key)
            if r is None:
                continue
            auc_p, auc_lo, auc_hi = r["roc_auc_ci"]
            pr_p, pr_lo, pr_hi = r["pr_auc_ci"]
            m_default = r["metrics_default_threshold"]
            m_tuned = r["metrics"]
            info = r["operating_threshold_info"]
            if info is not None:
                thr_str = f"{r['operating_threshold']:.3f}"
                sel_str = (f"val (target {info['target_sensitivity']:.0%}"
                           f"{', NOT MET' if not info.get('target_met_on_validation', True) else ''})")
            else:
                thr_str, sel_str = "0.500 (untuned)", "n/a — run select_operating_thresholds.py"
            f.write(f"| {r['name']} | {auc_p:.3f} [{auc_lo:.3f}, {auc_hi:.3f}] "
                    f"| {pr_p:.3f} [{pr_lo:.3f}, {pr_hi:.3f}] | {r['brier_score']:.3f} "
                    f"| {m_default['sensitivity']:.3f} | {m_default['specificity']:.3f} "
                    f"| {thr_str} | {m_tuned['sensitivity']:.3f} | {m_tuned['specificity']:.3f} "
                    f"| {m_tuned['balanced_accuracy']:.3f} | {sel_str} |\n")

        if results.get("localization"):
            f.write("\n## Localization\n\n")
            for k, v in results["localization"].items():
                f.write(f"- {k}: {v:.4f}\n")

        if results.get("segmentation"):
            f.write("\n## Segmentation\n\n")
            for k, v in results["segmentation"].items():
                f.write(f"- {k}: {v:.4f}\n")

        if results.get("explainability") and results["explainability"]["overlap_scores"]:
            f.write("\n## Explainability (qualitative + overlap)\n\n")
            f.write(f"Mean fraction of ground-truth lesion box covered by high-attention "
                    f"heatmap pixels: {np.mean(results['explainability']['overlap_scores']):.2%}\n")
            f.write(f"See figure: {results['explainability']['figure_path']}\n")

        f.write("\n## Robustness\n\n")
        f.write("Not evaluated — no external dataset was available for this project. "
                "Stated here as an explicit limitation.\n")

    print(f"\n\nSaved final report: {report_path}")
    print("\n" + "=" * 70)
    print("PHASE 14 FINAL EVALUATION COMPLETE.")
    print("The test set has now been used. Do not use it again for model selection or tuning.")
    print("=" * 70)


if __name__ == "__main__":
    main()