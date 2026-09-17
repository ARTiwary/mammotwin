"""
Computes accuracy (and full metrics + confusion matrix) on BOTH the
training set and the locked test set, for every classifier -- something
none of the existing scripts do (training scripts only ever print
validation metrics; Phase 14 only ever touches the test set).

Comparing train vs. test accuracy directly is the standard way to
diagnose overfitting: a large gap (e.g. train 95% vs test 60%) means the
model has memorized the training data rather than learned generalizable
patterns. A small gap is a good sign regardless of the absolute accuracy
level.

IMPORTANT: this script reads the test set but does NOT use it for any
model selection or tuning decision -- it only reports a number for an
already-finalized model, exactly like Phase 14 does. It is safe to run
after Phase 14 without violating the "touch the test set once" rule,
since no decision is made based on what it finds.

Usage:
    python scripts/report_train_test_accuracy.py \\
        --baseline-checkpoint models\\baseline_resnet50_20260913_175719.pt \\
        --lesion-crop-checkpoint models\\lesion_crop_resnet50_20260913_191937.pt \\
        --multimodal-checkpoint models\\multimodal_resnet50_20260913_192536.pt

If a --*-checkpoint flag is omitted, it auto-discovers from
models/registry.json the same way run_phase14_final_evaluation.py does.
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
from src.data.dataset import MammogramDataset
from src.data.crop_dataset import LesionCropDataset
from src.data.multimodal_dataset import MultimodalDataset
from src.data.tabular_preprocessing import TabularPreprocessor
from src.models.classifier import build_classifier
from src.models.multimodal_model import build_multimodal_model
from run_phase14_final_evaluation import find_best_checkpoint, run_classifier_inference


def evaluate_on_split(name, split_label, checkpoint_path, df, device, figures_dir,
                       dataset_type, tabular_pp=None, operating_threshold=0.5):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"  [{name} / {split_label}] No checkpoint found — skipping.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    if dataset_type == "multimodal":
        model = build_multimodal_model(config, tabular_input_dim=tabular_pp.output_dim).to(device)
        dataset = MultimodalDataset(df, config, tabular_pp)
        multimodal = True
    elif dataset_type == "lesion_crop":
        model = build_classifier(config).to(device)
        dataset = LesionCropDataset(df, config)
        multimodal = False
    else:
        model = build_classifier(config).to(device)
        dataset = MammogramDataset(df, config)
        multimodal = False

    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    y_true, y_prob = run_classifier_inference(model, loader, device, multimodal=multimodal)

    metrics = compute_classification_metrics(y_true, y_prob, threshold=operating_threshold)
    print(f"\n  --- {name} / {split_label} (n={len(y_true)}, threshold={operating_threshold:.3f}) ---")
    print("  " + format_metrics_report(metrics).replace("\n", "\n  "))

    cm_path = os.path.join(figures_dir, f"train_test_confusion_matrix_{name}_{split_label}.png")
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path,
                           title=f"{name} — {split_label} set (threshold={operating_threshold:.3f})")
    print(f"  Saved: {cm_path}")

    return {"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"], "n": len(y_true)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--lesion-crop-checkpoint", type=str, default=None)
    parser.add_argument("--multimodal-checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_dir = config["paths"]["data_metadata"]
    models_dir = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]

    train_df = pd.read_csv(os.path.join(metadata_dir, "train_split.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test_split.csv"))

    thresholds_path = os.path.join(metadata_dir, "operating_thresholds.json")
    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    print("=" * 70)
    print("TRAIN vs. TEST ACCURACY — overfitting check")
    print("(train accuracy is reported for diagnostic purposes only; it is")
    print(" NEVER used to pick a model or tune anything)")
    print("=" * 70)

    summary_rows = []

    # --- Whole-image baseline ---
    ckpt = args.baseline_checkpoint or find_best_checkpoint("6_baseline", models_dir, fallback_prefix="baseline_")
    thr = thresholds.get("whole_image_baseline", {}).get("operating_threshold", 0.5)
    train_r = evaluate_on_split("whole_image_baseline", "TRAIN", ckpt, train_df, device, figures_dir,
                                 "baseline", operating_threshold=thr)
    test_r = evaluate_on_split("whole_image_baseline", "TEST", ckpt, test_df, device, figures_dir,
                                "baseline", operating_threshold=thr)
    if train_r and test_r:
        summary_rows.append(("whole_image_baseline", train_r, test_r))

    # --- Lesion-crop ---
    train_bbox_csv = os.path.join(metadata_dir, "bbox_metadata_train.csv")
    test_bbox_csv = os.path.join(metadata_dir, "bbox_metadata_test.csv")
    if os.path.exists(train_bbox_csv) and os.path.exists(test_bbox_csv):
        train_bbox_df = pd.read_csv(train_bbox_csv)
        test_bbox_df = pd.read_csv(test_bbox_csv)
        ckpt = args.lesion_crop_checkpoint or find_best_checkpoint("9_lesion_crop", models_dir)
        thr = thresholds.get("lesion_crop", {}).get("operating_threshold", 0.5)
        train_r = evaluate_on_split("lesion_crop", "TRAIN", ckpt, train_bbox_df, device, figures_dir,
                                     "lesion_crop", operating_threshold=thr)
        test_r = evaluate_on_split("lesion_crop", "TEST", ckpt, test_bbox_df, device, figures_dir,
                                    "lesion_crop", operating_threshold=thr)
        if train_r and test_r:
            summary_rows.append(("lesion_crop", train_r, test_r))

    # --- Multimodal ---
    ckpt = args.multimodal_checkpoint or find_best_checkpoint("13_multimodal", models_dir)
    if ckpt and os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        cat_cols = checkpoint.get("tabular_categorical_cols", [])
        num_cols = checkpoint.get("tabular_numeric_cols", [])
        tabular_pp = TabularPreprocessor(cat_cols, num_cols)
        tabular_pp.fit(train_df)  # fit on TRAIN only, same as everywhere else in this project
        thr = thresholds.get("multimodal", {}).get("operating_threshold", 0.5)
        train_r = evaluate_on_split("multimodal", "TRAIN", ckpt, train_df, device, figures_dir,
                                     "multimodal", tabular_pp=tabular_pp, operating_threshold=thr)
        test_r = evaluate_on_split("multimodal", "TEST", ckpt, test_df, device, figures_dir,
                                    "multimodal", tabular_pp=tabular_pp, operating_threshold=thr)
        if train_r and test_r:
            summary_rows.append(("multimodal", train_r, test_r))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}  Interpretation")
    for name, tr, te in summary_rows:
        gap = tr["accuracy"] - te["accuracy"]
        if gap > 0.15:
            note = "Large gap -- likely overfitting"
        elif gap > 0.07:
            note = "Moderate gap -- some overfitting"
        else:
            note = "Small gap -- generalizing reasonably"
        print(f"{name:<22} {tr['accuracy']:>9.1%} {te['accuracy']:>9.1%} {gap:>+7.1%}  {note}")

    print("\nNote: accuracy at the operating threshold is expected to look worse than")
    print("at 0.5 for BOTH train and test, since the threshold is tuned for high")
    print("sensitivity, not accuracy. The train-vs-test GAP is what matters here,")
    print("not the absolute accuracy number.")


if __name__ == "__main__":
    main()