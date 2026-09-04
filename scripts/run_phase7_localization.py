"""
Phase 7 step 2: train and evaluate the Faster R-CNN lesion detector.

Prerequisite: run scripts/build_bbox_metadata.py first to produce
data/metadata/bbox_metadata.csv.

Real data:
    python scripts/run_phase7_localization.py --epochs 10

Detection is resolution-sensitive for small objects (lesions can be tiny
even within a full mammogram) — the classifier's 224x224 config.yaml
setting may be too small for good localization. Override just for this
script with --image-size, without touching config.yaml or Phase 6:
    python scripts/run_phase7_localization.py --epochs 20 --image-size 384

Demo mode (synthetic data, proves the full pipeline runs end-to-end):
    python scripts/run_phase7_localization.py --demo --epochs 2
"""

import os
import sys
import argparse
import copy
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

from src.utils.config import load_config, set_global_seed
from src.data.detection_dataset import LesionDetectionDataset, detection_collate_fn
from src.models.detector import build_detector


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float = 0.3):
    """Returns (mAP metrics dict, list of (image_tensor, pred_boxes, gt_boxes)
    examples for visualization)."""
    from src.utils.detection_metrics import compute_detection_metrics

    model.eval()
    all_predictions, all_ground_truths = [], []
    examples = []

    for images, targets in loader:
        images_device = [img.to(device) for img in images]
        predictions = model(images_device)

        for img, pred, tgt in zip(images, predictions, targets):
            all_predictions.append({"boxes": pred["boxes"].cpu(), "scores": pred["scores"].cpu()})
            all_ground_truths.append({"boxes": tgt["boxes"].cpu()})

            if len(examples) < 4:
                keep = pred["scores"] > score_threshold
                examples.append((
                    img.cpu(),
                    pred["boxes"][keep].cpu().numpy(),
                    pred["scores"][keep].cpu().numpy(),
                    tgt["boxes"].cpu().numpy(),
                ))

    results = compute_detection_metrics(all_predictions, all_ground_truths)
    return results, examples


def save_prediction_figure(examples, out_path):
    if not examples:
        return
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (img_tensor, pred_boxes, pred_scores, gt_boxes) in enumerate(examples):
        img = img_tensor.permute(1, 2, 0).numpy()[:, :, 0]  # take one channel (all 3 identical)
        axes[i].imshow(img, cmap="gray")

        for box in gt_boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="lime", facecolor="none", label="ground truth")
            axes[i].add_patch(rect)

        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            axes[i].add_patch(rect)
            axes[i].text(x1, max(y1 - 3, 0), f"{score:.2f}", color="red", fontsize=9)

        axes[i].set_title(f"Example {i+1} (green=GT, red=predicted)")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_demo_dataset(config, n=20, seed=42):
    """Reuse Phase 7's own bbox demo generator so this script can be tested
    without real data, then wrap it as a proper bbox_metadata-style df."""
    sys.path.insert(0, os.path.dirname(__file__))
    from build_bbox_metadata import generate_demo_rows, process_row

    df, raw_images_dir = generate_demo_rows(n=n, seed=seed)
    results = []
    for _, row in df.iterrows():
        result = process_row(row, raw_images_dir)
        result["image_id"] = row["image_id"]
        results.append(result)
    result_df = pd.DataFrame(results)
    return result_df[result_df["has_bbox"] == True].reset_index(drop=True)  # noqa: E712


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 7: lesion localization")
    parser.add_argument("--train-bbox-csv", type=str, default=None)
    parser.add_argument("--val-bbox-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Was 4. The Faster R-CNN detector is more memory-hungry per "
                              "sample than the classifiers (full-res images + region "
                              "proposals), so raise this cautiously -- drop to 4 or 2 if "
                              "you hit a CUDA out-of-memory error.")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="Was defaulting to 0 (fully serial data loading) regardless of "
                              "config.yaml's num_workers:4 -- that mismatch was likely the "
                              "biggest hidden cause of slow training. Set to 0 if you hit "
                              "multiprocessing/pickling errors on Windows.")
    parser.add_argument("--image-size", type=int, default=None,
                         help="Override preprocessing.image_size (both dims) for THIS "
                              "script only — doesn't touch config.yaml or Phase 6's "
                              "classifier. Detection is resolution-sensitive for small "
                              "lesions; try 384 or 512 if the default (from config.yaml, "
                              "usually 224) gives poor localization.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    if args.image_size:
        config = copy.deepcopy(config)
        config["preprocessing"]["image_size"] = [args.image_size, args.image_size]
        print(f"Overriding image_size to {args.image_size}x{args.image_size} for detection.\n")

    seed = config["project"]["seed"]
    set_global_seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 7 in DEMO mode on synthetic data.\n")
        train_bbox_df = generate_demo_dataset(config, n=20, seed=seed)
        val_bbox_df = generate_demo_dataset(config, n=6, seed=seed + 1)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        train_csv = args.train_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_train.csv")
        val_csv = args.val_bbox_csv or os.path.join(metadata_dir, "bbox_metadata_val.csv")
        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            print(f"Could not find {train_csv} / {val_csv}.")
            print("Run: python scripts/build_bbox_metadata.py --split train --raw-images-dir ...")
            print("And: python scripts/build_bbox_metadata.py --split val --raw-images-dir ...")
            print("(or pass --demo)")
            return
        train_bbox_df = pd.read_csv(train_csv)
        val_bbox_df = pd.read_csv(val_csv)
        train_bbox_df = train_bbox_df[train_bbox_df["has_bbox"] == True]  # noqa: E712
        val_bbox_df = val_bbox_df[val_bbox_df["has_bbox"] == True]  # noqa: E712

    train_dataset = LesionDetectionDataset(train_bbox_df, config)
    val_dataset = LesionDetectionDataset(val_bbox_df, config)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    if len(train_dataset) < 4 or len(val_dataset) < 1:
        print("Too few images with valid bounding boxes to train/evaluate. "
              "Run build_bbox_metadata.py with a higher --limit (or no limit), "
              "or check the failure reasons from that script.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=detection_collate_fn)

    model = build_detector(config).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    run_id = f"detector_fasterrcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.4f}")

    checkpoint_path = os.path.join(models_dir, f"{run_id}.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    print(f"\nSaved checkpoint: {checkpoint_path}")

    print("\nEvaluating (IoU / mAP)...")
    map_results, examples = evaluate(model, val_loader, device)
    print("\n=== Detection Metrics (val set) ===")
    for k, v in map_results.items():
        print(f"  {k}: {v:.4f}")

    # --- Update registry ---
    # This was missing entirely: without a registry entry, Phase 14's
    # find_best_checkpoint("7_localization", ...) can never find this
    # checkpoint, no matter how many times a detector is trained here —
    # it was silently falling through to "no detector checkpoint found"
    # on every single run. Ranked by "val_map" since a detector has no
    # AUC to rank by.
    registry_path = os.path.join(models_dir, "registry.json")
    registry = {}
    if os.path.exists(registry_path) and os.path.getsize(registry_path) > 0:
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            registry = {}
    registry[run_id] = {
        "checkpoint_path": checkpoint_path,
        "trained_date": datetime.now().isoformat(timespec="seconds"),
        "val_map": map_results["map"], "val_map_50": map_results["map_50"],
        "val_mean_top1_iou": map_results["mean_top1_iou"], "seed": seed,
        "phase": "7_localization",
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registered in {registry_path} as '{run_id}' (val_map={map_results['map']:.4f}).")

    # NOTE: unlike Phase 8/9/13, this checkpoint is the FINAL epoch, not the
    # best-val-mAP epoch (there is currently no in-loop best-checkpoint
    # tracking for the detector). For a short run this rarely matters much,
    # but if you increase --epochs meaningfully, add best-checkpoint
    # tracking here the same way run_phase8_segmentation.py already does.

    fig_path = os.path.join(figures_dir, "phase7_predicted_boxes.png")
    save_prediction_figure(examples, fig_path)
    print(f"\nSaved prediction visualization: {fig_path}")

    print("\nPhase 7 localization COMPLETE.")


if __name__ == "__main__":
    main()