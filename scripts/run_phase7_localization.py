"""
Phase 7 step 2: train and evaluate the Faster R-CNN lesion detector.

Prerequisite: run scripts/build_bbox_metadata.py first to produce
data/metadata/bbox_metadata.csv.

Real data:
    python scripts/run_phase7_localization.py --epochs 10

Demo mode (synthetic data, proves the full pipeline runs end-to-end):
    python scripts/run_phase7_localization.py --demo --epochs 2
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import DataLoader, random_split

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
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    examples = []

    for images, targets in loader:
        images_device = [img.to(device) for img in images]
        predictions = model(images_device)

        preds_formatted = [{
            "boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu(),
        } for p in predictions]
        targets_formatted = [{
            "boxes": t["boxes"].cpu(), "labels": t["labels"].cpu(),
        } for t in targets]
        metric.update(preds_formatted, targets_formatted)

        for img, pred, tgt in zip(images, predictions, targets):
            if len(examples) < 4:
                keep = pred["scores"] > score_threshold
                examples.append((
                    img.cpu(),
                    pred["boxes"][keep].cpu().numpy(),
                    pred["scores"][keep].cpu().numpy(),
                    tgt["boxes"].cpu().numpy(),
                ))

    results = metric.compute()
    return {k: float(v) for k, v in results.items() if v.numel() == 1}, examples


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
    parser.add_argument("--bbox-metadata-csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    seed = config["project"]["seed"]
    set_global_seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 7 in DEMO mode on synthetic data.\n")
        bbox_df = generate_demo_dataset(config)
    else:
        metadata_dir = config["paths"]["data_metadata"]
        bbox_csv = args.bbox_metadata_csv or os.path.join(metadata_dir, "bbox_metadata.csv")
        if not os.path.exists(bbox_csv):
            print(f"No bbox metadata at {bbox_csv}. Run scripts/build_bbox_metadata.py first, "
                  f"or pass --demo.")
            return
        bbox_df = pd.read_csv(bbox_csv)
        bbox_df = bbox_df[bbox_df["has_bbox"] == True]  # noqa: E712

    full_dataset = LesionDetectionDataset(bbox_df, config)
    print(f"Total usable images with bounding boxes: {len(full_dataset)}")
    if len(full_dataset) < 4:
        print("Too few images with valid bounding boxes to train/evaluate. "
              "Run build_bbox_metadata.py on more data, or check the failure reasons.")
        return

    n_val = max(1, int(len(full_dataset) * args.val_fraction))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    print(f"Train: {n_train} | Val: {n_val}\n")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=detection_collate_fn)

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

    fig_path = os.path.join(figures_dir, "phase7_predicted_boxes.png")
    save_prediction_figure(examples, fig_path)
    print(f"\nSaved prediction visualization: {fig_path}")

    print("\nPhase 7 localization COMPLETE.")


if __name__ == "__main__":
    main()