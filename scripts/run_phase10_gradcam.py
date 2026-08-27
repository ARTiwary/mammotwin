"""
Phase 10 deliverable: Grad-CAM explainability for the whole-image classifier.

Loads a trained Phase 6 checkpoint, picks a handful of validation examples
(a mix of correct and incorrect predictions where possible), and produces:
  - the preprocessed input image
  - a Grad-CAM heatmap overlay, with the ground-truth lesion box drawn on
    top where available (from Phase 7's bbox ground truth), so you can see
    directly whether the model's attention overlaps the actual lesion

Also runs an OPTIONAL, clearly-labeled-experimental counterfactual check:
masking the most-attended region and re-running the model, to see how much
its prediction changes.

IMPORTANT: a Grad-CAM heatmap shows where the model's evidence was
concentrated. It is a model explanation, not proof of cancer, and does not
by itself validate that a prediction is correct.

Real data:
    python scripts/run_phase10_gradcam.py --checkpoint models/baseline_resnet50_<timestamp>.pt

Demo mode (synthetic data + an UNTRAINED model — only proves the mechanics
run correctly; the heatmaps carry no real diagnostic meaning):
    python scripts/run_phase10_gradcam.py --demo
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

from src.utils.config import load_config, set_global_seed
from src.data.image_io import load_image, generate_synthetic_mammogram
from src.preprocessing.basic_preprocess import preprocess_image
from src.data.bbox_utils import transform_bbox_for_preprocessing
from src.models.classifier import build_classifier
from src.explainability.gradcam import GradCAM, overlay_heatmap

LABEL_NAMES = {0: "benign", 1: "malignant"}


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_classifier(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def prepare_example(row, config, path_col, bbox_lookup=None):
    """Preprocess one image and, if bbox ground truth is available for it,
    transform the box into the SAME preprocessed coordinate frame (reusing
    Phase 7's crop+resize transform) so it can be drawn on the heatmap."""
    img = load_image(row[path_col])
    result = preprocess_image(img, config, run_quality_gate=False)
    processed = result["processed"]
    crop_bbox = result["bbox"] or (0, 0, img.shape[1], img.shape[0])
    image_size = tuple(config["preprocessing"]["image_size"])

    gt_box_in_processed_frame = None
    if bbox_lookup is not None:
        image_id = row.get("image_id")
        if image_id in bbox_lookup:
            b = bbox_lookup[image_id]
            transformed = transform_bbox_for_preprocessing(
                (b["bbox_x"], b["bbox_y"], b["bbox_w"], b["bbox_h"]), crop_bbox, image_size
            )
            gt_box_in_processed_frame = transformed

    tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1).float().unsqueeze(0)  # (1, 3, H, W)
    return processed, tensor, gt_box_in_processed_frame


def counterfactual_mask_experiment(model, gradcam, input_tensor, target_class, threshold=0.5):
    """Experimental: mask the most-attended region (heatmap above threshold)
    and see how much the target class's predicted probability changes."""
    heatmap, _, original_prob = gradcam.generate(input_tensor, target_class=target_class)
    mask = (heatmap > threshold * heatmap.max()) if heatmap.max() > 0 else np.zeros_like(heatmap)

    masked_input = input_tensor.clone()
    mean_val = input_tensor.mean().item()
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(input_tensor.device)
    masked_input[0, :, mask_tensor.bool()] = mean_val

    with torch.no_grad():
        masked_output = model(masked_input)
        masked_prob = torch.softmax(masked_output, dim=1)[0, target_class].item()

    return {
        "original_prob": original_prob,
        "masked_prob": masked_prob,
        "prob_drop": original_prob - masked_prob,
        "mask_fraction": float(mask.mean()),
    }


def generate_demo_examples(config, n=6, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        img = generate_synthetic_mammogram(size=400, seed=i)
        import cv2
        path = f"/tmp/mammotwin_phase10_demo_{i}.jpg"
        cv2.imwrite(path, img)
        rows.append({
            "image_id": f"demo_{i}",
            "image_file_path_resolved": path,
            "pathology_binary": rng.choice(["benign", "malignant"]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="MammoTwin Phase 10: Grad-CAM explainability")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a trained Phase 6 classifier checkpoint (.pt)")
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--bbox-val-csv", type=str, default=None)
    parser.add_argument("--path-col", type=str, default="image_file_path_resolved")
    parser.add_argument("--n-examples", type=int, default=6)
    parser.add_argument("--run-counterfactual", action="store_true",
                         help="Also run the experimental mask-and-reclassify check")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.demo:
        print("Running Phase 10 in DEMO mode: synthetic data + an UNTRAINED model.")
        print("Heatmaps here verify the MECHANICS only — they carry no real "
              "diagnostic meaning since the model has random weights.\n")
        config = load_config(args.config) if args.config else load_config()
        set_global_seed(config["project"]["seed"])
        model = build_classifier(config).to(device)
        model.eval()
        df = generate_demo_examples(config, n=args.n_examples)
        bbox_lookup = None
    else:
        if not args.checkpoint:
            print("Pass --checkpoint path/to/your/phase6/checkpoint.pt, or use --demo.")
            return
        model, config = load_checkpoint_model(args.checkpoint, device)
        metadata_dir = config["paths"]["data_metadata"]
        val_csv = args.val_csv or os.path.join(metadata_dir, "val_split.csv")
        if not os.path.exists(val_csv):
            print(f"No val CSV at {val_csv}.")
            return
        df = pd.read_csv(val_csv).dropna(subset=[args.path_col, "pathology_binary"])
        df = df.sample(n=min(len(df), 40), random_state=config["project"]["seed"]).reset_index(drop=True)

        bbox_lookup = None
        bbox_val_csv = args.bbox_val_csv or os.path.join(metadata_dir, "bbox_metadata_val.csv")
        if os.path.exists(bbox_val_csv):
            bbox_df = pd.read_csv(bbox_val_csv)
            bbox_df = bbox_df[bbox_df["has_bbox"] == True]  # noqa: E712
            bbox_lookup = bbox_df.set_index("image_id")[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_dict("index")
            print(f"Loaded {len(bbox_lookup)} ground-truth boxes for overlay comparison.\n")

    backbone_name = config["model"]["backbone"]
    gradcam = GradCAM(model, backbone_name=backbone_name)

    # --- Run inference over the sample to categorize examples ---
    label_map = {"benign": 0, "malignant": 1}
    scored_rows = []
    for _, row in df.iterrows():
        try:
            processed, tensor, gt_box = prepare_example(row, config, args.path_col, bbox_lookup)
        except Exception as e:
            continue
        tensor = tensor.to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = int(probs.argmax(dim=1).item())
        true_class = label_map.get(row["pathology_binary"])
        scored_rows.append({
            "row": row, "processed": processed, "tensor": tensor, "gt_box": gt_box,
            "true_class": true_class, "pred_class": pred_class,
            "pred_prob": float(probs[0, pred_class].item()),
            "correct": (pred_class == true_class),
        })

    if not scored_rows:
        print("No usable examples found.")
        return

    # Prefer a MIX of correct and incorrect predictions for a more informative figure.
    correct_examples = [r for r in scored_rows if r["correct"]]
    incorrect_examples = [r for r in scored_rows if not r["correct"]]
    n = min(args.n_examples, len(scored_rows))
    n_incorrect = min(len(incorrect_examples), n // 2)
    selected = incorrect_examples[:n_incorrect] + correct_examples[:n - n_incorrect]
    selected = selected[:n]

    print(f"Selected {len(selected)} examples ({sum(1 for s in selected if not s['correct'])} "
          f"incorrect, {sum(1 for s in selected if s['correct'])} correct) for visualization.\n")

    # --- Generate figure ---
    fig, axes = plt.subplots(2, len(selected), figsize=(4 * len(selected), 8))
    if len(selected) == 1:
        axes = axes.reshape(2, 1)

    counterfactual_results = []
    for i, ex in enumerate(selected):
        heatmap, target_class, prob = gradcam.generate(ex["tensor"], target_class=ex["pred_class"])
        overlay = overlay_heatmap(ex["processed"], heatmap)

        axes[0, i].imshow(ex["processed"], cmap="gray")
        if ex["gt_box"] is not None:
            x, y, w, h = ex["gt_box"]
            axes[0, i].add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                                     edgecolor="lime", facecolor="none"))
        axes[0, i].set_title("Input" + (" (+ GT box)" if ex["gt_box"] is not None else ""))
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        if ex["gt_box"] is not None:
            x, y, w, h = ex["gt_box"]
            axes[1, i].add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                                     edgecolor="lime", facecolor="none"))
        true_name = LABEL_NAMES.get(ex["true_class"], "?")
        pred_name = LABEL_NAMES.get(ex["pred_class"], "?")
        correctness = "correct" if ex["correct"] else "WRONG"
        axes[1, i].set_title(f"True: {true_name} | Pred: {pred_name} ({ex['pred_prob']:.2f}) [{correctness}]",
                              fontsize=9)
        axes[1, i].axis("off")

        if args.run_counterfactual:
            cf = counterfactual_mask_experiment(model, gradcam, ex["tensor"], ex["pred_class"])
            counterfactual_results.append(cf)

    plt.tight_layout()
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "phase10_gradcam_examples.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved Grad-CAM visualization: {fig_path}")

    if args.run_counterfactual and counterfactual_results:
        print("\n=== EXPERIMENTAL: counterfactual masking check ===")
        print("(Masking the most-attended region and re-running the model — "
              "an exploratory sanity check, not a validated causal method.)")
        for i, cf in enumerate(counterfactual_results):
            print(f"  Example {i+1}: prob {cf['original_prob']:.3f} -> {cf['masked_prob']:.3f} "
                  f"(drop: {cf['prob_drop']:+.3f}, masked {cf['mask_fraction']:.1%} of pixels)")

    print("\n" + "=" * 70)
    print("REMINDER: a Grad-CAM heatmap shows where the model's evidence was")
    print("concentrated. It is a model explanation, not proof of cancer, and")
    print("does not by itself validate that a prediction is correct.")
    print("=" * 70)
    print("\nPhase 10 explainability COMPLETE.")


if __name__ == "__main__":
    main()