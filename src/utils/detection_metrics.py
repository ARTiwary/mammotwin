"""
Phase 7: lightweight object-detection metrics (mAP, IoU), implemented
directly on torchvision.ops.box_iou instead of pulling in torchmetrics
(which drags in scipy — and scipy's compiled qhull extension has been
observed to be blocked by some Windows enterprise "Application Control"
security policies, an environment issue outside our control).

Standard COCO-style single-class Average Precision:
  1. Collect all predictions across all images, sorted by confidence score.
  2. Walk down the sorted list; a prediction is a True Positive if it has
     IoU >= threshold with an as-yet-unmatched ground-truth box in the
     SAME image, otherwise it's a False Positive.
  3. Build the precision/recall curve from cumulative TP/FP counts.
  4. AP = area under the precision-envelope curve (all-points interpolation,
     the same method used by COCO/Pascal VOC's mAP definition).

mAP (COCO's primary "map" metric) is the mean of AP computed at IoU
thresholds 0.50, 0.55, ..., 0.95.
"""

import numpy as np
import torch
from torchvision.ops import box_iou


def compute_ap_at_iou(all_predictions: list, all_ground_truths: list, iou_threshold: float) -> float:
    """
    all_predictions:   list of {'boxes': Tensor[N,4], 'scores': Tensor[N]}, one per image
    all_ground_truths: list of {'boxes': Tensor[M,4]}, one per image, SAME ORDER as predictions
    """
    # Flatten all predictions across images into one sorted-by-score list,
    # remembering which image and which GT boxes are available to match.
    flat_preds = []  # (score, image_idx, box)
    total_gt = 0
    for img_idx, gt in enumerate(all_ground_truths):
        total_gt += len(gt["boxes"])
    for img_idx, pred in enumerate(all_predictions):
        for box, score in zip(pred["boxes"], pred["scores"]):
            flat_preds.append((float(score), img_idx, box))

    if total_gt == 0:
        return 0.0
    if len(flat_preds) == 0:
        return 0.0

    flat_preds.sort(key=lambda x: x[0], reverse=True)

    matched_gt = {img_idx: torch.zeros(len(gt["boxes"]), dtype=torch.bool)
                  for img_idx, gt in enumerate(all_ground_truths)}

    tp = np.zeros(len(flat_preds))
    fp = np.zeros(len(flat_preds))

    for i, (score, img_idx, box) in enumerate(flat_preds):
        gt_boxes = all_ground_truths[img_idx]["boxes"]
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)  # (M,)
        best_iou, best_gt_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not matched_gt[img_idx][best_gt_idx]:
            tp[i] = 1
            matched_gt[img_idx][best_gt_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-9)

    # All-points precision envelope (standard COCO/VOC AP definition):
    # precision at each recall level is the MAX precision at that recall
    # or any higher recall — makes the PR curve monotonically decreasing.
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_change_idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[recall_change_idx + 1] - recalls[recall_change_idx]) *
                precisions[recall_change_idx + 1])
    return float(ap)


def compute_mean_iou(all_predictions: list, all_ground_truths: list) -> float:
    """Average IoU between each image's TOP-SCORED prediction and its best-
    matching ground truth — a simple, intuitive localization-quality signal
    to report alongside mAP."""
    ious = []
    for pred, gt in zip(all_predictions, all_ground_truths):
        if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
            continue
        top_idx = pred["scores"].argmax()
        top_box = pred["boxes"][top_idx].unsqueeze(0)
        iou = box_iou(top_box, gt["boxes"]).max().item()
        ious.append(iou)
    return float(np.mean(ious)) if ious else 0.0


def compute_detection_metrics(all_predictions: list, all_ground_truths: list) -> dict:
    """Returns mAP averaged over IoU 0.5:0.95 (COCO's headline 'map'),
    mAP@50, mAP@75, and mean top-1 IoU."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [compute_ap_at_iou(all_predictions, all_ground_truths, t) for t in iou_thresholds]

    return {
        "map": float(np.mean(aps)),
        "map_50": compute_ap_at_iou(all_predictions, all_ground_truths, 0.5),
        "map_75": compute_ap_at_iou(all_predictions, all_ground_truths, 0.75),
        "mean_top1_iou": compute_mean_iou(all_predictions, all_ground_truths),
    }