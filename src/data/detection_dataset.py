"""
Phase 7: detection Dataset. Loads a full mammogram, runs it through the
SAME Phase 4 preprocessing pipeline used for classification, and carries
its ground-truth bounding box through the same crop+resize transform so
the box stays correctly aligned with the preprocessed image.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import preprocess_image
from src.data.bbox_utils import transform_bbox_for_preprocessing


class LesionDetectionDataset(Dataset):
    def __init__(self, bbox_metadata_df, config, path_col: str = "image_file_path_resolved"):
        df = bbox_metadata_df.copy()
        required = [path_col, "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        df = df.dropna(subset=[c for c in required if c in df.columns])
        if "has_bbox" in df.columns:
            df = df[df["has_bbox"] == True]  # noqa: E712
        self.df = df.reset_index(drop=True)
        self.config = config
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row[self.path_col])

        result = preprocess_image(img, self.config, run_quality_gate=False)
        processed = result["processed"]
        crop_bbox = result["bbox"] or (0, 0, img.shape[1], img.shape[0])
        image_size = tuple(self.config["preprocessing"]["image_size"])

        original_bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
        transformed = transform_bbox_for_preprocessing(original_bbox, crop_bbox, image_size)

        if transformed is None:
            # Lesion fell outside the background-removal crop (rare) —
            # fall back to a full-frame box rather than crashing training.
            transformed = (0, 0, image_size[1], image_size[0])

        x, y, w, h = transformed
        # torchvision detection API expects [x1, y1, x2, y2], and requires
        # x2 > x1, y2 > y1 strictly (guard against degenerate zero-area boxes).
        x2, y2 = max(x + w, x + 1), max(y + h, y + 1)
        boxes = torch.tensor([[x, y, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  # single class: "lesion"

        tensor_img = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
        tensor_img = tensor_img.repeat(3, 1, 1).float()

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return tensor_img, target


def detection_collate_fn(batch):
    """torchvision detection models expect a LIST of images and a LIST of
    target dicts (variable boxes per image), not a stacked batch tensor."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets