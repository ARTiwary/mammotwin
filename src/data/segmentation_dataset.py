"""
Phase 8: segmentation Dataset.

Reuses bbox_metadata_{split}.csv from Phase 7 — it already resolved and
recorded the correct ROI mask path per row (mask_path_used), so no new
ground-truth-building step is needed here.

Critical correctness point (same one that mattered in Phase 7): the mask
must go through the EXACT SAME crop as the image (from Phase 4's
background-removal step), or the two will no longer be spatially aligned.
Unlike Phase 7's bounding box (a few numbers to transform), here the mask
itself is cropped directly using the same crop box, then resized with
NEAREST interpolation (not the image's area interpolation) to keep it a
clean binary mask rather than blurring it into grayscale values.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import preprocess_image


class LesionSegmentationDataset(Dataset):
    def __init__(self, bbox_metadata_df, config,
                 path_col: str = "image_file_path_resolved",
                 mask_col: str = "mask_path_used"):
        df = bbox_metadata_df.copy()
        if "has_bbox" in df.columns:
            df = df[df["has_bbox"] == True]  # noqa: E712
        df = df.dropna(subset=[path_col, mask_col])
        self.df = df.reset_index(drop=True)
        self.config = config
        self.path_col = path_col
        self.mask_col = mask_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row[self.path_col])
        mask = cv2.imread(row[self.mask_col], cv2.IMREAD_GRAYSCALE)

        result = preprocess_image(img, self.config, run_quality_gate=False)
        processed_img = result["processed"]  # H x W, float32, [0,1]
        crop_bbox = result["bbox"] or (0, 0, img.shape[1], img.shape[0])
        image_size = tuple(self.config["preprocessing"]["image_size"])

        x0, y0, w, h = crop_bbox
        mask_cropped = mask[y0:y0 + h, x0:x0 + w]
        mask_resized = cv2.resize(mask_cropped, (image_size[1], image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 127).astype(np.float32)

        img_tensor = torch.from_numpy(np.ascontiguousarray(processed_img)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_binary)).unsqueeze(0).float()

        return img_tensor, mask_tensor