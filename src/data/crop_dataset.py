"""
Phase 9: lesion-crop classification dataset.

Crops the ORIGINAL full mammogram directly to the lesion region (using
Phase 7's bounding-box ground truth), with a small contextual margin, then
normalizes and resizes — deliberately SKIPPING Phase 4's background-removal
step. Background removal (Otsu + connected components, tuned to separate
breast tissue from a black background) is meaningless on an already-tiny,
already-homogeneous lesion crop and would behave unpredictably; the crop
itself already IS the region of interest.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import normalize_image, resize_image
from src.preprocessing.augmentation import augment_image
from src.utils.class_weights import LABEL_MAP


class LesionCropDataset(Dataset):
    def __init__(self, bbox_metadata_df, config,
                 path_col: str = "image_file_path_resolved",
                 label_col: str = "pathology_binary",
                 crop_padding_fraction: float = 0.2,
                 augment: bool = False):
        """
        crop_padding_fraction: extra context kept around the tight lesion
        bbox on each side (as a fraction of box width/height) — a small
        margin around the lesion is standard practice, since a zero-margin
        crop can cut off diagnostically relevant boundary/margin texture.

        augment: apply random flip/rotation/brightness-contrast to the
        CROPPED region — set True for training only, never val/test.
        """
        df = bbox_metadata_df.copy()
        if "has_bbox" in df.columns:
            df = df[df["has_bbox"] == True]  # noqa: E712
        required = [path_col, label_col, "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        df = df.dropna(subset=[c for c in required if c in df.columns])
        df = df[df[label_col].isin(LABEL_MAP.keys())]
        self.df = df.reset_index(drop=True)
        self.config = config
        self.path_col = path_col
        self.label_col = label_col
        self.crop_padding_fraction = crop_padding_fraction
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row[self.path_col])

        x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]
        pad_w, pad_h = w * self.crop_padding_fraction, h * self.crop_padding_fraction
        x0 = max(0, int(x - pad_w))
        y0 = max(0, int(y - pad_h))
        x1 = min(img.shape[1], int(x + w + pad_w))
        y1 = min(img.shape[0], int(y + h + pad_h))

        crop = img[y0:y1, x0:x1]
        if crop.size == 0:  # degenerate box, extremely rare edge case
            crop = img

        if self.augment:
            crop = augment_image(crop, seed=None)

        clip_percentile = tuple(self.config["preprocessing"]["clip_percentile"])
        image_size = tuple(self.config["preprocessing"]["image_size"])
        normalized = normalize_image(crop, clip_percentile)
        resized = resize_image(normalized, image_size)

        tensor = torch.from_numpy(np.ascontiguousarray(resized)).unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1).float()

        label = LABEL_MAP[row[self.label_col]]
        return tensor, label

    def class_counts(self) -> dict:
        return self.df[self.label_col].value_counts().to_dict()