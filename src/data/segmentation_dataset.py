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
from src.preprocessing.augmentation import augment_pair


class LesionSegmentationDataset(Dataset):
    def __init__(self, bbox_metadata_df, config,
                 path_col: str = "image_file_path_resolved",
                 mask_col: str = "mask_path_used",
                 augment: bool = False,
                 patch_size: int = None):
        """
        patch_size: if set, crop a patch_size x patch_size window centered
        on the lesion (instead of returning the whole preprocessed image).

        Why this exists: a full mammogram is mostly normal, often-dense
        tissue with a lesion occupying a tiny fraction of it. A from-scratch
        U-Net trained on whole images can satisfy the loss reasonably well
        by learning "highlight bright/dense-looking regions" rather than
        the much harder "find the specific lesion" -- which shows up as
        diffuse, spread-out predictions that loosely track tissue density
        rather than tight blobs on the actual lesion. Cropping to a patch
        around the lesion removes most of that easy, wrong shortcut by
        showing the model far less irrelevant tissue per example, the same
        way Phase 9's lesion-crop classifier already does for classification.
        """
        df = bbox_metadata_df.copy()
        if "has_bbox" in df.columns:
            df = df[df["has_bbox"] == True]  # noqa: E712
        df = df.dropna(subset=[path_col, mask_col])
        self.df = df.reset_index(drop=True)
        self.config = config
        self.path_col = path_col
        self.mask_col = mask_col
        self.augment = augment
        self.patch_size = patch_size

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

        # Both are already the same shape at this point (post crop+resize),
        # so paired augmentation can apply directly — flip/rotation stays
        # geometrically synchronized between image and mask.
        if self.augment:
            processed_img, mask_binary = augment_pair(processed_img, mask_binary, seed=None)
            mask_binary = (mask_binary > 0.5).astype(np.float32)  # rotation interpolation can blur; re-binarize

        if self.patch_size is not None:
            processed_img, mask_binary = self._crop_patch(processed_img, mask_binary)

        img_tensor = torch.from_numpy(np.ascontiguousarray(processed_img)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_binary)).unsqueeze(0).float()

        return img_tensor, mask_tensor

    def _crop_patch(self, img, mask):
        """Crop a self.patch_size window centered on the (possibly just-
        augmented) mask's own centroid -- computed fresh each call so it
        stays correct regardless of any flip/rotation already applied."""
        H, W = mask.shape
        ps = min(self.patch_size, H, W)
        ys, xs = np.where(mask > 0.5)
        if len(ys) > 0:
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            # Shouldn't happen (has_bbox rows always carry a real lesion),
            # but fall back to the image center rather than crashing.
            cy, cx = H // 2, W // 2

        if self.augment:
            # Jitter the crop so the lesion isn't dead-center in every
            # training example -- otherwise the model could learn to just
            # stare at the middle pixel instead of actually localizing it.
            jitter = ps // 4
            cy += int(np.random.randint(-jitter, jitter + 1))
            cx += int(np.random.randint(-jitter, jitter + 1))

        half = ps // 2
        y0 = max(0, min(cy - half, H - ps))
        x0 = max(0, min(cx - half, W - ps))
        return img[y0:y0 + ps, x0:x0 + ps], mask[y0:y0 + ps, x0:x0 + ps]