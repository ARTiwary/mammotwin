"""
Phase 6: PyTorch Dataset for mammogram classification.

Wraps the metadata CSVs (train/val/test_split.csv from Phase 3) and applies
the EXACT SAME preprocessing pipeline as Phase 4 (preprocess_image), so
there is no train/inference mismatch. Grayscale images are repeated across
3 channels since the standard torchvision backbones (ResNet/DenseNet/
EfficientNet) expect 3-channel input.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import preprocess_image
from src.preprocessing.augmentation import augment_image

LABEL_MAP = {"benign": 0, "malignant": 1}
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class MammogramDataset(Dataset):
    def __init__(self, metadata_df, config,
                 path_col: str = "image_file_path_resolved",
                 label_col: str = "pathology_binary",
                 run_quality_gate: bool = False,
                 augment: bool = False):
        """
        run_quality_gate=False by default during training for speed — the
        quality gate should be applied once, upstream, when deciding which
        rows go into the split CSVs at all (Phase 4's quality_report.csv),
        not re-run on every __getitem__ call during training.

        augment: apply random flip/rotation/brightness-contrast — set True
        for the TRAINING dataset only, never for val/test (augmenting
        evaluation data would make metrics non-reproducible and is not
        what augmentation is for).
        """
        self.df = metadata_df.dropna(subset=[path_col, label_col]).reset_index(drop=True)
        self.df = self.df[self.df[label_col].isin(LABEL_MAP.keys())].reset_index(drop=True)
        self.config = config
        self.path_col = path_col
        self.label_col = label_col
        self.run_quality_gate = run_quality_gate
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row[self.path_col])
        if self.augment:
            img = augment_image(img, seed=None)  # None -> fresh randomness each epoch
        result = preprocess_image(img, self.config, run_quality_gate=self.run_quality_gate)
        processed = result["processed"]  # H x W, float32, [0, 1]

        tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)  # 1xHxW
        tensor = tensor.repeat(3, 1, 1).float()  # 3xHxW for pretrained backbones

        label = LABEL_MAP[row[self.label_col]]
        return tensor, label

    def class_counts(self) -> dict:
        return self.df[self.label_col].value_counts().to_dict()