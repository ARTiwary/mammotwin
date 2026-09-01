"""
Phase 13: dataset pairing the whole-image preprocessing pipeline (same as
Phase 6) with the fitted TabularPreprocessor's structured feature vector.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_io import load_image
from src.preprocessing.basic_preprocess import preprocess_image
from src.preprocessing.augmentation import augment_image
from src.utils.class_weights import LABEL_MAP


class MultimodalDataset(Dataset):
    def __init__(self, df, config, tabular_preprocessor,
                 path_col: str = "image_file_path_resolved", label_col: str = "pathology_binary",
                 augment: bool = False):
        df = df.dropna(subset=[path_col, label_col]).reset_index(drop=True)
        df = df[df[label_col].isin(LABEL_MAP.keys())].reset_index(drop=True)
        self.df = df
        self.config = config
        self.path_col = path_col
        self.label_col = label_col
        self.augment = augment
        self.tabular_features = tabular_preprocessor.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row[self.path_col])
        if self.augment:
            img = augment_image(img, seed=None)
        result = preprocess_image(img, self.config, run_quality_gate=False)
        processed = result["processed"]

        image_tensor = torch.from_numpy(np.ascontiguousarray(processed)).unsqueeze(0)
        image_tensor = image_tensor.repeat(3, 1, 1).float()

        tabular_tensor = torch.from_numpy(self.tabular_features[idx]).float()
        label = LABEL_MAP[row[self.label_col]]

        return image_tensor, tabular_tensor, label

    def class_counts(self) -> dict:
        return self.df[self.label_col].value_counts().to_dict()