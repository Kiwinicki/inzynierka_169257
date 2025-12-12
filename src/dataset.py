import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Literal

CLASS_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
    "unknown",
]


class FERDataset(Dataset):
    def __init__(
        self,
        filename="../data/fer2013_clean.csv",
        transform=None,
        usage: Literal["Training", "PublicTest", "PrivateTest"] = "Training",
    ):
        self.data = pd.read_csv(filename)
        self.data = self.data[self.data["Usage"] == usage]
        for c in CLASS_LABELS:
            self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
        self.data[CLASS_LABELS] = self.data[CLASS_LABELS].fillna(0)

        denom = self.data[CLASS_LABELS].sum(axis=1).replace(0, np.nan)
        self.data[CLASS_LABELS] = (
            self.data[CLASS_LABELS].div(denom, axis=0).fillna(0.0).astype(np.float32)
        )
        self.data["pixels"] = self.data["pixels"].apply(self._parse_pixels)
        self.transform = transform

    def get_class_weights(self, beta=0.999):
        vote_sums = self.data[CLASS_LABELS].sum(axis=0)
        weights = (1 - beta) / (1 - beta ** vote_sums) # "Effective Number of Samples"
        return torch.tensor(weights.values, dtype=torch.float32)

    @staticmethod
    def _parse_pixels(pixels_str):
        pixels = np.array(pixels_str.split(), dtype=np.uint8)
        return pixels.reshape(48, 48)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.fromarray(row["pixels"], mode="L")
        emotion_dist = torch.from_numpy(row[CLASS_LABELS].to_numpy(np.float32))
        if self.transform:
            image = self.transform(image)
        return image, emotion_dist
