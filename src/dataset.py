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
]


class FERDataset(Dataset):
    def __init__(
        self,
        data_dir="../data",
        transform=None,
        usage: Literal["Training", "PublicTest", "PrivateTest"] = "Training",
    ):
        data_dir = Path(data_dir)
        fer2013 = pd.read_csv(data_dir / "fer2013.csv")
        fer_plus = pd.read_csv(data_dir / "fer2013new.csv")
        fer_plus = fer_plus.drop(columns=["Image name"])

        self.emotion_cols = CLASS_LABELS
        all_cols = self.emotion_cols + ["unknown", "NF"]

        self.data = pd.concat(
            [fer2013, fer_plus[all_cols]],
            axis=1,
        )
        self.data = self.data[self.data["Usage"] == usage]

        for c in all_cols:
            self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
        self.data[all_cols] = self.data[all_cols].fillna(0)

        denom = self.data[self.emotion_cols].sum(axis=1).replace(0, np.nan)
        self.data[self.emotion_cols] = (
            self.data[self.emotion_cols]
            .div(denom, axis=0)
            .fillna(0.0)
            .astype(np.float32)
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48, 1)
        image = Image.fromarray(pixels.squeeze(), mode="L")
        emotion_dist = torch.from_numpy(row[self.emotion_cols].to_numpy(np.float32))
        if self.transform:
            image = self.transform(image)
        return image, emotion_dist
