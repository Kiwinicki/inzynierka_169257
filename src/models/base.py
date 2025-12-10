from __future__ import annotations

from typing import Optional, Tuple, Dict, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from src.dataset import CLASS_LABELS


class BaseCNN(nn.Module):
    """
    Base class for CNN models (Plain CNN, ResNet, ResNeXt, ConvNeXt, ...).
    Handles model-agnostic parts:
      - Profiling: params, GFLOPs, activation memory, model size (MB)
      - Preprocessing pipeline for single-image inference
      - Optional: simple predict()
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (48, 48),
        in_channels: int = 1,
        num_classes: int = 8,
        base_channels: int = 32,
        head_hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            input_size: Tuple[int, int], input image size (H, W).
            in_channels: int, number of input channels (1 for grayscale).
            num_classes: int, number of output classes.
            base_channels: int, base channel multiplier.
        """
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_bytes(self) -> float:
        """Approximate model size in MB (float32)."""
        bytes_size = sum(p.numel() * 4 for p in self.parameters() if p.requires_grad)
        return bytes_size

    def flops(self):
        from torch.utils.flop_counter import FlopCounterMode

        device = next(self.parameters()).device
        with FlopCounterMode() as counter:
            with torch.no_grad():
                x = torch.randn(1, self.in_channels, *self.input_size, device=device)
                self(x)
        return counter.get_total_flops()

    def activation_memory_bytes(self, is_training: bool = False) -> int:
        orig_training = self.training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self = self.to(device)
            x = torch.randn(1, self.in_channels, *self.input_size, device=device)
            self.train(is_training)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad() if not is_training else torch.enable_grad():
                out = self(x)
            if is_training and out.requires_grad:
                (out.sum()).backward()
            peak = torch.cuda.max_memory_allocated(device)
            param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
            self.train(orig_training)
            return max(0, peak - (param_bytes + buffer_bytes))
        return 0

    @staticmethod
    def preprocess_image(
        image,
        input_size: Tuple[int, int] = (48, 48),
        mean: float = 0.5,
        std: float = 0.5,
    ) -> torch.Tensor:
        """
        Convert a numpy array or PIL Image into a model-ready tensor.
        Matches the transform used in app.py and data_loaders.py.

        Args:
            image: input image (np.ndarray or PIL.Image)
            input_size: target (H, W)
            grayscale: convert to grayscale
            mean, std: normalization stats
        Returns:
            x: tensor of shape (1, C, H, W)
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize(input_size, Image.BILINEAR)
        arr = np.asarray(image).astype(np.float32)
        arr = (arr / 255.0 - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return x

    def predict(
        self,
        image,
        return_probs: bool = False,
    ) -> Dict[str, float]:
        """
        Simple single-image inference compatible with app.py.
        """
        device = next(self.parameters()).device
        x = self.preprocess_image(image, input_size=self.input_size).to(device)
        with torch.inference_mode():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (C,)
            return probs

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs):
        from src.models import ARCHITECTURES

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint["state_dict"]
        arch = checkpoint.get("arch", kwargs.get("arch", "plain"))
        base_ch = checkpoint.get("base_ch", kwargs.get("base_ch", 32))
        num_classes = kwargs.get("num_classes", len(CLASS_LABELS))

        model_cls = ARCHITECTURES[arch]
        model = model_cls(base_ch=base_ch, num_classes=num_classes)
        model.load_state_dict(state_dict)

        return model, {"arch": arch, "base_ch": base_ch}
