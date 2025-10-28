import torch
import torch.nn as nn
import torch.optim as optim
import trackio as wandb
from models import PlainCNN, ResNet
from .data_loaders import get_data_loaders

import torch
from models.plain_cnn import PlainCNN

model = PlainCNN(base_ch=32, n_classes=8)
x = torch.randn(1, 1, 48, 48)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)

print(
    f"Model size: {sum([p.numel() for p in model.parameters()]) * 2 / 1024 / 1024:.2f} MB"
)
print(f"Activation memory: {model.activation_memory_bytes(x) / 1024 / 1024:.2f} MB")
