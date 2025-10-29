import torch
from torch import nn
from .base import BaseCNN


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=s, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if s != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(BaseCNN):
    def __init__(self, base_ch, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            ResNetBlock(base_ch, base_ch * 2, s=2),
            ResNetBlock(base_ch * 2, base_ch * 4, s=2),
            ResNetBlock(base_ch * 4, base_ch * 8, s=2),
            ResNetBlock(base_ch * 8, base_ch * 8, s=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_ch * 8, n_classes),
        )

    def forward(self, x):
        return self.layers(x)
