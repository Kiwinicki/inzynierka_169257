import torch
from torch import nn
from .base import BaseCNN


class PlainCNN(BaseCNN):
    def __init__(self, base_ch, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=4, stride=2, padding=1),  # 48->24
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),  # 24->12
            nn.ReLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),  # 12->6
            nn.ReLU(),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),  # 6->3
            nn.ReLU(),
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, 1, 0),  # 3->1
        )
        self.head = nn.Linear(base_ch * 8, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.head(x)
