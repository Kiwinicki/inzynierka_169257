import torch
from torch import nn
from .base import BaseCNN


class PlainBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PlainNet(BaseCNN):
    def __init__(self, base_ch, num_classes, stages=[2, 2, 2, 2]):
        super().__init__()
        dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        self.stages = nn.ModuleList()
        in_ch = base_ch
        for i, num_blocks in enumerate(stages):
            out_ch = dims[i]
            stride = 2 if i > 0 else 1

            blocks = []
            for j in range(num_blocks):
                blocks.append(
                    PlainBlock(
                        in_ch if j == 0 else out_ch, out_ch, stride if j == 0 else 1
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)
