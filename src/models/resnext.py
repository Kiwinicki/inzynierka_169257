import torch
from torch import nn
from .base import BaseCNN


class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=32):
        super().__init__()
        hidden_dim = out_ch // 2

        self.conv1 = nn.Conv2d(in_ch, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNeXt(BaseCNN):
    def __init__(self, base_ch, num_classes, groups=32, stages=[2, 2, 2, 2]):
        super().__init__()
        dims = [base_ch * 4, base_ch * 8, base_ch * 16, base_ch * 32] # hardcode dims widening

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        self.stages = nn.ModuleList()
        in_ch = base_ch
        for i, num_blocks in enumerate(stages):
            out_ch = dims[i]
            stride = 1 if i == 0 else 2

            blocks = []
            for j in range(num_blocks):
                blocks.append(
                    ResNeXtBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        stride if j == 0 else 1,
                        groups=groups,
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
