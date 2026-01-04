import torch
from torch import nn
from .base import BaseCNN


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        return identity + x


class ConvNeXt(BaseCNN):
    def __init__(self, base_ch, num_classes, stages=[3, 3, 9, 3]):
        super().__init__()
        dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], 4, 4, 0),
            nn.GroupNorm(1, dims[0]),
        )

        self.stages = nn.ModuleList()
        for i, num_blocks in enumerate(stages):
            blocks = []
            if i > 0:
                blocks.append(
                    nn.Sequential(
                        nn.GroupNorm(1, dims[i - 1]),
                        nn.Conv2d(dims[i - 1], dims[i], 2, 2),
                    )
                )
            blocks.extend([ConvNeXtBlock(dims[i]) for _ in range(num_blocks)])
            self.stages.append(nn.Sequential(*blocks))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)
