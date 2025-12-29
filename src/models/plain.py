import torch
from torch import nn
from .base import BaseCNN


class PlainCNN(BaseCNN):
    def __init__(self, base_ch, num_classes, stages=[1, 1, 1, 1]):
        super().__init__()
        dims = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.stages = nn.ModuleList()
        for i in range(len(stages)):
            in_dim = dims[i]
            out_dim = dims[i+1] if i < len(dims)-1 else dims[i]
            stride = 2 if i < len(dims)-1 else 1
            k = 4 if i < len(dims)-1 else 3
            pad = 1 if i < len(dims)-1 else 0
            
            stage = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_dim if j == 0 else out_dim, out_dim, k, stride if j == 0 else 1, pad if j == 0 else 1),
                    nn.ReLU()
                ) for j in range(stages[i])
            ])
            self.stages.append(stage)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)