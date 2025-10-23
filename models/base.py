# TODO: implement base abstract class for all models

import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, base_ch):
        super().__init__()

    def forward(self, x):
        pass
