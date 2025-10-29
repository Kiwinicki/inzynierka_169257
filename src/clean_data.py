from .dataset import FERDataset
import torch
import torchvision

ds = FERDataset("./data")

img, label = ds[0]
print(torchvision.transforms.functional.pil_to_tensor(img).shape, label)
