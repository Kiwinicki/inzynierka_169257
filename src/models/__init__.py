from .plainnet import PlainNet
from .resnet import ResNet
from .resnext import ResNeXt
from .convnext import ConvNeXt

ARCHITECTURES = {
    "plainnet": PlainNet,
    "resnet": ResNet,
    "resnext": ResNeXt,
    "convnext": ConvNeXt,
}
