from .plain import PlainCNN
from .resnet import ResNet
from .resnext import ResNeXt
from .convnext import ConvNeXt

ARCHITECTURES = {
    "plain": PlainCNN,
    "resnet": ResNet,
    "resnext": ResNeXt,
    "convnext": ConvNeXt,
}
