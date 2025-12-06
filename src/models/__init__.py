from .plain_cnn import PlainCNN
from .resnet import ResNet

ARCHITECTURES = {"plain": PlainCNN, "resnet": ResNet}
