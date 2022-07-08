import torch
import torch.nn as nn
from vision_transformer_pytorch.resnet import resnet50

net=resnet50()
inputs = torch.randn(4, 3, 224,224)
ab=net(inputs)
pass
