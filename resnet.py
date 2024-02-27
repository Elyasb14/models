# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py

from tinygrad import Tensor
import tinygrad.nn as nn
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch

class BasicBlock:
  expansion: int = 1
  def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1,
                base_width = 64, dilation = 1, norm_layer = None):
    if norm_layer is None: norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64: raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = norm_layer(planes)
    self.relu = Tensor.relu
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x: Tensor) -> Tensor:
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)
    return out

if __name__ == "__main__":
  #basic = BasicBlock()
  weights = torch_load(fetch("https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth")).items()
