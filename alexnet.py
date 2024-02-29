# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py

from tinygrad import Tensor
import tinygrad.nn as nn
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch

class Alexnet:
  def __init__(self, num_classes: int = 10, dropout: float = 0.5):
    self.features = [
      nn.Conv2d(1, 2, kernel_size=11, stride=2, padding=2),
      Tensor.relu,
      Tensor.max_pool2d
    ]
  def __call__(self, x: Tensor):
    x = x.sequential(self.features)
    return x

if __name__ == "__main__":
  model = Alexnet()
  model(Tensor([[1,2,3], [2,4,5]]))
  # weights = torch_load(fetch("https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth")).items()

