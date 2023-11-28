import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.helpers import dtypes

class TinyNet:
  def __init__(self):
    # l1 outfeatures == ls in features
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)
  
  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

net = TinyNet()

print(net(Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32)))