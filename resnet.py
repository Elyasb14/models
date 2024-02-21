from tinygrad import Tensor
import tinygrad.nn as nn

class BasicBlock:
  expansion: int = 1
  def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, groups: int = 1,
                base_width = 64, dilation = 1, norm_layer = None):
    if norm_layer is None: norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64: raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

if __name__ == "__main__":
  basic = BasicBlock()
