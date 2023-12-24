# https://arxiv.org/pdf/1608.06993.pdf
# https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
# papers/densenet.pdf

# NOTE: this will be densenet-121

from tinygrad import Tensor 
from tinygrad.nn import BatchNorm2d, Conv2d

class _DenseLayer:
  def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float) -> None:
    pass