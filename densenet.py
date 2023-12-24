# https://arxiv.org/pdf/1608.06993.pdf
# https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
# papers/densenet.pdf

# NOTE: this will be densenet-121

from tinygrad import Tensor 
from tinygrad.nn import BatchNorm2d, Conv2d
from typing import Any, List, Union
from tinygrad.jit import TinyJit

class _DenseLayer:
  def __init__(
    self,
    num_input_features: int,
    growth_rate: int,
    bn_size: int,
    drop_rate: float,
    memory_efficient: bool = False
    ) -> None:

    self.norm1 = BatchNorm2d(num_input_features)
    self.relu1 = Tensor.relu
    self.conv1 = Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
    self.norm2 = BatchNorm2d(bn_size * growth_rate)
    self.relu2 = Tensor.relu
    self.conv2 = Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    self.drop_rate = float(drop_rate)
    self.memory_efficient = memory_efficient
  
  def bn_function(self, inputs: List[Tensor]) -> Tensor:
    concated_features = Tensor.cat(input, 1) # concatenates the 'input' tensors along axis 1; (A,B,C) cat (A,D,C) == (A,B+D,C)
    bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features))) 
    return bottleneck_output
  
  # TODO: do we need this?
  def any_requires_grad(self, input: List[Tensor]) -> bool:
    for tensor in input:
      if tensor.requires_grad:
        return True
    return False
  
  # TODO: implement this fully
  def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
    print("memory efficiency not implemented yet")

  def __call__(self, input: Tensor):
    prev_features = input
    if self.memory_efficient and self.any_requires_grad(prev_features): 
      self.call_checkpoint_bottleneck()
    else: 
      bottleneck_output = self.bn_function(prev_features)
    new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    if self.drop_rate > 0: 
      # is this right?
      with Tensor.training():
        new_features = new_features.dropout(p=self.drop_rate)
    return new_features

class _DenseBlock:
  def __init__(
    self,
    num_layers: int,
    num_input_features: int,
    bn_size: int,
    growth_rate: int,
    drop_rate: float,
    memory_efficient: bool = False 
  ) -> None:
    for i in range(num_layers):
      layer = _DenseLayer(
        num_input_features + i * growth_rate,
        growth_rate = growth_rate,
        bn_size=bn_size,
        drop_rate=drop_rate,
        memory_efficient=memory_efficient
      )
  def __call__(self, init_features: Tensor) -> Tensor:
    