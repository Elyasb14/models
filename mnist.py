from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.helpers import dtypes
import numpy as np 
from pathlib import Path
import os, struct
from typing import Union

# import matplotlib.pyplot as plt
# plt.imshow(data[9,:,:], cmap='gray')
# plt.show()
def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

def parse_mnist(file, tensor=False) -> Union[np.ndarray, Tensor]:
    with open(file, "rb") as f:
      _, size = struct.unpack(">II", f.read(8))
      nrows, ncols = struct.unpack(">II", f.read(8))
      data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
      data = data.reshape((size, nrows, ncols))
    if tensor: return Tensor(data)
    else: return data

print(parse_mnist("/Users/ebianchi/code/ml/models/data/t10k-images.idx3-ubyte", tensor=False)[0,:,:])

class TinyNet:
  def __init__(self):
    # l1 outfeatures == ls in features
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

net = TinyNet()
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)
