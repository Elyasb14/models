# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import torch
import torch.nn as nn
from config import ModelConfig

CONFIG = ModelConfig()

class MLP(nn.Module):
  def __init__(self) -> None:
    super().__init__()