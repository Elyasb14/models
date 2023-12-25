# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
'''
takes the previous block_size tokens, encodes them with a lookup table,
concatenates the vectors and predicts the next token with an MLP.
'''

from typing import Any
from config import ModelConfig
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear

CONFIG = ModelConfig()

class MLP:
  def __init__(self):
    self.block_size = CONFIG.block_size
    self.vocab_size = CONFIG.vocab_size
    self.wte = Embedding(self.vocab_size + 1, CONFIG.n_embd)
    self.layers = [
      Linear(self.block_size * CONFIG.n_embd, CONFIG.n_embd2),
      Tensor.tanh,
      Linear(CONFIG.n_embd2, self.vocab_size)
    ]

  def get_block_size(self):
      return self.block_size

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    embs = []
    for i in range(self.block_size)
    