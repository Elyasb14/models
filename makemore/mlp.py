# https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig

config = ModelConfig()
metal = torch.device("mps")

class MLP(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.block_size = config.block_size
    self.vocab_size = config.vocab_size
    self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table, what is this?
    self.mlp = nn.Sequential(
      nn.Linear(self.block_size * config.n_embd, config.n_embd2),
      nn.Tanh(),
      nn.Linear(config.n_embd2, self.vocab_size )
    )
  
  def get_block_size(self): return self.block_size

  def forward(self, idx, targets=None):
    # gather the word embeddings of the previous 3 words
    embs = []
    for i in range(self.block_size):
     token_embedding = self.wte(idx) # token embeddings of shape (b, t, n_embd)
     idx = torch.roll(idx, 1, 1)
     idx[:, 0] = self.vocab_size # special <BLANK> token
     embs.append(token_embedding)
    # concat all of the embeddings together and pass through an MLP
    x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
    logits = self.mlp(x)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return logits
  
model = MLP(config).to(metal)

print(model)