from helpers import load_mnist, load_fashion
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from random import randint
from tinygrad.helpers import Timing

TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_fashion(tensors=True)

def sparse_categorical_crossentropy(self: Tensor, Y: Tensor, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

model = TinyNet()
opt = SGD(get_parameters(model), lr=3e-4)

def train(steps):
  losses = []
  with Tensor.train():
    for i in range(steps):
      samp = Tensor([randint(0, TEST_IM.shape[0]) for _ in range(64)])
      batch, labels = TRAIN_IM[samp], TRAIN_LAB[samp]
      opt.zero_grad()
      loss = model(batch).sparse_categorical_crossentropy(labels).backward()
      losses.append(loss)
      opt.step()
      pred = model(batch).argmax(axis=-1); acc = (pred == labels).mean() # what is argmax doing
      if i % 100 == 0:
        print(f"Step {i+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy() * 100}")
    state_dict = get_state_dict(model)
    safe_save(state_dict, "new_model.safetensors")

def get_test_acc() -> Tensor: return ((model(TEST_IM).argmax(axis=1) == TEST_LAB).mean()*100).numpy()

def evaluate():
  with Timing("Time: "):
    print(f"Test Accuracy: {get_test_acc()}")

if __name__ == "__main__":
  train(1000)
  evaluate()