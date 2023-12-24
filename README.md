# Models implemented in tinygrad

learning ml using [tinygrad](https://github.com/tinygrad/tinygrad).

## QUICKSTART

you can technically run these models directly on the cpu, but they will be painfilly slow. the general suggestion is to make sure `clinfo` works, or if on a mac, use the metal backend. clone the repo, update tinygrad to the head of the master branch, and train an mlp on mnist using the gpu (opencl) backend by running the following:

```bash
git clone git@github.com:Elyasb14/models.git
./update_tinygrad.sh
GPU=1 python3 mlp.py train
```

this will save the state of the model in `models/mlp.safetensors`. infer by running `GPU=1 python3 mlp.py infer`.

## MODELS

mlp.py: the most basic of all nns, an mlp. this trains mnist to around 93%, and fashion mnist to about 80%. it consists of 2 linear layers with a leakyrelu in between them.

## REFERENCES

- [tinygrad](https://github.com/tinygrad/tinygrad)
- [makemore](https://github.com/karpathy/makemore)
- [densenet](https://arxiv.org/pdf/1608.06993.pdf)

## LESSONS LEARNED

- when training fashion mnist on cnn.py, simply raising the training steps doesn't really contribute to notably better model accuracy. the loss function may get driven down a lot, but this doesn't necessarily contribute to a better model.
