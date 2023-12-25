# Models implemented in tinygrad

learning ml using [tinygrad](https://github.com/tinygrad/tinygrad).

## QUICKSTART

you can technically run these models directly on the cpu, but they will be painfilly slow. the general suggestion is to make sure `clinfo` works, or if on a mac, use the metal backend. clone the repo, update tinygrad to the head of the master branch, and train an mlp on mnist using the gpu (opencl) backend by running the following:

```bash
git clone git@github.com:Elyasb14/models.git
./update_tinygrad.sh
GPU=1 python3 mlp.py --train --dataset mnist
```

this will save the state of the model in `models/mlp.safetensors`. infer by running `GPU=1 python3 mlp.py infer`.

## REFERENCES

- [tinygrad](https://github.com/tinygrad/tinygrad)