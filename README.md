# Models implemented in tinygrad

learning ml using [tinygrad](https://github.com/tinygrad/tinygrad) 

## TINYGRAD QUICKSTART

make sure you have tinygrad installed, and run the following to train mnist to around 98% using a cnn and the GPU [(opencl)](https://www.khronos.org/opencl/) backend.

```bash
GPU=1 python3 cnn.py --train --dataset mnist
```

this will save the state of the model in `models/cnn.safetensors`. infer by running `GPU=1 python3 cnn.py --infer`.

## REFERENCES

- [tinygrad](https://github.com/tinygrad/tinygrad)
- [makemore](https://github.com/karpathy/makemore/tree/master)

## GOALS

- implement all of the makemore models
- implement a cnn that trains fashion mnist to over 98%