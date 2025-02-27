# PixelHop
A JAX Implementation of Saab transformation and shrinking operations.

## Installation

First, install JAX following the [official installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
# For Nvidia GPU 
pip install "jax[cuda12]"

# For CPU only
pip install jax
```

Then, install PixelHop package.
```bash
pip install .
```

## Usage 

Please reference following environment variable to allocate resources appropriately.

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]= "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
```

or add environment variable in the command

```python
CUDA_VISIBLE_DEVICES=1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python3 xxx.py
```


### Example

```python
import numpy as np
from pixelhop.PixelHop import PixelHop
from pixelhop.Layers import SaabLayer, ShrinkLayer

pixelhop = PixelHop(
    layers=[
        SaabLayer(threshold=0.007, channel_wise=False, apply_bias=False),
        SaabLayer(threshold=0.001, channel_wise=True, apply_bias=True),
    ],
    shrink_layers=[
        ShrinkLayer(pool=1, win=7, stride=1, pad=3),
        ShrinkLayer(pool=1, win=3, stride=1, pad=1),
    ],
)
print(pixelhop)

sample = np.random.randn(10000, 32, 32, 3)
pixelhop.fit(sample, batch_size=4096)

X = np.random.randn(2000, 32, 32, 3)
Xt = pixelhop.transform(X)
print(Xt.shape)
```