# PixelHop
A JAX Implementation of Saab transformation and shrinking operations.

## Installation

```bash
pip install .
```

## Usage

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