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

Alternatively, you can specify them in the command line:
```python
CUDA_VISIBLE_DEVICES=1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python3 xxx.py
```

### Example

```python
import time
import jax
import jax.numpy as jnp
import numpy as np
from pixelhop.PixelHop import PixelHop
from pixelhop.Layers import SaabLayer

# Uncomment this line to enable float64 precision globally (optional)
# jax.config.update("jax_enable_x64", True)

# Ensure JAX precompiles key operations
jnp.linalg.eigh(np.random.randn(147, 147))

# Define the PixelHop model with multiple Saab layers
pixelhop = PixelHop(
    layers=[
        SaabLayer(
            pool=1,
            win=7,
            stride=2,
            pad=3,
            threshold=0.007,
            channel_wise=False,
            apply_bias=False,
        ),
        SaabLayer(
            pool=1,
            win=3,
            stride=1,
            pad=1,
            threshold=0.00081,
            channel_wise=True,
            apply_bias=False,
        ),
        SaabLayer(
            pool=1,
            win=3,
            stride=1,
            pad=1,
            threshold=0.00081,
            channel_wise=True,
            apply_bias=False,
        ),
    ]
)

print(pixelhop)

# Generate a training dataset (10000 samples of 128x128x3 images)
sample = np.random.randn(10000, 128, 128, 3)

# Train the PixelHop model
print("Start training...")
start = time.time()
pixelhop.fit(sample, batch_size=100)
print(f"Training time: {time.time() - start:.2f} seconds")

# Generate test dataset (1000 samples of 128x128x3 images)
batch_size = 2000
X = np.random.randn(1000, 128, 128, 3)

# Transform test dataset using PixelHop
print("Start transform...")
start = time.time()
num_batches = (X.shape[0] // batch_size) + 1
Xt = [pixelhop.transform(X_batch) for X_batch in np.array_split(X, num_batches)]
print(f"Transformed shape: {Xt[0].shape}")
print(f"Transformation time: {time.time() - start:.2f} seconds")
```

### Handling GPU Memory
By default, JAX utilizes GPU memory aggressively, which may lead to CUDA Out of Memory (OOM) errors. To prevent this:

Use jax.device_get to move results to the CPU.
Use np.concatenate or np.array to merge arrays efficiently on the CPU.

```python
X = np.random.randn(20000, 128, 128, 3)
num_batches = (X.shape[0] // batch_size) + 1

Xt = [
    jax.device_get(pixelhop.transform(X_batch))  # Move output to CPU
    for X_batch in np.array_split(X, num_batches)
]

Xt = np.concatenate(Xt)  # Merge results on CPU
print(Xt.shape)
```

## Performance Profiling (Optional)
To profile JAX execution and analyze performance:

```python
import jax.profiler

jax.profiler.start_trace("./jax-trace")
# Run your PixelHop model training and inference here
jax.profiler.stop_trace()

```
