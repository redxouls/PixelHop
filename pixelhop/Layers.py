import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from pixelhop.Saab import Saab
from pixelhop.ops import shrink


class SaabLayer(Saab):
    def __init__(
        self,
        threshold=0.001,
        channel_wise=False,
        num_kernels=-1,
        apply_bias=False,
        batch_size=4096,
    ):
        super().__init__(num_kernels, apply_bias)
        self.threshold = threshold
        self.channel_wise = channel_wise
        self.batch_size = batch_size

    def resize_energy(self, energy_previous):
        if energy_previous is None:
            energy_previous = jnp.ones(self.C)
        if self.channel_wise:
            return repeat(energy_previous, "c -> c p", p=self.P)
        else:
            return repeat(energy_previous, "c -> 1 (c p)", p=self.P)

    def resize_input(self, X):
        if self.channel_wise:
            return rearrange(X, "n h w p c -> c (n h w) p")
        else:
            return rearrange(X, "n h w p c -> 1 (n h w) (c p)")

    def fit(self, X_batch, energy_previous=None):
        _, self.H, self.W, self.P, self.C = X_batch[0].shape
        X_batch = [self.resize_input(X) for X in X_batch]
        energy_previous = self.resize_energy(energy_previous)
        super().fit(X_batch, energy_previous, self.threshold)

    def transform(self, X):
        N = X.shape[0]
        X = self.resize_input(X)
        X = super().transform(X)
        X = rearrange(X, "(n h w) c  ->  n h w c", n=N, h=self.H, w=self.W)
        return X

    def fit_transform(self, X_batch, energy_previous=None):
        self.fit(X_batch, energy_previous=energy_previous)
        X_batch = [self.transform(X) for X in X_batch]
        return X_batch, self.energy


class ShrinkLayer:
    def __init__(self, pool, win, stride, pad, batch_size=2**13):
        self.pool = pool
        self.win = win
        self.stride = stride
        self.pad = pad
        self.batch_size = batch_size

    def transform(self, X):
        return shrink(X, self.pool, self.win, self.stride, self.pad)

    def transform_batch(self, X_batch):
        return [jax.device_get(self.transform(X)) for X in X_batch]


if __name__ == "__main__":
    import time
    import numpy as np

    X = np.random.randn(5000, 16, 16, 9, 3)

    saab_layer = SaabLayer(threshold=0.0001, channel_wise=True, apply_bias=False)
    saab_layer.fit(X)
    Xt = saab_layer.transform(X)

    # Start benchmark
    start = time.time()
    saab_layer = SaabLayer(thresholds=0.0001, channel_wise=False, apply_bias=False)

    saab_layer.fit(X)
    Xt = saab_layer.transform(X)
    print(Xt.shape)

    print(time.time() - start)
