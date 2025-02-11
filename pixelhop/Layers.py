from einops import rearrange

from pixelhop.Saab import Saab
from pixelhop.ops import shrink


class SaabLayer(Saab):
    def __init__(
        self, thresholds, channel_wise=False, num_kernels=-1, apply_bias=False
    ):
        super().__init__(num_kernels, apply_bias)
        self.thresholds = thresholds
        self.channel_wise = channel_wise

    def resize_input(self, X):
        if self.channel_wise:
            X = rearrange(X, "n h w p c -> c (n h w) p")
        else:
            X = rearrange(X, "n h w p c -> 1 (n h w) (c p)")
        return X

    def fit(self, X, energy_previous=None, bias_previous=None):
        _, self.H, self.W, _, self.C = X.shape
        X_resized = self.resize_input(X)
        super().fit(
            X_resized,
            energy_previous=energy_previous,
            bias_previous=bias_previous,
            threshold=self.thresholds[0],
        )

    def transform(self, X):
        N = X.shape[0]
        X = self.resize_input(X)
        X = super().transform(X)
        X = rearrange(X, "(n h w) c  ->  n h w c", n=N, h=self.H, w=self.W)
        return X

    def fit_transform(self, X, energy_previous=None, bias_previous=None):
        self.fit(X, energy_previous=energy_previous, bias_previous=bias_previous)
        X = self.transform(X)
        return X, self.energy

    def __str__(self):
        return f"SaabLayer(thresholds={self.thresholds}, channel_wise={self.channel_wise}, num_kernels={self.num_kernels}, apply_bias={self.apply_bias})"


class ShrinkLayer:
    def __init__(self, pool, win, stride, pad):
        self.pool = pool
        self.win = win
        self.stride = stride
        self.pad = pad

    def fit(self, X):
        return shrink(X, self.pool, self.win, self.stride, self.pad)

    def transform(self, X):
        return shrink(X, self.pool, self.win, self.stride, self.pad)

    def __str__(self):
        return f"ShrinkLayer(pool={self.pool}, win={self.win}, stride={self.stride}, pad={self.pad})"


if __name__ == "__main__":
    import time
    import numpy as np

    X = np.random.randn(5000, 16, 16, 9, 3)

    saab_layer = SaabLayer(
        thresholds=[0.002, 0.0001], channel_wise=True, apply_bias=False
    )
    saab_layer.fit(X)
    Xt = saab_layer.transform(X)

    # Start benchmark
    start = time.time()
    saab_layer = SaabLayer(
        thresholds=[0.002, 0.0001], channel_wise=False, apply_bias=False
    )

    saab_layer.fit(X)
    Xt = saab_layer.transform(X)
    print(Xt.shape)

    print(time.time() - start)
