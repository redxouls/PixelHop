import jax.numpy as jnp
from einops import rearrange

from .Saab import Saab


class SaabLayer(Saab):
    """
    SaabLayer applies the Saab transformation to the input data.
    """

    def __init__(
        self,
        pool,
        win,
        stride,
        pad,
        threshold=0.001,
        channel_wise=False,
        apply_bias=False,
    ):
        """
        Initialize a SaabLayer.

        Parameters
        ----------
        threshold : float
            Threshold for energy.
        channel_wise : bool
            Whether to apply channel-wise transformation.
        apply_bias : bool
            Whether to apply bias.
        batch_size : int
            Batch size for processing.
        """
        super().__init__(pool, win, stride, pad, threshold, apply_bias)
        self.channel_wise = channel_wise

    def resize_energy(self, energy_previous):
        if self.channel_wise:
            if energy_previous is None:
                energy_previous = jnp.ones(self.C)
            return rearrange(energy_previous, "c -> c 1")
        else:
            return jnp.ones((1, 1))

    def resize_input(self, X):
        if self.channel_wise:
            return rearrange(X, "n h w c -> c n h w 1")
        else:
            return rearrange(X, "n h w c -> 1 n h w c")

    def fit(self, X_batch, energy_previous):
        self.C = X_batch[0].shape[-1]
        X_batch = [self.resize_input(X) for X in X_batch]
        if self.channel_wise:
            X_batch = [[X[c] for X in X_batch] for c in range(self.C)]

        energy_previous = self.resize_energy(energy_previous)
        return super().fit(X_batch, energy_previous)

    def transform(self, X):
        N, H, W, _ = X.shape
        X = self.resize_input(X)
        X = super().transform(X)
        X = rearrange(X, "(n h w) c -> n h w c", n=N, h=H, w=W)
        return X

    def fit_transform(self, X_batch, energy_previous):
        energy_previous = self.fit(X_batch, energy_previous)
        X_batch = [self.transform(X) for X in X_batch]
        return X_batch, energy_previous

    def __str__(self):
        return f"SaabLayer(threshold={self.threshold}, channel_wise={self.channel_wise}, apply_bias={self.apply_bias})"

    def __repr__(self):
        return self.__str__()
