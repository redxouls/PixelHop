import jax.numpy as jnp
from einops import rearrange
from typing import Optional, List, Tuple
from .Saab import Saab


class SaabLayer(Saab):
    """
    SaabLayer applies the Saab transformation to input data.

    This layer extends the standard Saab transformation and supports:
    - Channel-wise transformations.
    - Optional bias application.
    """

    def __init__(
        self,
        pool: int,
        win: int,
        stride: int,
        pad: int,
        threshold: float = 0.001,
        channel_wise: bool = False,
        apply_bias: bool = False,
    ):
        """
        Initializes a SaabLayer.

        Parameters
        ----------
        pool : int
            Pooling size.
        win : int
            Window size for feature extraction.
        stride : int
            Stride for moving window.
        pad : int
            Padding size for input.
        threshold : float, optional (default=0.001)
            Threshold for energy pruning.
        channel_wise : bool, optional (default=False)
            If True, applies transformation independently per channel.
        apply_bias : bool, optional (default=False)
            If True, applies bias during transformation.
        """
        super().__init__(pool, win, stride, pad, threshold, apply_bias)
        self.channel_wise = channel_wise

    def resize_energy(
        self, energy_previous: Optional[jnp.ndarray], num_channels: int
    ) -> jnp.ndarray:
        """
        Resizes the energy array to match the channel-wise setting.

        Parameters
        ----------
        energy_previous : jnp.ndarray or None
            Energy values from the previous layer.
        num_channels : int
            Number of channels in the input.

        Returns
        -------
        jnp.ndarray
            Reshaped energy array.
        """
        if self.channel_wise:
            if energy_previous is None:
                energy_previous = jnp.ones(num_channels)
            return rearrange(energy_previous, "c -> c 1")
        else:
            return jnp.ones((1, 1))

    def resize_input(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Reshapes a single input tensor for Saab transformation.

        Parameters
        ----------
        X : jnp.ndarray
            Input tensor of shape (N, H, W, C).

        Returns
        -------
        jnp.ndarray
            Reshaped input tensor.
        """
        if self.channel_wise:
            return rearrange(X, "n h w c -> c n h w 1")
        else:
            return rearrange(X, "n h w c -> 1 n h w c")

    def fit(
        self, X_batch: List[jnp.ndarray], energy_previous: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Fits the SaabLayer to the input batch.

        Parameters
        ----------
        X_batch : List[jnp.ndarray]
            List of input batches.
        energy_previous : jnp.ndarray or None
            Energy values from the previous layer.

        Returns
        -------
        jnp.ndarray
            Updated energy values.
        """
        num_channels = X_batch[0].shape[-1]

        X_batch = [self.resize_input(X) for X in X_batch]
        if self.channel_wise:
            X_batch = list(map(list, zip(*X_batch)))

        energy_previous = self.resize_energy(energy_previous, num_channels)

        return super().fit(X_batch, energy_previous)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the Saab transformation to the input.

        Parameters
        ----------
        X : jnp.ndarray
            Input tensor of shape (N, H, W, C).

        Returns
        -------
        jnp.ndarray
            Transformed output tensor.
        """
        N, H, W, _ = X.shape
        X = self.resize_input(X)
        X = super().transform(X)
        return rearrange(X, "(n h w) c -> n h w c", n=N, h=H, w=W)

    def fit_transform(
        self, X_batch: List[jnp.ndarray], energy_previous: Optional[jnp.ndarray]
    ) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
        """
        Fits the SaabLayer and transforms the input batch.

        Parameters
        ----------
        X_batch : List[jnp.ndarray]
            List of input batches.
        energy_previous : jnp.ndarray or None
            Energy values from the previous layer.

        Returns
        -------
        Tuple[List[jnp.ndarray], jnp.ndarray]
            Transformed input batch and updated energy values.
        """
        energy_previous = self.fit(X_batch, energy_previous)
        X_batch = [self.transform(X) for X in X_batch]
        return X_batch, energy_previous

    def __str__(self) -> str:
        return (
            f"SaabLayer("
            f"pool={self.pool}, "
            f"win={self.win}, "
            f"stride={self.stride}, "
            f"pad={self.pad}, "
            f"threshold={self.threshold}, "
            f"channel_wise={self.channel_wise}, "
            f"apply_bias={self.apply_bias}"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()
