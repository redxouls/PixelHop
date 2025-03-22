import jax.numpy as jnp
from typing import Optional, List, Callable, Tuple
from .Saab import Saab


class SaabLayer(Saab):
    """
    Wrapper around the Saab feature extractor for modular layer-based usage.
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
        Initialize a SaabLayer with pooling and patch parameters.

        Args:
            pool (int): Pooling size.
            win (int): Patch window size.
            stride (int): Stride for patch extraction.
            pad (int): Padding for input feature maps.
            threshold (float): Energy threshold for kernel selection.
            channel_wise (bool): Whether to process each channel independently.
            apply_bias (bool): Whether to apply bias in transform.
        """
        super().__init__(pool, win, stride, pad, threshold, channel_wise, apply_bias)

    def fit(
        self,
        input_batch: List[jnp.ndarray],
        previous_energy: Optional[jnp.ndarray],
        input_height: int,
        input_width: int,
        previous_transform: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> Tuple[jnp.ndarray, int, int]:
        """
        Fit the SaabLayer to input data.

        Args:
            input_batch (List[jnp.ndarray]): List of input feature maps.
            previous_energy (Optional[jnp.ndarray]): Energy vector from previous layer.
            input_height (int): Height of input feature maps.
            input_width (int): Width of input feature maps.
            previous_transform (Callable): Transformation function for previous layer.

        Returns:
            Tuple[jnp.ndarray, int, int]: Energy vector and output feature map shape (H, W).
        """
        return super().fit(
            input_batch, previous_energy, input_height, input_width, previous_transform
        )

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the Saab transform to input data.

        Args:
            X (jnp.ndarray): Input data tensor.

        Returns:
            jnp.ndarray: Transformed output.
        """
        return super().transform(X)

    def __str__(self) -> str:
        return (
            f"SaabLayer("
            f"pool={self.pool}, "
            f"win={self.win}, "
            f"stride={self.stride}, "
            f"pad={self.pad}, "
            f"threshold={self.threshold}, "
            f"channel_wise={self.channel_wise}, "
            f"apply_bias={self.apply_bias})"
        )

    def __repr__(self) -> str:
        return self.__str__()
