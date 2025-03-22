import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Callable, Union


def make_transform_fn(layers_subset: List[Callable]) -> Callable:
    """
    Create a JAX-compiled transform function that sequentially applies a list of layers.

    Args:
        layers_subset (List[Callable]): Layers to apply in order.

    Returns:
        Callable: A function that applies all layers to an input tensor.
    """

    @jax.jit
    def transform(X: jnp.ndarray) -> jnp.ndarray:
        for layer in layers_subset:
            X = layer.transform(X)
        return X

    return transform


class PixelHop:
    """
    PixelHop class that applies a sequence of Saab-based transformations to input data.
    """

    def __init__(self, layers: List[Callable]):
        """
        Initialize a PixelHop instance.

        Args:
            layers (List[Callable]): List of fitted SaabLayer instances.
        """
        self.layers = layers

    def fit(self, X: np.ndarray, batch_size: int) -> None:
        """
        Fit the PixelHop model across all layers using the input data.

        Args:
            X (np.ndarray): Input data of shape (N, H, W, C).
            batch_size (int): Batch size for splitting input data.
        """
        num_samples, height, width, _ = X.shape
        num_batches = max(num_samples // batch_size, 1)
        X_batches = np.array_split(X[: num_batches * batch_size], num_batches)

        previous_energy = jnp.ones(1)

        # Initial no-op transformation
        for i, layer in enumerate(self.layers):
            transform_fn = make_transform_fn(self.layers[:i])
            previous_energy, height, width = layer.fit(
                X_batches, previous_energy, height, width, transform_fn
            )

    def transform(self, X: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply the fitted PixelHop transformation to input data.

        Args:
            X (np.ndarray | jnp.ndarray): Input data of shape (N, H, W, C).

        Returns:
            jnp.ndarray: Transformed output.
        """
        for layer in self.layers:
            X = layer.transform(X)
        return X

    def __str__(self) -> str:
        """
        Return a string representation of the PixelHop model.
        """
        desc = f"{self.__class__.__name__}(\n"
        for i, layer in enumerate(self.layers):
            desc += f"  (saab_{i}): {layer}\n"
        return desc + ")"

    def __repr__(self) -> str:
        """
        Return a detailed representation of the PixelHop model.
        """
        return self.__str__()
