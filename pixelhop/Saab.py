import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def _pca(covariance: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform Principal Component Analysis (PCA) on the covariance matrix."""
    eigen_values, eigen_vectors = jnp.linalg.eigh(covariance)
    ind = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[ind]
    kernels = eigen_vectors.T[ind]
    return kernels, eigen_values


@jit
def _locate_cutoff(energy: jnp.ndarray, threshold: float) -> int:
    """Locate the cutoff index where the energy drops below the threshold."""
    mask = jnp.concatenate([energy < threshold, jnp.array([True])], axis=0)
    return jnp.argmax(mask)


@jit
def _statistics_batch(X: jnp.ndarray) -> tuple[jnp.ndarray, float, jnp.ndarray]:
    """Calculate the DC component, bias, and mean of the batch."""
    dc = jnp.mean(X, axis=1, keepdims=True)
    X_centered = X - dc
    bias = jnp.max(jnp.linalg.norm(X_centered, axis=1))
    mean = jnp.mean(X_centered, axis=0, keepdims=True)
    return dc, bias, mean


@jit
def _covariance_batch(
    X: jnp.ndarray, dc: jnp.ndarray, mean: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the covariance matrix for the batch."""
    X_centered = X - dc - mean
    return X_centered.T @ X_centered


def _fit(X_batch: jnp.ndarray, energy_previous: jnp.ndarray, threshold: float) -> tuple:
    """Fit the PCA model to the data."""
    print("start fitting")

    num_kernels = X_batch[0].shape[1]

    dc_batch = []
    bias = 0
    mean = jnp.zeros((1, num_kernels))
    for X in X_batch:
        dc_local, bias_local, mean_local = _statistics_batch(X)
        dc_batch.append(dc_local)
        bias = jnp.maximum(bias_local, bias)
        mean += mean_local / len(X)

    covariance = jnp.zeros((num_kernels, num_kernels))
    for X, dc in zip(X_batch, dc_batch):
        covariance = covariance + _covariance_batch(X, dc, mean)

    kernels, eva = _pca(covariance)
    eva = eva / (sum([X_batch.shape[0] for X_batch in X_batch]) - 1)

    dc_kernel = 1 / jnp.sqrt(num_kernels) * jnp.ones((1, num_kernels))
    kernels = jnp.concatenate((dc_kernel, kernels[:-1]), axis=0).T

    dc = np.concatenate(dc_batch)
    largest_ev = jnp.var(dc * jnp.sqrt(num_kernels))
    energy = jnp.concatenate((jnp.array([largest_ev]), eva[:-1]), axis=0)
    energy = energy / jnp.sum(energy)
    energy = energy * energy_previous

    cutoff_index = _locate_cutoff(energy, threshold)
    return mean, bias, kernels, energy, cutoff_index


@jit
def _transform(
    X: jnp.ndarray, mean: jnp.ndarray, kernel: jnp.ndarray, bias: float
) -> jnp.ndarray:
    """Transform the data using the fitted PCA model."""
    X = X - mean
    X = X @ kernel
    X = X + bias
    return X


class Saab:
    def __init__(self, num_kernels: int = -1, apply_bias: bool = False):
        self.num_kernels = num_kernels
        self.apply_bias = apply_bias
        self.bias = []
        self.kernels = []
        self.mean = []
        self.energy = []

    def fit(self, X_batch, energy_previous: jnp.ndarray, threshold: float = 0):
        """Fit the Saab model to the input data."""
        self.mean = []
        self.bias = []
        self.kernels = []
        self.energy = []

        # for X_channel, energy_previous_channel in zip(X, energy_previous):
        num_channel, _, num_features = X_batch[0].shape
        for c in range(num_channel):
            X_channel = [X[c] for X in X_batch]
            mean, bias, kernels, energy, cutoff_index = _fit(
                X_channel, energy_previous[c], threshold
            )
            energy = jax.lax.dynamic_slice(energy, (0,), (cutoff_index,))
            kernels = jax.lax.dynamic_slice(
                kernels, (0, 0), (num_features, cutoff_index)
            )

            self.mean.append(mean)
            self.bias.append(bias)
            self.kernels.append(kernels)
            self.energy.append(energy)

        self.mean = jnp.array(self.mean)
        self.bias = jnp.array(self.bias)

        if not self.apply_bias:
            self.bias = jnp.zeros_like(self.bias)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Transform the input data using the fitted Saab model."""
        X = jnp.concatenate(
            [
                _transform(X_channel, mean, kernel, bias)
                for X_channel, mean, kernel, bias in zip(
                    X, self.mean, self.kernels, self.bias
                )
            ],
            axis=-1,
        )
        return X
