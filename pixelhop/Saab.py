import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


@jit
def pca(covariance: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform Principal Component Analysis (PCA) on the covariance matrix."""
    eigen_values, eigen_vectors = jnp.linalg.eigh(covariance)
    ind = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[ind]
    kernels = eigen_vectors.T[ind]
    return kernels, eigen_values


@jit
def locate_cutoff(energy: jnp.ndarray, threshold: float) -> int:
    """Locate the cutoff index where the energy drops below the threshold."""
    mask = jnp.concatenate([energy < threshold, jnp.array([True])], axis=0)
    return jnp.argmax(mask)


@jit
def statistics_batch(X: jnp.ndarray) -> tuple[jnp.ndarray, float, jnp.ndarray]:
    """Calculate the DC component, bias, and mean of the batch."""
    dc = jnp.mean(X, axis=1, keepdims=True)
    X = X - dc
    bias = jnp.max(jnp.linalg.norm(X, axis=1))
    mean = jnp.mean(X, axis=0, keepdims=True)
    return dc, bias, mean


@jit
def cov_batch(X: jnp.ndarray, dc: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """Calculate the covariance matrix for the batch."""
    return (X - dc - mean).T @ (X - dc - mean)


def fit(X: jnp.ndarray, energy_previous: jnp.ndarray, threshold: float) -> tuple:
    """Fit the PCA model to the data."""
    print("start fitting")

    mean = jnp.zeros((1, X[0][0].shape[1]))
    dc = []
    bias = 0
    for X_batch in X:
        dc_batch, bias_batch, mean_batch = statistics_batch(X_batch[0])
        mean += mean_batch / len(X)
        bias = jnp.maximum(bias_batch, bias)
        dc.append(dc_batch)
    dc = jnp.concatenate(dc, axis=0)

    covariance = jnp.zeros((X[0][0].shape[1], X[0][0].shape[1]))
    for X_batch, dc_batch in zip(X, np.array_split(dc, len(X))):
        covariance = covariance + cov_batch(X_batch[0], dc_batch, mean)

    kernels, eva = pca(covariance)
    eva = eva / (sum([X_batch.shape[0] for X_batch in X]) - 1)
    print(eva.shape)

    num_kernels = X[0][0].shape[-1]
    dc_kernel = 1 / jnp.sqrt(num_kernels) * jnp.ones((1, num_kernels))
    kernels = jnp.concatenate((dc_kernel, kernels[:-1]), axis=0).T
    print(kernels.shape)

    largest_ev = jnp.var(dc * jnp.sqrt(num_kernels))
    energy = jnp.concatenate((jnp.array([largest_ev]), eva[:-1]), axis=0)
    energy = energy / jnp.sum(energy)
    # energy = energy * energy_previous
    print(energy.shape)

    cutoff_index = locate_cutoff(energy, threshold)
    return mean, bias, kernels, energy, cutoff_index


@jit
def transform(
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

    def fit(self, X, energy_previous: jnp.ndarray, threshold: float = 0):
        """Fit the Saab model to the input data."""
        # assert len(X.shape) == 3, "Input must be a 3D array!"

        self.mean = []
        self.bias = []
        self.kernels = []
        self.energy = []

        # for X_channel, energy_previous_channel in zip(X, energy_previous):
        mean, bias, kernels, energy, cutoff_index = fit(X, energy_previous, threshold)
        energy = jax.lax.dynamic_slice(energy, (0,), (cutoff_index,))
        kernels = jax.lax.dynamic_slice(
            kernels, (0, 0), (kernels.shape[0], cutoff_index)
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
                transform(X_channel, mean, kernel, bias)
                for X_channel, mean, kernel, bias in zip(
                    X, self.mean, self.kernels, self.bias
                )
            ],
            axis=-1,
        )
        return X
