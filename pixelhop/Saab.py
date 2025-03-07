import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


# @jit
def pca(covariance: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform Principal Component Analysis (PCA) on the covariance matrix."""
    eigen_values, eigen_vectors = np.linalg.eigh(np.array(covariance))
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
    dc = jnp.mean(X, axis=(1, 2), keepdims=True)
    X_centered = X - dc
    bias = jnp.max(jnp.linalg.norm(X_centered, axis=(1, 2)))
    mean = jnp.mean(X_centered, axis=0, keepdims=True)
    return dc, bias, mean


@jit
def covariance_batch(X: jnp.ndarray, dc: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """Calculate the covariance matrix for the batch."""
    X_centered = X - dc - mean
    return jnp.einsum("npc,nqd->pcqd", X_centered, X_centered)


def fit(X_batch: jnp.ndarray, energy_previous: jnp.ndarray, threshold: float) -> tuple:
    """Fit the PCA model to the data."""
    """Input: N P C"""
    N, P, C = X_batch[0].shape

    dc_batch = []
    bias = 0
    mean = jnp.zeros((1, P, C))
    for X in X_batch:
        dc_local, bias_local, mean_local = statistics_batch(X)
        dc_batch.append(dc_local)
        bias = jnp.maximum(bias_local, bias)
        mean += mean_local / len(X)

    covariance = jnp.zeros((P * C, P * C))
    for X, dc in zip(X_batch, dc_batch):
        covariance = covariance + covariance_batch(X, dc, mean).reshape(P * C, P * C)
    covariance = covariance / (N * len(X_batch) - 1)

    kernels, eva = pca(covariance)
    dc_kernel = 1 / jnp.sqrt(P * C) * jnp.ones((1, P * C))
    kernels = jnp.concatenate((dc_kernel, kernels[:-1]), axis=0).T

    dc = jnp.concatenate(dc_batch)
    largest_ev = jnp.var(dc * jnp.sqrt(P * C))
    energy = jnp.concatenate((jnp.array([largest_ev]), eva[:-1]), axis=0)
    energy = energy / jnp.sum(energy)
    energy = energy * energy_previous

    cutoff_index = locate_cutoff(energy, threshold)
    return mean, bias, kernels, energy, cutoff_index


@jit
def transform(
    X: jnp.ndarray, mean: jnp.ndarray, kernel: jnp.ndarray, bias: float
) -> jnp.ndarray:
    """Transform the data using the fitted PCA model."""
    X = X - mean
    X = jnp.einsum("nij,ijk->nk", X, kernel.reshape(X.shape[1], X.shape[2], -1))
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

        _, P, C = X_batch[0].shape
        # for c in range(num_channel):
        # X_channel = [X[c] for X in X_batch]
        c = 0
        X_channel = X_batch
        mean, bias, kernels, energy, cutoff_index = fit(
            X_channel, energy_previous[c], threshold
        )
        energy = jax.lax.dynamic_slice(energy, (0,), (cutoff_index,))
        kernels = jax.lax.dynamic_slice(kernels, (0, 0), (P * C, cutoff_index))

        self.mean.append(mean)
        self.bias.append(bias)
        self.kernels.append(kernels)
        self.energy.append(energy)

        self.energy = jnp.concatenate(self.energy)
        self.mean = jnp.array(self.mean)
        self.bias = jnp.array(self.bias)

        if not self.apply_bias:
            self.bias = jnp.zeros_like(self.bias)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Transform the input data using the fitted Saab model."""
        X = np.concatenate(
            [
                transform(X, mean, kernel, bias)
                for mean, kernel, bias in zip(self.mean, self.kernels, self.bias)
            ],
            axis=-1,
        )
        return X
