import jax
import jax.numpy as jnp
from jax import jit, vmap
from einops import rearrange


@jit
def pca(X):
    covariance = X.T @ X
    eigen_values, eigen_vectors = jnp.linalg.eigh(covariance)
    ind = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[ind]
    kernels = eigen_vectors.T[ind]
    return kernels, eigen_values / (X.shape[0] - 1)


@jit
def locate_cutoff(energy, threshold):
    mask = jnp.concat([energy < threshold, jnp.array((True,))], axis=0)
    return jnp.argmax(mask)


@jit
def fit(X, energy_previous, threshold):
    # Remove dc
    dc = jnp.mean(X, axis=1, keepdims=True)
    X = X - dc

    # Calculate bias at the current Hop
    bias = jnp.max(jnp.linalg.norm(X, axis=1))
    mean = jnp.mean(X, axis=0, keepdims=True)
    X = X - mean

    # Perform PCA
    kernels, eva = pca(X)

    # Concatenate with DC kernel
    num_kernels = X.shape[-1]
    dc_kernel = 1 / jnp.sqrt(num_kernels) * jnp.ones((1, num_kernels))
    kernels = jnp.concatenate((dc_kernel, kernels[:-1]), axis=0).T

    # Concatenate with DC energy
    largest_ev = jnp.var(dc * jnp.sqrt(num_kernels))
    energy = jnp.concatenate((jnp.array([largest_ev]), eva[:-1]), axis=0)
    energy = energy / jnp.sum(energy)
    energy = energy * energy_previous

    # Find cutoff index
    cutoff_index = locate_cutoff(energy, threshold)
    return mean, bias, kernels, energy, cutoff_index


@jit
def transform(X, mean, kernel, bias):
    X = X - mean
    X = X @ kernel
    X = X + bias
    return X


class Saab:
    def __init__(self, num_kernels=-1, apply_bias=False):
        self.num_kernels = num_kernels
        self.apply_bias = apply_bias
        self.bias = []  # bias for the current Hop
        self.kernels = []
        self.mean = []  # feature mean of AC
        self.energy = []  # kernel energy list

    def fit(self, X, energy_previous, threshold=0):
        assert len(X.shape) == 3, "Input must be a 3D array!"

        fit_batch = vmap(lambda X, energy_previous: fit(X, energy_previous, threshold))
        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = fit_batch(
            X, energy_previous
        )
        self.energy = jnp.concat(
            [
                jax.lax.dynamic_slice(self.energy[i], (0,), (self.cutoff_index[i],))
                for i in range(len(self.cutoff_index))
            ],
            axis=-1,
        )
        self.kernels = [
            jax.lax.dynamic_slice(
                self.kernels[i],
                (0, 0),
                (self.kernels[i].shape[0], self.cutoff_index[i]),
            )
            for i in range(len(self.cutoff_index))
        ]

        if not self.apply_bias:
            self.bias = jnp.zeros_like(self.bias)

    def transform(self, X):
        X = jnp.concat(
            [
                transform(X_channel, mean, kernel, bias)
                for X_channel, mean, kernel, bias in zip(
                    X, self.mean, self.kernels, self.bias
                )
            ],
            axis=-1,
        )
        return X


if __name__ == "__main__":
    import time
    import numpy as np

    X = np.random.randn(200, 16, 16, 27)
    X = rearrange(X[0:1000], "b h w c -> 1 b (h w c)")

    saab = Saab(num_kernels=-1, apply_bias=True)
    saab.fit(X, threshold=7e-2, bias_previous=0)
    Xt = saab.transform(X)

    saab = Saab(num_kernels=-1, apply_bias=True)
    saab.fit(X, threshold=7e-2, bias_previous=0)
    Xt = saab.transform(X)
    print(Xt)

    X = np.random.randn(200, 16 * 16 * 27)
    start = time.time()
    Xt = saab.transform(X)
    for i in range(1000):
        Xt += saab.transform(X)
    print(Xt)

    print(time.time() - start)
