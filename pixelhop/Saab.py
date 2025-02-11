import jax
import jax.numpy as jnp
from jax import jit, vmap
from einops import rearrange


@jit
def pca(X):
    cov = X.T @ X
    eva, eve = jnp.linalg.eigh(cov)
    inds = eva.argsort()[::-1]
    eva = eva[inds]
    kernels = eve.T[inds]
    return kernels, eva / (X.shape[0] - 1)


@jit
def fit(X, bias_pre, energy_pre, threshold):
    X = X + bias_pre

    # Remove dc
    dc = jnp.mean(X, axis=1, keepdims=True)
    X = X - dc

    # Calcualte bias at the current Hop
    bias_current = jnp.max(jnp.linalg.norm(X, axis=1))
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
    energy = energy * energy_pre

    # Find cutoff index
    mask = jnp.concat([energy < threshold, jnp.array((True,))], axis=0)
    cutoff_index = jnp.argmax(mask)
    return mean, bias_current, kernels, energy, cutoff_index


@jit
def transform(X, bias_pre, mean, kernel):
    X = X + bias_pre
    X = X - mean
    X = X @ kernel
    return X


class Saab:
    def __init__(self, num_kernels=-1, apply_bias=False):
        self.num_kernels = num_kernels
        self.apply_bias = apply_bias
        self.bias_pre = 0
        self.bias_current = []  # bias for the current Hop
        self.kernels = []
        self.mean = []  # feature mean of AC
        self.energy = []  # kernel energy list

    def fit(self, X, energy_pre=None, threshold=0, bias_pre=None):
        assert len(X.shape) == 3, "Input must be a 3D array!"
        X = X.astype(jnp.float32)

        if energy_pre is None:
            energy_pre = jnp.ones(X.shape[-1])

        if not self.apply_bias:
            self.bias_pre = 0
        else:
            self.bias_pre = bias_pre

        fit_batch = vmap(lambda X: fit(X, self.bias_pre, energy_pre, threshold))
        self.mean, self.bias_current, self.kernels, self.energy, self.cutoff_index = (
            fit_batch(X)
        )

    def transform(self, X):
        X = X.astype(jnp.float32)
        transform_batch = vmap(
            lambda X, mean, kernels: transform(X, self.bias_pre, mean, kernels)
        )
        X = transform_batch(X, self.mean, self.kernels)
        X = jnp.concat(
            [
                jax.lax.dynamic_slice(
                    X[i], (0, 0), (X[i].shape[0], self.cutoff_index[i])
                )
                for i in range(len(self.cutoff_index))
            ],
            axis=-1,
        )
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    import time
    import numpy as np

    X = np.random.randn(200, 16, 16, 27)
    X = rearrange(X[0:1000], "b h w c -> 1 b (h w c)")

    saab = Saab(num_kernels=-1, apply_bias=True)
    saab.fit(X, bias_pre=0)
    Xt = saab.transform(X)

    saab = Saab(num_kernels=-1, apply_bias=True)
    saab.fit(X, threshold=7e-3, bias_pre=0)
    Xt = saab.transform(X)
    print(Xt)

    X = np.random.randn(200, 16 * 16 * 27)
    start = time.time()
    Xt = saab.transform(X)
    for i in range(1000):
        Xt += saab.transform(X)
    print(Xt)

    print(time.time() - start)
