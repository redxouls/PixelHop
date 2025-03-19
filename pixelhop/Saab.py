import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial


@jax.jit
def _pca(covariance: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Perform Principal Component Analysis (PCA) on the covariance matrix."""
    eigen_values, eigen_vectors = jnp.linalg.eigh(covariance)
    ind = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[ind]
    kernels = eigen_vectors.T[ind]
    return kernels, eigen_values


@jax.jit
def _locate_cutoff(energy: jnp.ndarray, threshold: float) -> int:
    """Locate the cutoff index where the energy drops below the threshold."""
    mask = jnp.concatenate([energy < threshold, jnp.array([True])], axis=0)
    return jnp.argmax(mask)


@partial(jax.jit, static_argnames=["pool", "pad", "win", "stride"])
def _extract_patches(X, pool, pad, win, stride):
    X = jax.lax.reduce_window(
        X,
        -jnp.inf,
        jax.lax.max,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        padding="VALID",
    )

    # ---- Apply Padding ----
    X = jnp.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")

    B, H, W, C = X.shape
    out_h = (H - win) // stride + 1
    out_w = (W - win) // stride + 1

    def get_patch(i, j, X):
        return jax.lax.dynamic_slice(
            X, (0, i * stride, j * stride, 0), (B, win, win, C)
        )

    # Vectorized patch extraction
    i_vals, j_vals = jnp.arange(out_h), jnp.arange(out_w)
    patches = jax.vmap(lambda i: jax.vmap(lambda j: get_patch(i, j, X))(j_vals))(i_vals)
    return patches


@jax.jit
def _statistics(patches):
    dc_local = jnp.mean(patches, axis=-1, keepdims=True)  # Mean per patch
    X_centered = patches - dc_local

    bias_local = jnp.max(jnp.linalg.norm(X_centered, axis=-1))
    mean_local = jnp.mean(X_centered, axis=0, keepdims=True)
    return dc_local, bias_local, mean_local


def _compute_statistics(X_batch, extract_patches, num_channel, num_kernel):
    dc_batch = []
    mean = jnp.zeros((num_channel, 1, num_kernel))
    bias = jnp.zeros((num_channel,))
    for X_cur in X_batch:
        patches = extract_patches(X_cur)
        dc_local, bias_local, mean_local = jax.vmap(_statistics)(patches)

        mean = mean + mean_local
        bias = jnp.maximum(bias, bias_local)
        dc_batch.append(dc_local)

    mean /= len(X_batch)
    return mean, bias, dc_batch


@jax.jit
def _covariance(patches, dc, mean):
    X_centered = patches - dc - mean
    return jnp.einsum(
        "...i,...j->ij", X_centered, X_centered, precision=jax.lax.Precision.HIGHEST
    )


def _compute_covariance(
    X_batch, dc_batch, mean, extract_patches, num_channel, num_kernel
):
    covariance = jnp.zeros((num_channel, num_kernel, num_kernel))
    for X_cur, dc in zip(X_batch, dc_batch):
        patches = extract_patches(X_cur)
        covariance += jax.vmap(_covariance)(patches, dc, mean)
    return covariance


@jax.jit
def _compute_kernel(covariance, dc_batch, energy_previous, threshold):
    # Apply PCA
    num_kernel = covariance.shape[0]
    kernels, eva = _pca(covariance)
    dc_kernel = 1 / jnp.sqrt(num_kernel) * jnp.ones((1, num_kernel))
    kernels = jnp.concatenate((dc_kernel, kernels[:-1]), axis=0).T

    # Apply PCA
    dc = jnp.concatenate(dc_batch)
    largest_ev = jnp.var(dc * jnp.sqrt(num_kernel))
    energy = jnp.concatenate((jnp.array([largest_ev]), eva[:-1]), axis=0)
    energy = energy / jnp.sum(energy)
    energy = energy * energy_previous

    cutoff_index = _locate_cutoff(energy, threshold)
    return kernels, energy, cutoff_index


# @partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"])
def _fit(X_batch, energy_previous, extract_patches, threshold):
    batch_size, H, W, C = X_batch[0].shape
    num_batches = len(X_batch)
    num_channel, _, num_kernel = extract_patches(X_batch[0]).shape

    # ---- Compute Mean and Bias ----
    mean, bias, dc_batch = _compute_statistics(
        X_batch, extract_patches, num_channel, num_kernel
    )

    # ---- Compute Covariance ----
    covariance = _compute_covariance(
        X_batch, dc_batch, mean, extract_patches, num_channel, num_kernel
    )
    covariance /= batch_size * num_batches * H * W - 1

    kernels, energy, cutoff_index = jax.vmap(
        lambda covariance, dc_batch, energy_previous: _compute_kernel(
            covariance, dc_batch, energy_previous, threshold
        )
    )(covariance, dc_batch, energy_previous)

    return mean, bias, kernels, energy, cutoff_index


# @jax.jit
@partial(jax.jit)
def transform(
    X: jnp.ndarray, mean: jnp.ndarray, kernel: jnp.ndarray, bias: float
) -> jnp.ndarray:
    X = X - mean
    X = X @ kernel
    X = X + bias
    return X


class Saab:
    def __init__(
        self,
        pool,
        win,
        stride,
        pad,
        threshold,
        channel_wise,
        apply_bias: bool = False,
    ):
        self.pool, self.win, self.stride, self.pad = pool, win, stride, pad
        self.threshold = threshold
        self.channel_wise = channel_wise
        self.apply_bias = apply_bias
        self.bias = []
        self.kernels = []
        self.mean = []
        self.energy = []

    def fit(self, X_batch, energy_previous: jnp.ndarray, transform_previous):
        """Fit the Saab model to the input data."""

        def extract_patches(X):
            patches = _extract_patches(
                transform_previous(X), self.pool, self.pad, self.win, self.stride
            )
            if self.channel_wise:
                return rearrange(patches, "h w b p q c -> c (b h w) (p q)")
            else:
                return rearrange(patches, "h w b p q c -> 1 (b h w) (p q c)")

        if not self.channel_wise:
            energy_previous = jnp.ones(1)

        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = _fit(
            X_batch, energy_previous, extract_patches, self.threshold
        )

        # Remove kernel and engery below thresholds
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

        return self.energy

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """Transform the input data using the fitted Saab model."""
        patches = _extract_patches(X, self.pool, self.pad, self.win, self.stride)
        H, W, N = patches.shape[:3]
        if self.channel_wise:
            patches = rearrange(patches, "h w b p q c -> c (b h w) (p q)")
        else:
            patches = rearrange(patches, "h w b p q c -> 1 (b h w) (p q c)")

        X = jnp.concat(
            [
                transform(X_channel, mean, kernel, bias)
                for X_channel, mean, kernel, bias in zip(
                    patches, self.mean, self.kernels, self.bias
                )
            ],
            axis=-1,
        )
        return rearrange(X, "(n h w) c -> n h w c", n=N, h=H, w=W)
