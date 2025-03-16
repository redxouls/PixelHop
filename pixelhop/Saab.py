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


@partial(jax.jit, static_argnames=["win", "stride"])
def _extract_patches(X, win, stride):
    """
    Extracts sliding patches dynamically from input X using `jax.lax.dynamic_slice`.

    Args:
        X: Input tensor (B, H, W, C)
        win: Patch size.
        stride: Stride for sliding window.

    Returns:
        Patches of shape (B * out_H * out_W, win * win * C)
    """
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
    return rearrange(patches, "h w b p q c -> (b h w) (p q c)")


@partial(jax.jit, static_argnames=["extract_patches", "num_kernel"])
def _compute_statistics(X_batch, extract_patches, num_kernel):
    """
    Computes local mean (dc_local), bias, and mean update for each batch.

    Args:
        X_batch: Batched input tensor.
        win: Patch size.
        stride: Stride for patches.
        num_batches: Number of mini-batches.
        num_kernel: Number of elements in a single patch.

    Returns:
        mean, bias, dc_batch
    """

    def scan_statistics(carry, X_cur):
        mean, bias = carry  # Unpack carry state
        patches = extract_patches(X_cur)
        dc_local = jnp.mean(patches, axis=-1, keepdims=True)  # Mean per patch
        X_centered = patches - dc_local

        bias_local = jnp.max(jnp.linalg.norm(X_centered, axis=-1))
        mean_local = jnp.mean(X_centered, axis=0, keepdims=True)

        new_mean = mean + mean_local
        new_bias = jnp.maximum(bias, bias_local)

        return (new_mean, new_bias), dc_local

    mean_init = jnp.zeros((1, num_kernel))
    bias_init = 0.0

    (final_mean, final_bias), dc_batch = jax.lax.scan(
        scan_statistics, (mean_init, bias_init), X_batch
    )
    final_mean /= len(X_batch)

    return final_mean, final_bias, dc_batch


@partial(jax.jit, static_argnames=["num_kernel", "extract_patches"])
def _compute_covariance(X_batch, dc_batch, mean, extract_patches, num_kernel):
    def scan_covariance(carry, inputs):
        X_cur, dc = inputs
        patches = extract_patches(X_cur)
        X_centered = patches - dc - mean

        carry += jnp.einsum(
            "...i,...j->ij", X_centered, X_centered, precision=jax.lax.Precision.HIGHEST
        )
        return carry, None

    covariance_init = jnp.zeros((num_kernel, num_kernel))

    covariance, _ = jax.lax.scan(scan_covariance, covariance_init, (X_batch, dc_batch))

    return covariance


@partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"])
def _fit(X_batch, energy_previous, pool, win, stride, pad, threshold):
    """
    Compute the mean, bias, and covariance of sliding patches.

    Args:
        X: Input tensor (B, H, W, C)
        pool: Pooling size.
        win: Patch size.
        stride: Stride for patches.
        pad: Padding size.

    Returns:
        covariance matrix, mean
    """
    batch_size, H, W, C = X_batch[0].shape
    num_batches = len(X_batch)
    num_kernel = win * win * C

    X_batch = jnp.stack(X_batch)

    # ---- Apply Max Pooling ----
    X_batch = jax.lax.reduce_window(
        X_batch,
        -jnp.inf,
        jax.lax.max,
        (1, 1, pool, pool, 1),
        (1, 1, pool, pool, 1),
        padding="VALID",
    )

    # ---- Apply Padding ----
    X_batch = jnp.pad(
        X_batch, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
    )

    # ---- Compute Mean and Bias ----
    mean, bias, dc_batch = _compute_statistics(
        X_batch, lambda X: _extract_patches(X, win, stride), num_kernel
    )

    # ---- Compute Covariance ----
    covariance = _compute_covariance(
        X_batch, dc_batch, mean, lambda X: _extract_patches(X, win, stride), num_kernel
    )
    covariance /= batch_size * num_batches * H * W - 1

    # Apply PCA
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
    return mean, bias, kernels, energy, cutoff_index


# @jax.jit
@partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"])
def transform(
    X: jnp.ndarray,
    mean: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: float,
    pool,
    win,
    stride,
    pad,
) -> jnp.ndarray:
    """Transform the data using the fitted PCA model."""
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
    N, H, W, C = X.shape
    X = _extract_patches(X, win, stride)

    X = X - mean
    X = X @ kernel
    X = X + bias

    out_h = (H - win) // stride + 1
    out_w = (W - win) // stride + 1
    X = rearrange(X, "(n h w) c -> n h w c", n=N, h=out_h, w=out_w)
    return X


class Saab:
    def __init__(self, pool, win, stride, pad, threshold, apply_bias: bool = False):
        self.pool, self.win, self.stride, self.pad = pool, win, stride, pad
        self.threshold = threshold
        self.apply_bias = apply_bias
        self.bias = []
        self.kernels = []
        self.mean = []
        self.energy = []

    def fit(self, X_batch, energy_previous: jnp.ndarray):
        """Fit the Saab model to the input data."""
        results = []
        for X_batch_channel, energy_prev in zip(X_batch, energy_previous):
            result = _fit(
                X_batch_channel,
                energy_prev,
                self.pool,
                self.win,
                self.stride,
                self.pad,
                self.threshold,
            )
            results.append(result)

        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = zip(
            *results
        )

        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = (
            jnp.array(self.mean),
            jnp.array(self.bias),
            jnp.array(self.kernels),
            jnp.array(self.energy),
            jnp.array(self.cutoff_index),
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
        X = jnp.concat(
            [
                transform(
                    X_channel,
                    mean,
                    kernel,
                    bias,
                    self.pool,
                    self.win,
                    self.stride,
                    self.pad,
                )
                for X_channel, mean, kernel, bias in zip(
                    X, self.mean, self.kernels, self.bias
                )
            ],
            axis=-1,
        )
        return X
