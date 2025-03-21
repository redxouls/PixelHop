import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial


@jax.jit
def _pca(covariance):
    """Perform Principal Component Analysis (PCA) on the covariance matrix."""
    eigen_values, eigen_vectors = jnp.linalg.eigh(covariance)
    ind = eigen_values.argsort()[::-1]
    return eigen_vectors.T[ind], eigen_values[ind]


@jax.jit
def _locate_cutoff(energy, threshold):
    """Find the index where energy drops below the threshold."""
    return jnp.argmax(jnp.concatenate([energy < threshold, jnp.array([True])]))


@partial(jax.jit, static_argnames=["pool", "pad", "win", "stride", "out_h", "out_w"])
def _extract_patches(X, pool, pad, win, stride, out_h, out_w):
    B = X.shape[0]
    C = X.shape[-1]
    """Extract sliding patches from the input tensor."""
    # Compute output height and width after pooling
    X = jax.lax.reduce_window(
        X,
        -jnp.inf,
        jax.lax.max,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        padding="VALID",
    )

    X = jnp.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")

    def get_patch(i, j, X):
        return jax.lax.dynamic_slice(
            X, (0, i * stride, j * stride, 0), (B, win, win, C)
        )

    return jax.vmap(
        lambda i: jax.vmap(lambda j: get_patch(i, j, X))(jnp.arange(out_w))
    )(jnp.arange(out_h))


@jax.jit
def _statistics(patches):
    """Compute local mean and bias for patches."""
    dc_local = jnp.mean(patches, axis=-1, keepdims=True)
    X_centered = patches - dc_local
    return (
        dc_local,
        jnp.max(jnp.linalg.norm(X_centered, axis=-1)),
        jnp.mean(X_centered, axis=0, keepdims=True),
    )


def _compute_statistics(X_batch, extract_patches, num_channel, num_kernel):
    """Compute mean and bias statistics across a batch."""
    dc_batch = []
    mean, bias = jnp.zeros((num_channel, 1, num_kernel)), jnp.zeros(num_channel)

    for X_cur in X_batch:
        patches = extract_patches(X_cur)
        dc_local, bias_local, mean_local = jax.vmap(_statistics)(patches)
        mean += mean_local
        bias = jnp.maximum(bias, bias_local)
        dc_batch.append(dc_local)

    return mean / len(X_batch), bias, dc_batch


@jax.jit
def _covariance(patches, dc, mean):
    """Compute covariance of patches."""
    X_centered = patches - dc - mean
    return jnp.einsum(
        "...i,...j->ij", X_centered, X_centered, precision=jax.lax.Precision.HIGHEST
    )


def _compute_covariance(
    X_batch, dc_batch, mean, extract_patches, num_channel, num_kernel
):
    """Compute covariance matrix across a batch."""
    covariance = jnp.zeros((num_channel, num_kernel, num_kernel))
    for X_cur, dc in zip(X_batch, dc_batch):
        covariance += jax.vmap(_covariance)(extract_patches(X_cur), dc, mean)
    return covariance


@jax.jit
def _compute_kernel(covariance, dc_batch, energy_previous, threshold):
    """Compute PCA kernels and energy distribution."""
    print("Compiling _compute_kernel()", covariance.shape)
    num_kernel = covariance.shape[0]
    kernels, eigen_values = _pca(covariance)
    kernels = jnp.concatenate(
        (jnp.ones((1, num_kernel)) / jnp.sqrt(num_kernel), kernels[:-1]), axis=0
    ).T

    dc = jnp.concatenate(dc_batch)
    largest_ev = jnp.var(dc * jnp.sqrt(num_kernel))
    energy = jnp.concatenate([jnp.array([largest_ev]), eigen_values[:-1]]) / jnp.sum(
        eigen_values
    )
    energy *= energy_previous

    return kernels, energy, _locate_cutoff(energy, threshold)


def _fit(X_batch, energy_previous, extract_patches, threshold):
    """Fit the Saab model to a batch of input data."""
    num_batches = len(X_batch)
    batch_size, H, W, _ = X_batch[0].shape
    num_channel, _, num_kernel = extract_patches(X_batch[0][0:1]).shape

    mean, bias, dc_batch = _compute_statistics(
        X_batch, extract_patches, num_channel, num_kernel
    )
    covariance = _compute_covariance(
        X_batch, dc_batch, mean, extract_patches, num_channel, num_kernel
    )
    covariance /= batch_size * num_batches * H * W - 1

    threshold = jnp.ones((num_channel,)) * threshold
    kernels, energy, cutoff_index = jax.vmap(_compute_kernel)(
        covariance, dc_batch, energy_previous, threshold
    )

    return mean, bias, kernels, energy, cutoff_index


@jax.jit
def _transform_channel(X, mean, kernel, bias):
    """Apply learned transformation."""
    print("Compiling _transform_channel()", X.shape)
    return (X - mean) @ kernel + bias


@partial(jax.jit, static_argnames=["out_h", "out_w"])
def _resize_output(X, out_h, out_w):
    print("Compiling _resize_output()", out_h, out_w)
    X = jnp.concatenate(X, axis=-1)
    return rearrange(X, "(n h w) c -> n h w c", h=out_h, w=out_w)


@partial(jax.jit, static_argnames=["extract_patches", "out_h", "out_w"])
def _transform(X, mean, bias, kernels, extract_patches, out_h, out_w):
    print("Compiling _transform()", X.shape)
    patches = tuple(jnp.unstack(extract_patches(X)))
    X = jax.tree.map(_transform_channel, patches, mean, kernels, bias)
    return _resize_output(X, out_h, out_w)


class Saab:
    """Saab feature extraction model."""

    def __init__(
        self, pool, win, stride, pad, threshold, channel_wise, apply_bias=False
    ):
        self.pool, self.win, self.stride, self.pad = pool, win, stride, pad
        self.threshold = threshold
        self.channel_wise = channel_wise
        self.apply_bias = apply_bias
        self.bias, self.kernels, self.mean, self.energy = [], [], [], []

    @partial(jax.jit, static_argnames=["self"])
    def extract_patches(self, X):
        _, H, W, _ = X.shape
        print(
            "Compiling extract_patches()",
            X.shape,
            self.channel_wise,
            self.pool,
            self.pad,
            self.win,
            self.stride,
            self.out_h,
            self.out_w,
        )
        patches = _extract_patches(
            X, self.pool, self.pad, self.win, self.stride, self.out_h, self.out_w
        )
        if self.channel_wise:
            return rearrange(patches, "h w b p q c -> c (b h w) (p q)")
        else:
            return rearrange(patches, "h w b p q c -> 1 (b h w) (p q c)")

    def fit(self, X_batch, energy_previous, H, W, transform_previous):
        """Fit the Saab model to input data."""
        # Compute output height and width after pooling
        H = (H - self.pool) // self.pool + 1
        W = (W - self.pool) // self.pool + 1

        # Compute output height and width after padding and window sliding
        self.out_h, self.out_w = (
            (H + 2 * self.pad - self.win) // self.stride + 1,
            (W + 2 * self.pad - self.win) // self.stride + 1,
        )

        if not self.channel_wise:
            energy_previous = jnp.ones(1)

        @jax.jit
        def extract_patches(X):
            return self.extract_patches(transform_previous(X))

        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = _fit(
            X_batch, energy_previous, extract_patches, self.threshold
        )

        self.energy = jnp.concatenate(
            [
                jax.lax.dynamic_slice(self.energy[i], (0,), (self.cutoff_index[i],))
                for i in range(len(self.cutoff_index))
            ],
            axis=-1,
        )

        print(self.energy[:30])

        self.kernels = tuple(
            jax.lax.dynamic_slice(
                self.kernels[i],
                (0, 0),
                (self.kernels[i].shape[0], self.cutoff_index[i]),
            )
            for i in range(len(self.cutoff_index))
        )

        if not self.apply_bias:
            self.bias = jnp.zeros_like(self.bias)

        self.mean = tuple(jnp.unstack(self.mean))
        self.bias = tuple(jnp.unstack(self.bias))
        print(len(self.kernels), len(self.mean), len(self.bias))

        return self.energy, self.out_h, self.out_w

    def transform(self, X):
        """Transform input data using the fitted Saab model."""
        return _transform(
            X,
            self.mean,
            self.bias,
            self.kernels,
            self.extract_patches,
            self.out_h,
            self.out_w,
        )
