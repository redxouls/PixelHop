import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial


@jax.jit
def _compute_pca(covariance):
    """
    Perform Principal Component Analysis (PCA) on a covariance matrix.

    Args:
        covariance (jax.numpy.ndarray): A symmetric covariance matrix.

    Returns:
        tuple: Sorted eigenvectors and eigenvalues in descending order.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    sorted_indices = eigenvalues.argsort()[::-1]
    return eigenvectors.T[sorted_indices], eigenvalues[sorted_indices]


@jax.jit
def _find_energy_cutoff_index(energy, threshold):
    """
    Find the index at which energy drops below a specified threshold.

    Args:
        energy (jax.numpy.ndarray): Array of energy values.
        threshold (float): Threshold to determine cutoff.

    Returns:
        int: Index of the first value below threshold.
    """
    return jnp.argmax(jnp.concatenate([energy < threshold, jnp.array([True])]))


@partial(jax.jit, static_argnames=["pool", "pad", "win", "stride", "out_h", "out_w"])
def _extract_patches(input_tensor, pool, pad, win, stride, out_h, out_w):
    batch_size = input_tensor.shape[0]
    channels = input_tensor.shape[-1]

    pooled = jax.lax.reduce_window(
        input_tensor,
        -jnp.inf,
        jax.lax.max,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        padding="VALID",
    )

    padded = jnp.pad(pooled, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")

    def get_patch(i, j, x):
        return jax.lax.dynamic_slice(
            x, (0, i * stride, j * stride, 0), (batch_size, win, win, channels)
        )

    return jax.vmap(
        lambda i: jax.vmap(lambda j: get_patch(i, j, padded))(jnp.arange(out_w))
    )(jnp.arange(out_h))


@jax.jit
def _compute_patch_statistics(patches):
    local_mean = jnp.mean(patches, axis=-1, keepdims=True)
    dc_mean = jnp.mean(local_mean)
    dc_var = jnp.var(local_mean)

    centered = patches - local_mean
    max_norm = jnp.max(jnp.linalg.norm(centered, axis=-1))
    mean_centered = jnp.mean(centered, axis=0, keepdims=True)
    return dc_mean, dc_var, max_norm, mean_centered


def _aggregate_statistics(input_batch, patch_extractor, num_channels, num_kernels):
    batch_dc_means = []
    batch_dc_vars = []
    global_mean = jnp.zeros((num_channels, 1, num_kernels))
    global_bias = jnp.zeros(num_channels)

    _compute_patch_statistics_batch = jax.jit(jax.vmap(_compute_patch_statistics))
    for sample in input_batch:
        patches = patch_extractor(sample)
        dc_mean, dc_var, max_norm, mean_centered = _compute_patch_statistics_batch(
            patches
        )
        global_mean += mean_centered
        global_bias = jnp.maximum(global_bias, max_norm)

        batch_dc_means.append(dc_mean)
        batch_dc_vars.append(dc_var)

    batch_dc_means = jnp.asarray(batch_dc_means)
    batch_dc_vars = jnp.asarray(batch_dc_vars)

    global_dc_mean = jnp.mean(batch_dc_means, axis=0)
    global_dc_var = jnp.mean(
        (batch_dc_vars + (batch_dc_means - global_dc_mean) ** 2), axis=0
    )
    return global_mean / len(input_batch), global_bias, global_dc_var


@jax.jit
def _compute_covariance_matrix(patches, global_mean):
    local_mean = jnp.mean(patches, axis=-1, keepdims=True)
    centered = patches - local_mean - global_mean
    return jnp.einsum(
        "...i,...j->ij", centered, centered, precision=jax.lax.Precision.HIGHEST
    )


def _aggregate_covariance(
    input_batch, global_mean, patch_extractor, num_channels, num_kernels
):
    _compute_covariance_matrix_batch = jax.jit(jax.vmap(_compute_covariance_matrix))
    covariance_matrix = jnp.zeros((num_channels, num_kernels, num_kernels))
    for sample in input_batch:
        patches = patch_extractor(sample)
        covariance_matrix += _compute_covariance_matrix_batch(patches, global_mean)
    return covariance_matrix


@jax.jit
def _compute_kernels_and_energy(covariance, global_dc_var, previous_energy, threshold):
    eigenvectors, eigenvalues = _compute_pca(covariance)
    num_kernels = covariance.shape[0]

    kernels = jnp.concatenate(
        (jnp.ones((1, num_kernels)) / jnp.sqrt(num_kernels), eigenvectors[:-1]), axis=0
    ).T

    leading_ev = global_dc_var * num_kernels
    energy_values = jnp.concatenate(
        [jnp.array([leading_ev]), eigenvalues[:-1]]
    ) / jnp.sum(eigenvalues)
    energy_values *= previous_energy

    return kernels, energy_values, _find_energy_cutoff_index(energy_values, threshold)


def _fit_saab(input_batch, previous_energy, patch_extractor, threshold):
    num_batches = len(input_batch)
    batch_size, height, width, _ = input_batch[0].shape
    num_channels, _, num_kernels = patch_extractor(input_batch[0][0:1]).shape

    global_mean, global_bias, global_dc_var = _aggregate_statistics(
        input_batch, patch_extractor, num_channels, num_kernels
    )

    covariance = _aggregate_covariance(
        input_batch,
        global_mean,
        patch_extractor,
        num_channels,
        num_kernels,
    )
    covariance /= batch_size * num_batches * height * width - 1

    threshold_array = jnp.ones((num_channels,)) * threshold
    _compute_kernels_and_energy_batch = jax.jit(jax.vmap(_compute_kernels_and_energy))
    kernels, energy_values, cutoff_indices = _compute_kernels_and_energy_batch(
        covariance, global_dc_var, previous_energy, threshold_array
    )

    return global_mean, global_bias, kernels, energy_values, cutoff_indices


@jax.jit
def _apply_kernel_transform(patch, mean, kernel, bias):
    return (patch - mean) @ kernel + bias


@partial(jax.jit, static_argnames=["out_h", "out_w"])
def _reshape_output(output_list, out_h, out_w):
    concatenated = jnp.concatenate(output_list, axis=-1)
    return rearrange(concatenated, "(n h w) c -> n h w c", h=out_h, w=out_w)


@partial(jax.jit, static_argnames=["patch_extractor", "out_h", "out_w"])
def _apply_saab_transform(
    input_tensor, mean, bias, kernels, patch_extractor, out_h, out_w
):
    patches = tuple(jnp.unstack(patch_extractor(input_tensor)))
    transformed = jax.tree.map(_apply_kernel_transform, patches, mean, kernels, bias)
    return _reshape_output(transformed, out_h, out_w)


class Saab:
    """Saab (Subspace Approximation with Adjusted Bias) feature extraction model."""

    def __init__(
        self, pool, win, stride, pad, threshold, channel_wise, apply_bias=False
    ):
        self.pool = pool
        self.win = win
        self.stride = stride
        self.pad = pad
        self.threshold = threshold
        self.channel_wise = channel_wise
        self.apply_bias = apply_bias

        self.bias = []
        self.kernels = []
        self.mean = []
        self.energy = []

    @partial(jax.jit, static_argnames=["self"])
    def extract_patches(self, input_tensor):
        patches = _extract_patches(
            input_tensor,
            self.pool,
            self.pad,
            self.win,
            self.stride,
            self.out_h,
            self.out_w,
        )
        if self.channel_wise:
            return rearrange(patches, "h w b p q c -> c (b h w) (p q)")
        else:
            return rearrange(patches, "h w b p q c -> 1 (b h w) (p q c)")

    def fit(self, input_batch, previous_energy, height, width, previous_transform):
        """
        Fit the Saab model using the input batch.

        Args:
            input_batch (list): Batch of input tensors.
            previous_energy (jax.numpy.ndarray): Energy from previous layer.
            height (int): Height of input images.
            width (int): Width of input images.
            previous_transform (Callable): Transformation applied before this layer.

        Returns:
            tuple: (energy, output_height, output_width)
        """
        height = (height - self.pool) // self.pool + 1
        width = (width - self.pool) // self.pool + 1

        self.out_h = (height + 2 * self.pad - self.win) // self.stride + 1
        self.out_w = (width + 2 * self.pad - self.win) // self.stride + 1

        if not self.channel_wise:
            previous_energy = jnp.ones(1)

        @jax.jit
        def patch_extractor(x):
            return self.extract_patches(previous_transform(x))

        self.mean, self.bias, self.kernels, self.energy, self.cutoff_index = _fit_saab(
            input_batch, previous_energy, patch_extractor, self.threshold
        )

        self.energy = jnp.concatenate(
            [
                jax.lax.dynamic_slice(self.energy[i], (0,), (self.cutoff_index[i],))
                for i in range(len(self.cutoff_index))
            ],
            axis=-1,
        )

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

        return self.energy, self.out_h, self.out_w

    def transform(self, input_tensor):
        """
        Transform input data using the fitted Saab model.

        Args:
            input_tensor (jax.numpy.ndarray): Input tensor.

        Returns:
            jax.numpy.ndarray: Transformed tensor.
        """
        return _apply_saab_transform(
            input_tensor,
            self.mean,
            self.bias,
            self.kernels,
            self.extract_patches,
            self.out_h,
            self.out_w,
        )
