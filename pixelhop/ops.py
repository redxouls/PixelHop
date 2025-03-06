import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# @partial(jax.jit, static_argnames=["win", "stride"], backend="cpu")
# def extract_patches_dynamic(X, win, stride):
#     batch, h, w, c = X.shape
#     num_patches_h = (h - win) // stride + 1
#     num_patches_w = (w - win) // stride + 1

#     def slice_window(b, i, j):
#         """Slice out a patch without copying."""
#         return (
#             jax.lax.dynamic_slice(X, (b, i * stride, j * stride, 0), (1, win, win, c))
#             .squeeze(0)
#             .reshape(win * win, -1)
#         )

#     # Vectorized version using vmap
#     extract_patches_vmap = jax.vmap(
#         jax.vmap(jax.vmap(slice_window, (None, None, 0)), (None, 0, None)),
#         (0, None, None),
#     )

#     return extract_patches_vmap(
#         jnp.arange(batch), jnp.arange(num_patches_h), jnp.arange(num_patches_w)
#     )


def extract_patches_numpy(X, win, stride):
    """NumPy equivalent patch extraction using efficient slicing (view-based)."""
    batch, h, w, c = X.shape
    num_patches_h = (h - win) // stride + 1
    num_patches_w = (w - win) // stride + 1

    # Efficient patch extraction via strides
    patches = np.lib.stride_tricks.sliding_window_view(X, (1, win, win, c))[
        ..., ::stride, ::stride, :, :, :, :
    ]
    patches = patches.reshape(batch, num_patches_h, num_patches_w, win * win, c)
    return patches


# @partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"], backend="cpu")
def shrink(X, pool, win, stride, pad):
    """
    Apply shrink operation using max pooling and neighborhood construction.

    Parameters
    ----------
    X : jnp.ndarray
        Input array of shape (batch, height, width, channels).
    pool : int
        Pooling window size.
    win : int
        Neighborhood window size.
    stride : int
        Stride size for neighborhood construction.
    pad : int
        Padding size.

    Returns
    -------
    jnp.ndarray
        Transformed array.
    """

    # ---- max pooling ----
    X = jax.lax.reduce_window(
        X,
        -jnp.inf,
        jax.lax.max,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        padding="VALID",
    )

    # ---- neighborhood construction ----
    X = jnp.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
    X = extract_patches_numpy(X, win, stride)
    return X
