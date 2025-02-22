import jax
import flax
import jax.numpy as jnp
from einops import rearrange
# from functools import partial


# @partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"])
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
    X = rearrange(X, "b h w c -> b c h w")

    # ---- max pooling----
    X = flax.linen.max_pool(X, (pool, pool), strides=(pool, pool))

    # ---- neighborhood construction
    X = jnp.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
    X = jax.lax.conv_general_dilated_patches(
        lhs=X, filter_shape=(win, win), window_strides=(stride, stride), padding="VALID"
    )

    X = rearrange(X, "b (c p) h w -> b h w p c", p=win**2)
    return X
