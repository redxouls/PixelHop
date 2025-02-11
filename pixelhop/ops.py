import jax
import flax
import jax.numpy as jnp
from einops import rearrange
from functools import partial


@partial(jax.jit, static_argnames=["pool", "win", "stride", "pad"])
def shrink(X, pool, win, stride, pad):
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


if __name__ == "__main__":
    import numpy as np
    from skimage.measure import block_reduce
    from skimage.util import view_as_windows

    def shrink_old(X, shrink_args):
        # ---- max pooling----
        pool = shrink_args["pool"]
        # TODO: fill in the rest of max pooling
        X = block_reduce(X, (1, pool, pool, 1), np.max)

        # ---- neighborhood construction
        win = shrink_args["win"]
        stride = shrink_args["stride"]
        pad = shrink_args["pad"]

        # TODO: fill in the rest of neighborhood construction
        # numpy padding
        X = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
        X = view_as_windows(
            X, (1, win, win, X.shape[-1]), (1, stride, stride, X.shape[-1])
        )

        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    X = np.random.rand(100, 16, 16, 3)
    # a = flax.linen.max_pool(X, (1, 3, 3))
    # print(a.shape)

    shrink_args = {"func": shrink, "win": 3, "stride": 1, "pad": 1, "pool": 1}
    Y = shrink_args["func"](
        X,
        shrink_args["pool"],
        shrink_args["win"],
        shrink_args["stride"],
        shrink_args["pad"],
    )
    print(Y.shape)

    shrink_args = {"func": shrink, "win": 5, "stride": 1, "pad": 1, "pool": 1}
    Y = shrink_args["func"](
        X,
        shrink_args["pool"],
        shrink_args["win"],
        shrink_args["stride"],
        shrink_args["pad"],
    )
    print(Y.shape)
