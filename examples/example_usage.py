if __name__ == "__main__":
    import time
    import jax
    import numpy as np
    import jax.numpy as jnp
    from pixelhop.PixelHop import PixelHop
    from pixelhop.Layers import SaabLayer, ShrinkLayer

    pixelhop = PixelHop(
        layers=[
            SaabLayer(threshold=0.007, channel_wise=False, apply_bias=False),
            SaabLayer(threshold=0.001, channel_wise=True, apply_bias=True),
        ],
        shrink_layers=[
            ShrinkLayer(pool=1, win=7, stride=1, pad=3),
            ShrinkLayer(pool=1, win=3, stride=1, pad=1),
        ],
    )
    print(pixelhop)

    # jax.profiler.start_trace("./jax-trace")
    sample = np.random.randn(10000, 32, 32, 3)

    start = time.time()
    pixelhop.fit(sample, batch_size=4096)

    batch_size = 4096
    X = np.random.randn(20000, 32, 32, 3)
    num_batches = (X.shape[0] // batch_size) + 1
    Xt = [
        jax.device_get(pixelhop.transform(X_batch))
        for X_batch in jnp.array_split(X, num_batches)
    ]
    Xt = jnp.concatenate(Xt)
    print(Xt)
    print(Xt.shape)
    print(time.time() - start)
    # jax.profiler.stop_trace()
