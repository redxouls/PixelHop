if __name__ == "__main__":
    import time
    import jax.numpy as jnp
    import numpy as np
    from pixelhop.PixelHop import PixelHop
    from pixelhop.Layers import SaabLayer

    jnp.linalg.eigh(np.random.randn(147, 147))

    pixelhop = PixelHop(
        layers=[
            SaabLayer(
                pool=1,
                win=7,
                stride=1,
                pad=3,
                threshold=0.007,
                channel_wise=False,
                apply_bias=False,
            ),
            SaabLayer(
                pool=1,
                win=3,
                stride=1,
                pad=1,
                threshold=0.00082,
                channel_wise=True,
                apply_bias=False,
            ),
            # SaabLayer(threshold=0.001, channel_wise=False, apply_bias=False),
        ]
    )

    print(pixelhop)

    # jax.profiler.start_trace("./jax-trace")
    sample = np.random.randn(10000, 64, 64, 3)
    pixelhop.fit(sample, batch_size=100)

    print("Start training...")
    start = time.time()
    pixelhop.fit(sample, batch_size=100)
    print(time.time() - start)

    batch_size = 2000
    X = np.random.randn(1000, 64, 64, 3)
    print("Start transform...")
    start = time.time()
    num_batches = (X.shape[0] // batch_size) + 1
    Xt = [pixelhop.transform(X_batch) for X_batch in np.array_split(X, num_batches)]
    print(Xt[0].shape)
    print(time.time() - start)
    # jax.profiler.stop_trace()
