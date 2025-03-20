if __name__ == "__main__":
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pixelhop.PixelHop import PixelHop
    from pixelhop.Layers import SaabLayer

    # jax.config.update("jax_enable_x64", True)  # Enables float64 precision globally
    jnp.linalg.eigh(np.random.randn(147, 147))

    pixelhop = PixelHop(
        layers=[
            SaabLayer(
                pool=1,
                win=7,
                stride=2,
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
                threshold=0.003,
                channel_wise=True,
                apply_bias=False,
            ),
            # SaabLayer(
            #     pool=1,
            #     win=3,
            #     stride=1,
            #     pad=1,
            #     threshold=0.00081,
            #     channel_wise=True,
            #     apply_bias=False,
            # ),
        ]
    )

    print(pixelhop)

    jax.profiler.start_trace("./jax-trace")
    sample = np.random.randn(100, 128, 128, 3)
    pixelhop.fit(sample, batch_size=50)

    print("Start training...")
    start = time.time()
    pixelhop.fit(sample, batch_size=50)
    print(time.time() - start)

    batch_size = 200
    print(pixelhop.transform(np.random.randn(batch_size, 128, 128, 3)))
    X = np.random.randn(2000, 128, 128, 3)
    print("Start transform...")
    start = time.time()
    num_batches = max(X.shape[0] // batch_size, 1)
    Xt = [pixelhop.transform(X_batch) for X_batch in np.array_split(X, num_batches)]
    print(Xt[0].shape)
    print(Xt[0])
    print(time.time() - start)
    jax.profiler.stop_trace()
