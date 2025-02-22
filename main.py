if __name__ == "__main__":
    # import os

    # os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import time
    import jax
    import numpy as np
    import jax.numpy as jnp
    from pixelhop.PixelHop import PixelHop

    pixelhop = PixelHop()
    print(pixelhop)

    jax.profiler.start_trace("./jax-trace")
    sample = np.random.randn(10000, 32, 32, 3)

    start = time.time()
    pixelhop.fit(sample)

    batch_size = 8192
    X = np.random.randn(20000, 32, 32, 3)
    num_batches = (X.shape[0] // batch_size) + 1
    Xt = [
        jax.device_get(pixelhop.transform(X_batch))
        for X_batch in jnp.array_split(X, num_batches)
    ]
    Xt = jnp.concat(Xt)
    print(Xt)
    print(Xt.shape)
    # # print(Xt.device)
    # print(Xt.shape)
    print(time.time() - start)
    jax.profiler.stop_trace()
