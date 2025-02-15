if __name__ == "__main__":
    # import os

    # os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    from pixelhop.PixelHop import PixelHop

    pixelhop = PixelHop()
    print(pixelhop)

    pixelhop.fit(np.random.randn(10, 32, 32, 3))
    Xt = pixelhop.transform(np.random.randn(10, 32, 32, 3))
    # print(Xt)

    start = time.time()
    pixelhop.fit(np.random.randn(5000, 32, 32, 3))

    batch_size = 512
    X = np.random.randn(1000, 32, 32, 3)
    Xt = pixelhop.transform(X)
    num_batches = (X.shape[0] // batch_size) + 1
    Xt = [
        jax.device_get(pixelhop.transform(X_batch))
        for X_batch in jnp.split(X, num_batches)
    ]
    Xt = jnp.concat(Xt)
    print(Xt)
    print(Xt.device)
    print(Xt.shape)
    print(time.time() - start)
