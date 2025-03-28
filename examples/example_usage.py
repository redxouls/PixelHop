if __name__ == "__main__":
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pixelhop.PixelHop import PixelHop
    from pixelhop.Layers import SaabLayer

    # jax.config.update("jax_explain_cache_misses", True)

    # Warm up JAX linear algebra JIT cache
    _ = jnp.linalg.eigh(np.random.randn(32, 32))

    # Define PixelHop model
    pixelhop = PixelHop(
        [
            SaabLayer(1, 7, 1, 3, threshold=0.00685),
            SaabLayer(2, 3, 1, 1, threshold=0.0035, channel_wise=True),
            SaabLayer(2, 3, 1, 1, threshold=0.0015, channel_wise=True),
        ]
    )

    print("\n[Model Configuration]\n", pixelhop)

    # Start profiler (optional)
    jax.profiler.start_trace("./jax-trace")

    # Simulated training data
    train_data = np.random.randn(2000, 512, 512, 3).astype(np.float32)

    # ------------------------
    # Training
    # ------------------------
    print("\n[Training] Fitting model...")
    start_time = time.time()
    pixelhop.fit(train_data, batch_size=20)
    print(f"[Training] Completed in {time.time() - start_time:.2f} seconds.")

    # ------------------------
    # Save the trained model
    # ------------------------
    # save_path = "models/pixelhop_trained.npz"
    # pixelhop.save(save_path)
    # print(f"[Checkpoint] Model saved to {save_path}")

    # ------------------------
    # Load the model from disk
    # ------------------------
    # loaded_pixelhop = PixelHop.load(save_path)
    # print("[Checkpoint] Model successfully loaded from disk.")

    # ------------------------
    # Inference on one batch
    # ------------------------
    test_sample = np.random.randn(20, 512, 512, 3).astype(np.float32)
    output = pixelhop.transform(test_sample)
    output.block_until_ready()
    print(f"[Inference] Output shape: {output.shape}")

    # ------------------------
    # Full batch inference
    # ------------------------
    print("\n[Inference] Starting full transform on new data...")
    X = np.random.randn(2000, 512, 512, 3).astype(np.float32)
    batch_size = 20
    num_batches = max(X.shape[0] // batch_size, 1)

    start_time = time.time()
    Xt = [
        loaded_pixelhop.transform(X_batch) for X_batch in np.array_split(X, num_batches)
    ]
    Xt[0].block_until_ready()
    total_time = time.time() - start_time

    print(f"[Inference] Completed {len(Xt)} batches in {total_time:.2f} seconds.")
    print(f"[Output Example] First batch shape: {Xt[0].shape}")
    print("[Profiler] Stopping JAX trace...")
    jax.profiler.stop_trace()
