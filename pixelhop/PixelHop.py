import numpy as np
from pixelhop.Layers import SaabLayer, ShrinkLayer


class PixelHop:
    def __init__(self):
        self.layers = [
            SaabLayer(threshold=0.007, channel_wise=False, apply_bias=False),
            # SaabLayer(threshold=0.001, channel_wise=True, apply_bias=True),
        ]
        self.shrink_layers = [
            ShrinkLayer(pool=1, win=7, stride=1, pad=3),
            # ShrinkLayer(pool=1, win=3, stride=1, pad=1),
        ]

    def fit(self, X):
        # print(f"Input shape: {X.shape}")
        energy_previous = None
        for layer, shrink_layer in zip(self.layers, self.shrink_layers):
            X_batches = shrink_layer.transform(X, split_batch=True)
            layer.fit(X_batches, energy_previous=energy_previous)
            # print(layer)
            # print(f"Output Dimension: {X.shape}\n")

    def transform(self, X):
        # print(f"Input shape: {X.shape}")
        for layer, shrink_layer in zip(self.layers, self.shrink_layers):
            X = shrink_layer.transform(X)
            X = layer.transform(X)
            # print(layer)
            # print(f"Output Dimension: {X.shape}\n")
        return X


if __name__ == "__main__":
    import time
    import numpy as np

    pixelhop = PixelHop()
    pixelhop.fit(np.random.randn(10, 16, 16, 3))

    print()
    print("Test")
    start = time.time()
    pixelhop = PixelHop()
    pixelhop.fit(np.random.randn(5000, 16, 16, 3))
    Xt = pixelhop.transform(np.random.randn(1000, 16, 16, 3))
    print(Xt.shape)
    print(time.time() - start)
    print("done")
