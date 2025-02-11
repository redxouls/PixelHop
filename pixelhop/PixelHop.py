from pixelhop.Layers import SaabLayer, ShrinkLayer


class PixelHop:
    def __init__(self):
        self.layers = [
            ShrinkLayer(pool=1, win=3, stride=1, pad=1),
            SaabLayer(thresholds=[0.002, 0.0001], channel_wise=False, apply_bias=False),
            ShrinkLayer(pool=1, win=3, stride=1, pad=1),
            SaabLayer(thresholds=[0.002, 0.0001], channel_wise=True, apply_bias=False),
        ]

    def fit(self, X):
        # print(f"Input shape: {X.shape}")
        for layer in self.layers:
            X = layer.fit(X)
            # print(layer)
            # print(f"Output Dimension: {X.shape}\n")

    def transform(self, X):
        # print(f"Input shape: {X.shape}")
        for layer in self.layers:
            X = layer.transform(X)
            # print(layer)
            # print(f"Output Dimension: {X.shape}\n")
        return X

    def __str__(self):
        return f"{self.__class__!s}\n{self.layers!r}"


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
    # print(Xt)
    print(time.time() - start)
    print("done")
