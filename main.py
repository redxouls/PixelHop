from pixelhop.PixelHop import PixelHop

if __name__ == "__main__":
    import time
    import numpy as np

    pixelhop = PixelHop()
    pixelhop.fit(np.random.randn(10, 16, 16, 3))
    print(pixelhop)

    start = time.time()
    pixelhop = PixelHop()
    pixelhop.fit(np.random.randn(5000, 16, 16, 3))
    Xt = pixelhop.transform(np.random.randn(1000, 16, 16, 3))
    print(Xt.shape)
    print(time.time() - start)
    print("done")
