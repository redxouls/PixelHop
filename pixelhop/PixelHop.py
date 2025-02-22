import numpy as np


class PixelHop:
    """
    PixelHop class that applies multiple Saab and Shrink layers to the input data.
    """

    def __init__(self, layers=[], shrink_layers=[]):
        """
        Initialize a PixelHop instance.

        Parameters
        ----------
        layers : list
            List of SaabLayer instances.
        shrink_layers : list
            List of ShrinkLayer instances.
        """
        self.layers = layers
        self.shrink_layers = shrink_layers

    def fit(self, X, batch_size):
        """
        Fit the PixelHop model to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        batch_size : int
            Batch size for processing.
        """
        num_batch = X.shape[0] // batch_size
        X_batch = np.split(X[: num_batch * batch_size], num_batch)

        energy_previous = None
        for layer, shrink_layer in zip(self.layers, self.shrink_layers):
            X_batch = shrink_layer.transform_batch(X_batch)
            X_batch, energy_previous = layer.fit_transform(X_batch, energy_previous)

    def transform(self, X):
        """
        Transform the input data using the fitted PixelHop model.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        for layer, shrink_layer in zip(self.layers, self.shrink_layers):
            X = shrink_layer.transform(X)
            X = layer.transform(X)
        return X
