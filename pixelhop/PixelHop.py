import numpy as np


class PixelHop:
    """
    PixelHop class that applies multiple Saab and Shrink layers to the input data.
    """

    def __init__(self, layers=[]):
        """
        Initialize a PixelHop instance.

        Parameters
        ----------
        layers : list
            List of SaabLayer instances.
        """
        self.layers = layers

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

        energy_previous = None
        num_batch = max(X.shape[0] // batch_size, 1)
        X_batch = np.array_split(X[: num_batch * batch_size], num_batch)
        for i, layer in enumerate(self.layers):
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
        jnp.ndarray
            Transformed data.
        """
        for layer in self.layers:
            X = layer.transform(X)
        return X

    def __str__(self):
        """
        Return a string representation of the PixelHop instance.
        """
        main_str = self.__class__.__name__ + "(\n"
        for i, layer in enumerate(self.layers):
            main_str += f"  (saab_{i}): {layer}\n"
        main_str += ")"
        return main_str

    def __repr__(self):
        """
        Return a detailed string representation of the PixelHop instance.
        """
        return self.__str__()
