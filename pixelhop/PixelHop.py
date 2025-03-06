import numpy as np
import jax.numpy as jnp


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
        num_batch = max(X.shape[0] // batch_size, 1)
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

    def __str__(self):
        """
        Return a string representation of the PixelHop instance.
        """
        main_str = self.__class__.__name__ + "(\n"
        for i, (layer, shrink_layer) in enumerate(zip(self.layers, self.shrink_layers)):
            main_str += f"  (shrink_{i}): {shrink_layer}\n"
            main_str += f"  (saab_{i}): {layer}\n"
        main_str += ")"
        return main_str

    def __repr__(self):
        """
        Return a detailed string representation of the PixelHop instance.
        """
        return self.__str__()
