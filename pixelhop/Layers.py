import jax.numpy as jnp
from typing import Optional, List, Tuple
from .Saab import Saab


class SaabLayer(Saab):
    def __init__(
        self,
        pool: int,
        win: int,
        stride: int,
        pad: int,
        threshold: float = 0.001,
        channel_wise: bool = False,
        apply_bias: bool = False,
    ):
        super().__init__(pool, win, stride, pad, threshold, channel_wise, apply_bias)

    def fit(
        self,
        X_batch: List[jnp.ndarray],
        energy_previous: Optional[jnp.ndarray],
        transform_previous,
    ) -> jnp.ndarray:
        return super().fit(X_batch, energy_previous, transform_previous)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        X = super().transform(X)
        return X

    def fit_transform(
        self,
        X_batch: List[jnp.ndarray],
        energy_previous: Optional[jnp.ndarray],
        transform_previous,
    ) -> Tuple[List[jnp.ndarray], jnp.ndarray]:
        energy_previous = self.fit(X_batch, energy_previous, transform_previous)
        return energy_previous

    def __str__(self) -> str:
        return (
            f"SaabLayer("
            f"pool={self.pool}, "
            f"win={self.win}, "
            f"stride={self.stride}, "
            f"pad={self.pad}, "
            f"threshold={self.threshold}, "
            f"channel_wise={self.channel_wise}, "
            f"apply_bias={self.apply_bias}"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()
