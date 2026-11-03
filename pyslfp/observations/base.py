"""
The shape every observation model shares.

An observation model takes its observable as an operator from the model space
into the observable's own space (a field, a displacement, the components of
the potential change) and adds what the measurement does: the sampling into a
data space, the network's metadata, and the noise. Whatever built the
observable holds the physics, direct terms included, so one model serves an
elastic set-up and a viscoelastic run alike.
"""

from __future__ import annotations

import copy
from typing import Optional, Union

import numpy as np

from pygeoinf import GaussianMeasure, HilbertSpace, LinearOperator

__all__ = ["ObservationModel"]

#: What a noise may be given as: a measure on the data space, one standard
#: deviation for every datum, or one per datum.
Noise = Union[GaussianMeasure, float, np.ndarray, None]


class ObservationModel:
    """
    An observable sampled into a data space, with an optional noise.

    `forward_operator` is `sampling_operator @ observable`, from the model
    space to the data space.
    """

    def __init__(
        self,
        observable: LinearOperator,
        sampling: LinearOperator,
        /,
        *,
        noise: Noise = None,
    ) -> None:
        """
        Args:
            observable: The observable, an operator from the model space into
                the observable's space.
            sampling: The measurement, from the observable's space into the
                data space.
            noise: A Gaussian measure on the data space; or a standard
                deviation, one for every datum or an array of one per datum,
                for independent noise; or None.

        Raises:
            ValueError: If the sampling does not act on the observable's
                space, or a noise measure is not on the data space.
        """
        if sampling.domain != observable.codomain:
            raise ValueError("the sampling must act on the observable's space")
        self._observable = observable
        self._sampling = sampling
        self._forward = sampling @ observable
        self._noise = self._as_noise(noise)

    def _as_noise(self, noise: Noise) -> Optional[GaussianMeasure]:
        if noise is None:
            return None
        if isinstance(noise, GaussianMeasure):
            if noise.domain != self.data_space:
                raise ValueError("the noise must be a measure on the data space")
            return noise
        deviations = np.asarray(noise, dtype=float)
        if deviations.ndim == 0:
            return GaussianMeasure.from_standard_deviation(
                self.data_space, float(deviations)
            )
        return GaussianMeasure.from_standard_deviations(self.data_space, deviations)

    @property
    def observable(self) -> LinearOperator:
        """The observable, from the model space into its own space."""
        return self._observable

    @property
    def sampling_operator(self) -> LinearOperator:
        """The measurement, from the observable's space into the data space."""
        return self._sampling

    @property
    def forward_operator(self) -> LinearOperator:
        """From the model space to the data space."""
        return self._forward

    @property
    def model_space(self) -> HilbertSpace:
        """The space the observable acts on."""
        return self._observable.domain

    @property
    def data_space(self) -> HilbertSpace:
        """The space of the data."""
        return self._sampling.codomain

    @property
    def noise(self) -> Optional[GaussianMeasure]:
        """The noise, a Gaussian measure on the data space, or None."""
        return self._noise

    def with_noise(self, noise: Noise, /) -> ObservationModel:
        """The same model with another noise, given as for the constructor."""
        other = copy.copy(self)
        other._noise = other._as_noise(noise)
        return other
