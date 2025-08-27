"""
Module for the LoveNumbers class, which handles loading and processing
of elastic Love numbers for glacial isostatic adjustment models.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

from . import DATADIR

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .physical_parameters import EarthModelParameters


class LoveNumbers:
    """
    A class to load, non-dimensionalize, and provide elastic Love numbers.
    """

    def __init__(
        self,
        lmax: int,
        params: EarthModelParameters,
        /,
        *,
        file: Optional[str] = None,
    ):
        """
        Initializes the LoveNumbers object.

        Args:
            lmax: The maximum spherical harmonic degree to load.
            params: An EarthModelParameters instance containing the
                non-dimensionalization scales.
            file: Path to the Love number data file. If None, a default
                file based on the PREM model is used.
        """

        if file is None:
            file = DATADIR + "/love_numbers/PREM_4096.dat"

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if lmax > data_degree:
            raise ValueError(
                f"lmax ({lmax}) is larger than the maximum degree "
                f"in the Love number file ({data_degree})."
            )

        # Non-dimensionalize the Love numbers using the provided parameters
        self._h_u = data[: lmax + 1, 1] * params.load_scale / params.length_scale
        self._k_u = (
            data[: lmax + 1, 2]
            * params.load_scale
            / params.gravitational_potential_scale
        )
        self._h_phi = data[: lmax + 1, 3] * params.load_scale / params.length_scale
        self._k_phi = (
            data[: lmax + 1, 4]
            * params.load_scale
            / params.gravitational_potential_scale
        )
        self._h = self._h_u + self._h_phi
        self._k = self._k_u + self._k_phi
        self._ht = (
            data[: lmax + 1, 5]
            * params.gravitational_potential_scale
            / params.length_scale
        )
        self._kt = data[: lmax + 1, 6]

    @property
    def h(self) -> np.ndarray:
        """The total displacement Love numbers, h."""
        return self._h

    @property
    def k(self) -> np.ndarray:
        """The total gravitational Love numbers, k."""
        return self._k

    @property
    def ht(self) -> np.ndarray:
        """The tidal displacement Love numbers, h_t."""
        return self._ht

    @property
    def kt(self) -> np.ndarray:
        """The tidal gravitational Love numbers, k_t."""
        return self._kt
