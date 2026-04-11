"""
Abstract base classes for surface load and ice models.

This module defines the required interface for any data provider
aiming to supply background states (ice thickness and topography)
to the pyslfp physical engine.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from pyshtools import SHGrid


class BaseIceModel(ABC):
    """
    Abstract base class for all spatial ice and topography models.

    Any custom model (e.g., ICE-7G, AWIC, synthetic loads) must inherit
    from this class and implement the `get_ice_thickness_and_sea_level` method
    to be fully compatible with the EarthState container.
    """

    @abstractmethod
    def get_ice_thickness_and_sea_level(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> Tuple[SHGrid, SHGrid]:
        """
        Evaluates the model at a specific time and spatial resolution.

        Args:
            date: The time in kiloyears before present (or appropriate model units).
            lmax: The maximum spherical harmonic degree.
            grid: The pyshtools grid format (e.g., "DH", "GLQ").
            sampling: The grid sampling factor (1 for standard, 2 for oversampled).
            extend: Whether to extend the grid to the poles.

        Returns:
            Tuple[SHGrid, SHGrid]: The (ice_thickness, sea_level) grids,
            strictly adhering to the requested spatial properties.
        """
        pass
