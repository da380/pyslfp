"""
Abstract base classes for surface mass and ice models.

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

    Any custom model (e.g., ICE-7G, BedMachine, AWIC, synthetic loads) must
    inherit from this class and implement the `get_ice_thickness_and_sea_level`
    method to be fully compatible with the EarthState container.
    """

    def __init__(self, /, *, length_scale: float = 1.0) -> None:
        """
        Initializes the base ice model.

        Args:
            length_scale (float): The scaling factor to non-dimensionalize
                the spatial outputs. Defaults to 1.0 (returns raw meters).
        """
        self._length_scale = length_scale

    @property
    def length_scale(self) -> float:
        """The scaling factor used to non-dimensionalize output grids."""
        return self._length_scale

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
            date (float): The time in kiloyears before present (ka).
            lmax (int): The maximum spherical harmonic degree.
            grid (str): The pyshtools grid format (e.g., "DH", "GLQ").
            sampling (int): The grid sampling factor (1 for standard, 2 for oversampled).
            extend (bool): Whether to extend the grid to include the poles.

        Returns:
            Tuple[SHGrid, SHGrid]: The (ice_thickness, sea_level) grids,
            scaled by the model's `length_scale`.
        """
        pass
