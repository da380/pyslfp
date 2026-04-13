"""
Analytical surface models for testing the pyslfp physical engine.

Provides perfectly smooth, infinitely differentiable background states
(ice and topography) to avoid spherical harmonic aliasing and Gibbs
ringing during low-resolution test suites.
"""

from typing import Tuple
import numpy as np
from pyshtools import SHGrid

from .ice_models import BaseIceModel


class AnalyticalIceModel(BaseIceModel):
    """
    A synthetic ice model generating smooth, Gaussian-based continents and ice caps.

    Perfect for unit testing the SeaLevelEquation solver at low spherical
    harmonic degrees without triggering noise or convergence issues.
    """

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
        Generates synthetic ice and sea level fields for a given "date".
        (date = 0.0 -> present day, date = 21.0 -> LGM)
        """
        # Create an empty grid just to extract the required lat/lon arrays
        template = SHGrid.from_zeros(lmax, grid=grid, sampling=sampling, extend=extend)
        lats, lons = np.meshgrid(template.lats(), template.lons(), indexing="ij")

        # Convert to radians for the math
        colat_rad = np.deg2rad(90.0 - lats)
        lon_rad = np.deg2rad(lons)

        def gaussian_feature(
            peak_height: float, center_colat: float, center_lon: float, width: float
        ) -> np.ndarray:
            """Helper to generate a smooth, localized spherical bump using great-circle distance."""
            # Spherical law of cosines to find the true angular distance between points
            cos_dist = np.cos(colat_rad) * np.cos(center_colat) + np.sin(
                colat_rad
            ) * np.sin(center_colat) * np.cos(lon_rad - center_lon)

            # Clip to [-1, 1] to avoid floating point errors feeding into arccos
            dist = np.arccos(np.clip(cos_dist, -1.0, 1.0))

            return peak_height * np.exp(-(dist**2) / (2 * width**2))

        # ---------------------------------------------------------
        # 1. Topography / Sea Level
        # ---------------------------------------------------------
        # Base ocean depth of 4000m globally
        raw_sea_level = np.full_like(lats, 4000.0)

        # Continent 1: North Polar landmass (Negative sea level = land)
        # Peak elevation 2000m -> Requires subtracting 6000m from base ocean
        raw_sea_level -= gaussian_feature(
            6000.0, center_colat=0.0, center_lon=0.0, width=0.4
        )

        # Continent 2: low-latitude landmass (e.g., roughly 0N, 90E)
        raw_sea_level -= gaussian_feature(
            5000.0, center_colat=np.pi / 2, center_lon=np.pi / 2, width=0.5
        )

        # ---------------------------------------------------------
        # 2. Ice Thickness
        # ---------------------------------------------------------
        # Base ice cap on the North Polar continent (always present)
        raw_ice = gaussian_feature(3000.0, center_colat=0.0, center_lon=0.0, width=0.2)

        # Time-evolving mid-latitude ice sheet (melts as date approaches 0)
        # Assuming date=21 is LGM and date=0 is present
        if date > 0:
            ice_factor = min(date / 21.0, 1.0)
            raw_ice += gaussian_feature(
                2500.0 * ice_factor,
                center_colat=np.pi / 4,
                center_lon=np.pi / 2,
                width=0.15,
            )

        # Mask the ice with a smooth sigmoid to ensure it only exists on land
        # This prevents sharp gradients (aliasing) at the coastline
        smooth_land_mask = 1.0 / (1.0 + np.exp(raw_sea_level / 100.0))
        raw_ice *= smooth_land_mask

        raw_ice[raw_ice < 1e-3] = 0.0

        # ---------------------------------------------------------
        # 3. Apply scales and return
        # ---------------------------------------------------------
        sea_level_grid = SHGrid.from_array(raw_sea_level / self.length_scale, grid=grid)
        ice_thickness_grid = SHGrid.from_array(raw_ice / self.length_scale, grid=grid)

        return ice_thickness_grid, sea_level_grid
