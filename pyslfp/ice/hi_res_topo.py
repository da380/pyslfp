"""
Module for loading high-resolution bathymetry/topography models combined 
with historical ice coverage from ICE-NG models, featuring spectral smoothing.
"""

from __future__ import annotations
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid

from .ice_models import BaseIceModel, apply_cosine_taper
from .ice_ng import IceNG, IceNGVersion


class HighResCompositeIceModel(BaseIceModel):
    """
    A composite ice model that pairs high-resolution static topography 
    with historical ice thickness data from the ICE-NG models.

    To avoid the Gibbs phenomenon (ringing), this class internally oversamples
    the topography to a higher maximum degree, then applies a smooth cosine taper 
    down to the requested target resolution.
    """

    def __init__(
        self,
        topo_file: Union[str, Path],
        /,
        *,
        version: IceNGVersion = "ICE7G",
        topo_var: str = "elevation",
        lon_var: str = "lon",
        lat_var: str = "lat",
        topo_includes_ice: bool = True,
        length_scale: float = 1.0,
        oversample_buffer: int = 64,
        taper_ratio: float = 0.9,
    ) -> None:
        """
        Initializes the composite high-resolution model.

        Args:
            topo_file (str | Path): Path to the high-res NetCDF topography file.
            version (IceNGVersion): The ICE-NG model version to use.
            topo_var (str): The variable name for topography in the NetCDF file.
            lon_var (str): The variable name for longitude in the NetCDF file.
            lat_var (str): The variable name for latitude in the NetCDF file.
            topo_includes_ice (bool): If True, subtracts present-day (0 ka) ICE-NG 
                ice thickness from the topography to isolate the bedrock.
            length_scale (float): Scaling factor to non-dimensionalize outputs.
            oversample_buffer (int): The number of extra spherical harmonic degrees 
                to use internally before tapering. (e.g., if lmax=256, it builds 
                the grid at 320 to capture sharp features, then tapers).
            taper_ratio (float): The fraction of target_lmax where the smooth 
                cosine taper begins (e.g., 0.9 means taper starts at 90% of lmax).
        """
        super().__init__(length_scale=length_scale)
        self._version = version
        self._topo_includes_ice = topo_includes_ice
        self._oversample_buffer = oversample_buffer
        self._taper_ratio = taper_ratio

        self._ice_provider = IceNG(version=version, length_scale=length_scale)

        self._topo_file = Path(topo_file)
        if not self._topo_file.exists():
            raise FileNotFoundError(f"Topography data file not found: {self._topo_file}")

        print(f"Loading high-resolution topography from {self._topo_file.name}...")
        data = xr.open_dataset(self._topo_file)
        
        self._topo_interpolator = RegularGridInterpolator(
            (data[lat_var].values, data[lon_var].values),
            data[topo_var].values,
            bounds_error=False,
            fill_value=None,
        )

        self._ice_density = 917.0
        self._water_density = 1028.0

    @property
    def version(self) -> IceNGVersion:
        return self._version

    def get_ice_thickness_and_topography(
        self,
        date: float,
        target_lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> Tuple[SHGrid, SHGrid]:
        """
        Returns the spectrally-smoothed, scaled ice thickness and topography.
        """
        # Calculate working resolutions for anti-aliasing
        working_lmax = target_lmax + self._oversample_buffer
        l_start = int(target_lmax * self._taper_ratio)

        # 1. Fetch historical ice thickness on the *oversampled* grid
        ice_high_res = self._ice_provider.get_ice_thickness(
            date, working_lmax, grid=grid, sampling=sampling, extend=extend
        )

        # 2. Prepare the empty oversampled topography grid
        topo_high_res = SHGrid.from_zeros(
            working_lmax, grid=grid, sampling=sampling, extend=extend
        )
        lats, lons = np.meshgrid(topo_high_res.lats(), topo_high_res.lons(), indexing="ij")

        # 3. Interpolate the custom high-res static topography
        topo_high_res.data = self._topo_interpolator((lats, lons)) / self.length_scale

        # 4. Correct for modern-day ice if necessary
        if self._topo_includes_ice:
            present_day_ice = self._ice_provider.get_ice_thickness(
                0.0, working_lmax, grid=grid, sampling=sampling, extend=extend
            )
            topo_high_res.data = topo_high_res.data - present_day_ice.data

        # 5. Apply the Cosine Taper to safely bring both fields down to target_lmax
        # This converts to SH domain, cuts off high frequencies smoothly, and reconstructs.
        final_ice = apply_cosine_taper(
            ice_high_res, target_lmax, l_start, grid_format=grid, extend=extend
        )
        final_topo = apply_cosine_taper(
            topo_high_res, target_lmax, l_start, grid_format=grid, extend=extend
        )

        return final_ice, final_topo

    def get_ice_thickness_and_sea_level(
        self,
        date: float,
        target_lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> Tuple[SHGrid, SHGrid]:
        """
        Returns the scaled ice thickness and sea level for a given date,
        calculating flotation using the smoothed topography.
        """
        ice_thickness, topography = self.get_ice_thickness_and_topography(
            date, target_lmax, grid=grid, sampling=sampling, extend=extend
        )

        ice_shelf_thickness = SHGrid.from_array(
            np.where(
                np.logical_and(topography.data < 0, ice_thickness.data > 0),
                ice_thickness.data,
                0,
            ),
            grid=grid,
        )
        
        sea_level = SHGrid.from_array(
            np.where(
                topography.data < 0,
                -topography.data,
                -topography.data + ice_thickness.data,
            ),
            grid=grid,
        )

        flotation_ratio = self._ice_density / self._water_density
        sea_level = sea_level + (ice_shelf_thickness * flotation_ratio)

        return ice_thickness, sea_level