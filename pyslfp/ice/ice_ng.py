"""
Module for loading and interpolating the ICE-5G, ICE-6G, and ICE-7G
global ice history models.
"""

from __future__ import annotations
from typing import Tuple, Literal
import bisect

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid

from pyslfp.data import DATADIR, ensure_data
from .ice_models import BaseIceModel


# Define the literal type for valid model versions
IceNGVersion = Literal["ICE5G", "ICE6G", "ICE7G"]


class IceNG(BaseIceModel):
    """
    A data loader for the ICE-5G, ICE-6G, and ICE-7G glacial isostatic
    adjustment models.

    This class retrieves ice thickness, topography, and sea level for a given
    date, linearly interpolating between the model's standard time slices.
    """

    def __init__(
        self, /, *, version: IceNGVersion = "ICE7G", length_scale: float = 1.0
    ) -> None:
        """
        Initializes the ICE-NG data loader.

        Args:
            version (IceNGVersion): The ice model version to use. Defaults to "ICE7G".
            length_scale (float): The scaling factor to non-dimensionalize
                the spatial outputs. Defaults to 1.0 (returns raw meters).
        """
        super().__init__(length_scale=length_scale)
        self._version: IceNGVersion = version

        ensure_data(self._version)

        # Internal densities used exclusively for ice shelf flotation logic
        self._ice_density = 917.0
        self._water_density = 1028.0

    @property
    def version(self) -> IceNGVersion:
        """The specific ICE-NG model version being used."""
        return self._version

    def _date_to_file(self, date: float) -> str:
        """Converts a date into the appropriate data file name."""
        if self._version in ["ICE6G", "ICE7G"]:
            date_string = f"{int(date):d}" if date == int(date) else f"{date:3.1f}"
        else:
            date_string = f"{date:04.1f}"

        if self._version == "ICE7G":
            return str(DATADIR / "ice7g" / f"I7G_NA.VM7_1deg.{date_string}.nc")
        elif self._version == "ICE6G":
            return str(DATADIR / "ice6g" / f"I6_C.VM5a_1deg.{date_string}.nc")
        else:
            return str(DATADIR / "ice5g" / f"ice5g_v1.2_{date_string}k_1deg.nc")

    def _find_files(self, date: float) -> Tuple[str, str, float]:
        """Finds the bounding data files and interpolation fraction for a given date."""
        if self._version in ["ICE6G", "ICE7G"]:
            dates = np.append(np.linspace(0, 21, 43), np.linspace(22, 26, 5))
        else:
            dates = np.append(np.linspace(0, 17, 35), np.linspace(18, 21, 4))

        i = bisect.bisect_left(dates, date)
        if i == 0:
            date1 = date2 = dates[0]
        elif i == len(dates):
            date1 = date2 = dates[i - 1]
        else:
            date1 = dates[i - 1]
            date2 = dates[i]

        fraction = (date2 - date) / (date2 - date1) if date1 != date2 else 0.0

        return self._date_to_file(date1), self._date_to_file(date2), fraction

    def _get_time_slice(
        self, file: str, lmax: int, /, *, grid: str, sampling: int, extend: bool
    ) -> Tuple[SHGrid, SHGrid]:
        """Reads a netCDF file, interpolates fields, and applies non-dimensionalization."""
        data = xr.open_dataset(file)
        ice_thickness = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )
        topography = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )

        if self._version == "ICE5G":
            ice_var, topo_var = "sftgit", "orog"
            lon_var = "long"
        else:
            ice_var, topo_var = "stgit", "Topo"
            lon_var = "lon"

        ice_thickness_function = RegularGridInterpolator(
            (data.lat.values, data[lon_var].values),
            data[ice_var].values,
            bounds_error=False,
            fill_value=None,
        )
        topography_function = RegularGridInterpolator(
            (data.lat.values, data[lon_var].values),
            data[topo_var].values,
            bounds_error=False,
            fill_value=None,
        )

        lats, lons = np.meshgrid(
            ice_thickness.lats(), ice_thickness.lons(), indexing="ij"
        )

        # Interpolate and immediately scale by the length_scale
        ice_thickness.data = ice_thickness_function((lats, lons)) / self.length_scale
        topography.data = topography_function((lats, lons)) / self.length_scale

        return ice_thickness, topography

    def get_ice_thickness_and_topography(
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
        Returns the scaled ice thickness and topography for a given date.
        """
        file1, file2, fraction = self._find_files(date)
        if file1 == file2:
            ice_thickness, topography = self._get_time_slice(
                file1, lmax, grid=grid, sampling=sampling, extend=extend
            )
        else:
            ice_thickness1, topography1 = self._get_time_slice(
                file1, lmax, grid=grid, sampling=sampling, extend=extend
            )
            ice_thickness2, topography2 = self._get_time_slice(
                file2, lmax, grid=grid, sampling=sampling, extend=extend
            )
            ice_thickness = fraction * ice_thickness1 + (1 - fraction) * ice_thickness2
            topography = fraction * topography1 + (1 - fraction) * topography2

        return ice_thickness, topography

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
        Returns the scaled ice thickness and sea level for a given date.
        """
        ice_thickness, topography = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
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
