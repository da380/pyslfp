"""
Module for loading and interpolating the ICE-5G, ICE-6G, and ICE-7G
global ice history models.
"""

from __future__ import annotations
from typing import Tuple
import xarray as xr
from pyshtools import SHGrid
import numpy as np
import bisect
from scipy.interpolate import RegularGridInterpolator
from enum import Enum

from . import DATADIR


class IceModel(Enum):
    """An enumeration for the different ICE-NG model versions."""

    ICE5G = 5
    ICE6G = 6
    ICE7G = 7


class IceNG:
    """
    A data loader for the ICE-5G, ICE-6G, and ICE-7G glacial isostatic
    adjustment models.

    This class provides methods to retrieve ice thickness, topography, and
    sea level for a given date, interpolating between the model's time
    slices as needed.
    """

    def __init__(self, /, *, version: IceModel = IceModel.ICE7G):
        """
        Args:
            version: The ice model version to use. Defaults to ICE-7G.
        """
        self._version = version
        # Set the densities of ice and water in kg/m^3.
        self.ice_density = 917.0
        self.water_density = 1028.0

    def _date_to_file(self, date: float) -> str:
        """Converts a date into the appropriate data file name."""
        if self._version in [IceModel.ICE6G, IceModel.ICE7G]:
            date_string = f"{int(date):d}" if date == int(date) else f"{date:3.1f}"
        else:
            date_string = f"{date:04.1f}"

        if self._version == IceModel.ICE7G:
            return DATADIR + "/ice7g/I7G_NA.VM7_1deg." + date_string + ".nc"
        elif self._version == IceModel.ICE6G:
            return DATADIR + "/ice6g/I6_C.VM5a_1deg." + date_string + ".nc"
        else:
            return DATADIR + "/ice5g/ice5g_v1.2_" + date_string + "k_1deg.nc"

    def _find_files(self, date: float) -> Tuple[str, str, float]:
        """
        Given a date, finds the data files that bound it and the fraction
        for linear interpolation.
        """
        if self._version in [IceModel.ICE6G, IceModel.ICE7G]:
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
        """Reads a data file and interpolates fields onto a pyshtools grid."""
        data = xr.open_dataset(file)
        ice_thickness = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )
        topography = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )

        if self._version == IceModel.ICE5G:
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
        ice_thickness.data = ice_thickness_function((lats, lons))
        topography.data = topography_function((lats, lons))

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
        Returns the ice thickness and topography (in meters) for a given date.

        If the date does not exist within the data set, linear interpolation is
        used. If the date is out of range, constant extrapolation is applied.

        Args:
            date: The date in thousands of years before present (ka).
            lmax: The maximum spherical harmonic degree for the output grids.
            grid: The `pyshtools` grid type. Defaults to 'DH'.
            sampling: The `pyshtools` grid sampling. Defaults to 1.
            extend: `pyshtools` grid extension option. Defaults to True.

        Returns:
            A tuple containing the ice thickness and topography as `SHGrid` objects.
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
        Returns the ice thickness and sea level (in meters) for a given date.

        Sea level is computed from topography assuming isostatic balance for
        floating ice shelves.

        Args:
            date: The date in thousands of years before present (ka).
            lmax: The maximum spherical harmonic degree for the output grids.
            grid: The `pyshtools` grid type. Defaults to 'DH'.
            sampling: The `pyshtools` grid sampling. Defaults to 1.
            extend: `pyshtools` grid extension option. Defaults to True.

        Returns:
            A tuple containing the ice thickness and sea level as `SHGrid` objects.
        """
        ice_thickness, topography = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        # Compute the sea level using isostatic balance within ice shelves.
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
        sea_level += self.ice_density * ice_shelf_thickness / self.water_density
        return ice_thickness, sea_level

    def get_ice_thickness(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the ice thickness (in meters) for a given date."""
        ice_thickness, _ = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return ice_thickness

    def get_sea_level(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the sea level (in meters) for a given date."""
        _, sea_level = self.get_ice_thickness_and_sea_level(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return sea_level

    def get_topography(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the topography (in meters) for a given date."""
        _, topography = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return topography
