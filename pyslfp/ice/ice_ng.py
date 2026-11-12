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
from pyslfp.core.water_bodies import landlocked_mask
from pyslfp.core.model import ICE_DENSITY, WATER_DENSITY
from .ice_models import BaseIceSeaLevelModel


# Define the literal type for valid model versions
IceNGVersion = Literal["ICE5G", "ICE6G", "ICE7G"]


def standard_dates(version: IceNGVersion) -> np.ndarray:
    """
    The dates of a model's stored time slices, in kyr before present.

    Args:
        version (IceNGVersion): The ice model version.

    Returns:
        np.ndarray: The dates, increasing from the present.
    """
    if version in ["ICE6G", "ICE7G"]:
        return np.append(np.linspace(0, 21, 43), np.linspace(22, 26, 5))
    return np.append(np.linspace(0, 17, 35), np.linspace(18, 21, 4))


def slice_file(version: IceNGVersion, date: float) -> str:
    """
    The data file holding a model's time slice.

    Args:
        version (IceNGVersion): The ice model version.
        date (float): The date in kyr before present, one of
            `standard_dates(version)`.

    Returns:
        str: The file's path under the pyslfp data directory.
    """
    if version in ["ICE6G", "ICE7G"]:
        date_string = f"{int(date):d}" if date == int(date) else f"{date:3.1f}"
    else:
        date_string = f"{date:04.1f}"

    if version == "ICE7G":
        return str(DATADIR / "ice7g" / f"I7G_NA.VM7_1deg.{date_string}.nc")
    elif version == "ICE6G":
        return str(DATADIR / "ice6g" / f"I6_C.VM5a_1deg.{date_string}.nc")
    else:
        return str(DATADIR / "ice5g" / f"ice5g_v1.2_{date_string}k_1deg.nc")


class IceNG(BaseIceSeaLevelModel):
    """
    A data loader for the ICE-5G, ICE-6G, and ICE-7G glacial isostatic
    adjustment models.

    This class retrieves ice thickness, topography, and sea level for a given
    date, linearly interpolating between the model's standard time slices.
    """

    def __init__(
        self,
        /,
        *,
        version: IceNGVersion = "ICE7G",
        length_scale: float = 1.0,
        ice_density: float = ICE_DENSITY,
        water_density: float = WATER_DENSITY,
    ) -> None:
        """
        Initialises the ICE-NG data loader.

        Args:
            version (IceNGVersion): The ice model version to use. Defaults to "ICE7G".
            length_scale (float): The scaling factor that non-dimensionalises
                the spatial outputs. Defaults to 1.0 (returns raw meters).
            ice_density (float): The density of ice. Only its ratio to
                `water_density` is used, so any common unit will do.
            water_density (float): The density of water. The two set the
                fraction of a floating ice shelf's thickness that lies below
                the sea surface, by which the sea level is built from the
                topography. They must be the densities of the Earth model
                the sea level is used with, whose ocean function tests
                flotation with the same ratio; the defaults are the default
                model's (`pyslfp.core.model.ICE_DENSITY`, `WATER_DENSITY`).

        Raises:
            ValueError: If either density is not positive.
        """
        super().__init__(length_scale=length_scale)
        self._version: IceNGVersion = version

        if not (ice_density > 0 and water_density > 0):
            raise ValueError("ice_density and water_density must be positive")

        ensure_data(self._version)

        self._ice_density = float(ice_density)
        self._water_density = float(water_density)

    @property
    def flotation_ratio(self) -> float:
        """The ratio of the ice density to the water density with which
        the sea level is built under floating ice."""
        return self._ice_density / self._water_density

    @property
    def version(self) -> IceNGVersion:
        """The specific ICE-NG model version being used."""
        return self._version

    def _date_to_file(self, date: float) -> str:
        """Converts a date into the appropriate data file name."""
        return slice_file(self._version, date)

    def _find_files(self, date: float) -> Tuple[str, str, float]:
        """Finds the bounding data files and interpolation fraction for a given date."""
        dates = standard_dates(self._version)

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
        grid, sampling = self.resolve_grid(grid, sampling)
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

        with xr.open_dataset(file) as data:
            lat_values = data.lat.values
            lon_values = data[lon_var].values
            ice_values = data[ice_var].values
            topo_values = data[topo_var].values

        ice_thickness_function = RegularGridInterpolator(
            (lat_values, lon_values),
            ice_values,
            bounds_error=False,
            fill_value=None,
        )
        topography_function = RegularGridInterpolator(
            (lat_values, lon_values),
            topo_values,
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
        """The scaled ice thickness and topography at a date; arguments as
        `BaseIceModel.get_ice_thickness_and_sea_level`."""
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

    def get_land(
        self,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """The present-day landlocked depressions (`pyslfp.core.water_bodies`),
        held as land at every date: ICE-NG describes the Earth, where
        flotation alone would extend the ocean into the Caspian and the other
        basins below sea level that are cut off from it. Arguments as
        `BaseIceSeaLevelModel.get_land`."""
        grid, sampling = self.resolve_grid(grid, sampling)
        template = SHGrid.from_zeros(lmax, grid=grid, sampling=sampling, extend=extend)
        mask = landlocked_mask(template.lats(), template.lons(), extend=extend)
        return SHGrid.from_array(mask.astype(float), grid=grid)

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
        """The scaled ice thickness and sea level at a date, interpolated
        linearly between the model's slices; arguments as
        `BaseIceModel.get_ice_thickness_and_sea_level`."""
        grid, sampling = self.resolve_grid(grid, sampling)
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

        sea_level = sea_level + (ice_shelf_thickness * self.flotation_ratio)

        return ice_thickness, sea_level
