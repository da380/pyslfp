"""
State containers and spatial masking for the pyslfp library.

This module defines the EarthState class, which represents a snapshot of the
Earth's surface (ice thickness and sea level) and handles all geographic
projections, masking, and point-generation logic.
"""

from __future__ import annotations
from functools import cached_property
from typing import Tuple, List

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid

from .core import EarthModel
from .regions import Regions


class EarthState(Regions):
    """
    A pure data container representing a snapshot of the Earth's surface state.

    Handles the lazy evaluation of ocean functions, surface integration,
    and geographic masking. Decoupled entirely from the elastic SLE physics.
    """

    def __init__(
        self,
        ice_thickness: SHGrid,
        sea_level: SHGrid,
        model: EarthModel,
        /,
        *,
        exclude_caspian: bool = True,
    ):
        super().__init__()

        # Validate Grid Compatibility
        if ice_thickness.lmax != sea_level.lmax:
            raise ValueError(
                "Ice thickness and sea level grids must have the same lmax."
            )
        if ice_thickness.grid != sea_level.grid:
            raise ValueError(
                "Ice thickness and sea level grids must use the same grid type."
            )
        if ice_thickness.extend != sea_level.extend:
            raise ValueError(
                "Ice thickness and sea level grids must have matching extend properties."
            )

        self._ice_thickness = ice_thickness
        self._sea_level = sea_level
        self._model = model
        self._exclude_caspian = exclude_caspian

        # Precompute the spherical integration factor
        self._integration_factor = (
            np.sqrt(4 * np.pi) * self._model.parameters.mean_sea_floor_radius**2
        )

    # ---------------------------------------------------------#
    #                 Read-Only Properties                     #
    # ---------------------------------------------------------#

    @property
    def ice_thickness(self) -> SHGrid:
        return self._ice_thickness

    @property
    def sea_level(self) -> SHGrid:
        return self._sea_level

    @property
    def model(self) -> EarthModel:
        return self._model

    @property
    def grid(self) -> str:
        return self._ice_thickness.grid

    @property
    def lmax(self) -> int:
        return self._ice_thickness.lmax

    @property
    def extend(self) -> bool:
        return self._ice_thickness.extend

    @property
    def sampling(self) -> int:
        """Returns the grid sampling factor, defaulting to 1 for grids like GLQ that don't use it."""
        return getattr(self._ice_thickness, "sampling", 1)

    @property
    def grid_type(self) -> str:
        """Returns the specific string expected by pyshtools expand()"""
        return self.grid if self.sampling == 1 else "DH2"

    @property
    def exclude_caspian(self) -> bool:
        """Returns true if the caspian sea is excluded from the oceans"""
        return self._exclude_caspian

    # ---------------------------------------------------------#
    #                 Lazy Evaluated Masks                     #
    # ---------------------------------------------------------#

    @cached_property
    def ocean_function(self) -> SHGrid:
        """Returns the ocean function (1 over oceans, 0 elsewhere). Computed lazily."""
        ocean_data = np.where(
            self.model.parameters.water_density * self.sea_level.data
            - self.model.parameters.ice_density * self.ice_thickness.data
            > 0,
            1,
            0,
        )

        if self.exclude_caspian:
            caspian_mask_data = self.caspian_sea_projection(value=0).data
            ocean_data = np.where(ocean_data - caspian_mask_data > 0, 1, 0)

        return SHGrid.from_array(ocean_data, grid=self.grid)

    @cached_property
    def one_minus_ocean_function(self) -> SHGrid:
        """Returns 1 - C, where C is the ocean function."""
        tmp = self.ocean_function.copy()
        tmp.data = 1 - tmp.data
        return tmp

    @cached_property
    def ocean_area(self) -> float:
        """Returns the total non-dimensional ocean area. Computed lazily."""
        return self.integrate(self.ocean_function)

    # ---------------------------------------------------------#
    #                     Spatial Utilities                    #
    # ---------------------------------------------------------#

    def integrate(self, f: SHGrid) -> float:
        """Integrates an SHGrid function over the surface of the sphere."""
        # Expands to degree 0 to get the mean component
        coeffs = f.expand(lmax_calc=0, normalization="ortho", csphase=1).coeffs
        return float(self._integration_factor * coeffs[0, 0, 0])

    def lats(self) -> np.ndarray:
        return self.ice_thickness.lats()

    def lons(self) -> np.ndarray:
        return self.ice_thickness.lons()

    # ---------------------------------------------------------#
    #                 Geographic Projections                   #
    # ---------------------------------------------------------#

    def ocean_projection(
        self, /, *, value: float = np.nan, exclude_ice_shelves: bool = False
    ) -> SHGrid:
        if exclude_ice_shelves:
            mask = (self.ocean_function.data > 0) & (self.ice_thickness.data == 0)
        else:
            mask = self.ocean_function.data > 0
        return SHGrid.from_array(np.where(mask, 1, value), grid=self.grid)

    def ice_projection(
        self,
        /,
        *,
        value: float = np.nan,
        exclude_ice_shelves: bool = False,
        exclude_glaciers: bool = True,
    ) -> SHGrid:
        if exclude_ice_shelves:
            mask = (self.ice_thickness.data > 0) & (self.ocean_function.data == 0)
        else:
            mask = self.ice_thickness.data > 0

        if exclude_glaciers:
            glacier_mask = self.glacier_projection(value=0).data == 1
            mask = mask & ~glacier_mask

        return SHGrid.from_array(np.where(mask, 1, value), grid=self.grid)

    def land_projection(
        self, /, *, value: float = np.nan, exclude_ice: bool = False
    ) -> SHGrid:
        if exclude_ice:
            mask = (self.ice_thickness.data == 0) & (self.ocean_function.data == 0)
        else:
            mask = self.ocean_function.data == 0
        return SHGrid.from_array(np.where(mask, 1, value), grid=self.grid)

    def northern_hemisphere_projection(self, /, *, value: float = np.nan) -> SHGrid:
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid=self.grid)

    def southern_hemisphere_projection(self, /, *, value: float = np.nan) -> SHGrid:
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid=self.grid)

    def altimetry_projection(
        self,
        /,
        *,
        latitude_min: float = -66,
        latitude_max: float = 66,
        value: float = np.nan,
        ice_threshold: float = 0.0,
    ) -> SHGrid:
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        ocean_mask = self.ocean_function.data > 0
        lat_mask = np.logical_and(lats > latitude_min, lats < latitude_max)
        ice_free_mask = self.ice_thickness.data <= ice_threshold
        combined_mask = ocean_mask & lat_mask & ice_free_mask
        return SHGrid.from_array(np.where(combined_mask, 1, value), grid=self.grid)

    # ---------------------------------------------------------#
    #                 Point Generation Logic                   #
    # ---------------------------------------------------------#

    def _filter_independent_points(
        self, points: np.ndarray, mask: SHGrid
    ) -> List[Tuple[float, float]]:
        grid_lats = mask.lats()
        grid_lons = mask.lons()
        lats_asc = grid_lats[::-1]
        data_asc = mask.data[::-1, :]

        interpolator = RegularGridInterpolator(
            (lats_asc, grid_lons), data_asc, bounds_error=False, fill_value=0.0
        )
        mask_values = interpolator(points)
        valid_points = points[mask_values > 0.5]
        return [(float(lat), float(lon)) for lat, lon in valid_points]

    def ocean_altimetry_points(
        self,
        /,
        *,
        spacing_degrees: float = 2.0,
        latitude_min: float = -66.0,
        latitude_max: float = 66.0,
    ) -> List[Tuple[float, float]]:
        lats = np.arange(latitude_min, latitude_max + 1e-9, spacing_degrees)
        lons = np.arange(0.0, 360.0, spacing_degrees)
        lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
        candidate_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        mask = self.altimetry_projection(
            latitude_min=latitude_min, latitude_max=latitude_max, value=0
        )
        return self._filter_independent_points(candidate_points, mask)

    def ice_altimetry_points(
        self,
        /,
        *,
        spacing_degrees: float = 2.0,
        exclude_ice_shelves: bool = False,
        exclude_glaciers: bool = True,
    ) -> List[Tuple[float, float]]:
        lats = np.arange(-90.0, 90.0 + 1e-9, spacing_degrees)
        lons = np.arange(0.0, 360.0, spacing_degrees)
        lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
        candidate_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))
        mask = self.ice_projection(
            value=0,
            exclude_ice_shelves=exclude_ice_shelves,
            exclude_glaciers=exclude_glaciers,
        )
        return self._filter_independent_points(candidate_points, mask)

    # ---------------------------------------------------------#
    #                 Load Converters                          #
    # ---------------------------------------------------------#

    def direct_load_from_ice_thickness_change(
        self, ice_thickness_change: SHGrid
    ) -> SHGrid:
        """Converts an ice thickness change into the associated surface mass load."""
        return (
            self.model.parameters.ice_density
            * self.one_minus_ocean_function
            * ice_thickness_change
        )

    def direct_load_from_sea_level_change(self, sea_level_change: SHGrid) -> SHGrid:
        """Converts a sea level change into the associated surface mass load."""
        return (
            self.model.parameters.water_density * self.ocean_function * sea_level_change
        )

    def direct_load_from_density_change(self, density_change: SHGrid) -> SHGrid:
        """Converts an ocean density change into the associated surface mass load."""
        return self.sea_level * self.ocean_function * density_change

    # ---------------------------------------------------------#
    #                 Convenience Loads                        #
    # ---------------------------------------------------------#

    def disk_load(
        self, delta: float, latitude: float, longitude: float, amplitude: float
    ) -> SHGrid:
        """Returns a circular disk load."""
        # Note: sampling parameter added to ensure exact grid alignment
        return amplitude * SHGrid.from_cap(
            delta,
            latitude,
            longitude,
            lmax=self.lmax,
            grid=self.grid,
            extend=self.extend,
            sampling=self.sampling,
        )

    def northern_hemisphere_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Northern Hemisphere ice."""
        ice_change = (
            -fraction
            * self.ice_thickness
            * self.northern_hemisphere_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def southern_hemisphere_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Southern Hemisphere ice."""
        ice_change = (
            -fraction
            * self.ice_thickness
            * self.southern_hemisphere_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def greenland_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the Greenland ice sheet."""
        ice_change = -fraction * self.ice_thickness * self.greenland_projection(value=0)
        return self.direct_load_from_ice_thickness_change(ice_change)

    def west_antarctic_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the West Antarctic ice sheet."""
        ice_change = (
            -fraction * self.ice_thickness * self.west_antarctic_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def east_antarctic_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the East Antarctic ice sheet."""
        ice_change = (
            -fraction * self.ice_thickness * self.east_antarctic_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)
