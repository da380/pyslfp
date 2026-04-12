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
from pyshtools import SHGrid, SHCoeffs

from .core import EarthModel
from .regions import Regions


class EarthState(Regions):
    """
    A pure data container representing a snapshot of the Earth's surface state.

    Handles the lazy evaluation of ocean functions, surface integration,
    and geographic masking. It is decoupled entirely from the elastic SLE
    physics to ensure a strict one-way data flow.
    """

    def __init__(
        self,
        ice_thickness: SHGrid,
        sea_level: SHGrid,
        model: EarthModel,
        /,
        *,
        exclude_caspian: bool = True,
    ) -> None:
        """
        Initializes an EarthState container.

        Args:
            ice_thickness (SHGrid): The non-dimensional ice thickness grid.
            sea_level (SHGrid): The non-dimensional sea level (topography) grid.
            model (EarthModel): The physical Earth model parameters and Love numbers.
            exclude_caspian (bool): If True, the Caspian Sea is strictly treated as
                land in all flotation and ocean function calculations. Defaults to True.

        Raises:
            ValueError: If the input grids do not match in lmax, grid type, or extension.
        """
        super().__init__()

        # Check data compatibility
        model.check_field(sea_level)
        model.check_field(ice_thickness)

        self._ice_thickness = ice_thickness
        self._sea_level = sea_level
        self._model = model
        self._exclude_caspian = exclude_caspian

    # ---------------------------------------------------------#
    #                 Read-Only Properties                     #
    # ---------------------------------------------------------#

    @property
    def ice_thickness(self) -> SHGrid:
        """The non-dimensional ice thickness grid."""
        return self._ice_thickness

    @property
    def sea_level(self) -> SHGrid:
        """The non-dimensional sea level (bathymetry/topography) grid."""
        return self._sea_level

    @property
    def model(self) -> EarthModel:
        """The physical Earth model configuration."""
        return self._model

    @property
    def exclude_caspian(self) -> bool:
        """Returns true if the Caspian Sea is excluded from the global ocean."""
        return self._exclude_caspian

    @property
    def grid(self) -> str:
        """Return spatial grid option."""
        return self.model.grid

    @property
    def lmax(self) -> int:
        """The maximum spherical harmonic degree."""
        return self.model.lmax

    # ---------------------------------------------------------#
    #                 Lazy Evaluated Masks                     #
    # ---------------------------------------------------------#

    @cached_property
    def ocean_function(self) -> SHGrid:
        """
        Returns the ocean function (1 over oceans, 0 elsewhere).
        Computed lazily using the state's internal flotation conditions.
        """
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
        return self.model.integrate(self.ocean_function)

    # ---------------------------------------------------------#
    #                     Spatial Utilities                    #
    # ---------------------------------------------------------#

    def lats(self) -> np.ndarray:
        """Returns the 1D array of grid latitudes."""
        return self.ice_thickness.lats()

    def lons(self) -> np.ndarray:
        """Returns the 1D array of grid longitudes."""
        return self.ice_thickness.lons()

    # ---------------------------------------------------------#
    #                 Geographic Projections                   #
    # ---------------------------------------------------------#

    def ocean_projection(
        self, /, *, value: float = np.nan, exclude_ice_shelves: bool = False
    ) -> SHGrid:
        """
        Returns a grid that is 1 over oceans and `value` elsewhere.

        Args:
            value (float): The value to assign outside the ocean. Defaults to NaN.
            exclude_ice_shelves (bool): If True, exclude floating ice shelves.
        """
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
        """
        Returns a grid that is 1 over ice sheets and `value` elsewhere.

        Args:
            value (float): The value to assign outside the ice sheet. Defaults to NaN.
            exclude_ice_shelves (bool): If True, exclude ice shelves.
            exclude_glaciers (bool): If True, exclude mountain glaciers.
        """
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
        """
        Returns a grid that is 1 over land and `value` elsewhere.

        Args:
            value (float): The value to assign outside the land. Defaults to NaN.
            exclude_ice (bool): If True, exclude ice-covered areas.
        """
        if exclude_ice:
            mask = (self.ice_thickness.data == 0) & (self.ocean_function.data == 0)
        else:
            mask = self.ocean_function.data == 0
        return SHGrid.from_array(np.where(mask, 1, value), grid=self.grid)

    def northern_hemisphere_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 in the Northern Hemisphere and `value` elsewhere."""
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid=self.grid)

    def southern_hemisphere_projection(self, /, *, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 in the Southern Hemisphere and `value` elsewhere."""
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid=self.grid)

    def altimetry_projection(
        self,
        /,
        *,
        latitude_min: float = -66.0,
        latitude_max: float = 66.0,
        value: float = np.nan,
        ice_threshold: float = 0.0,
    ) -> SHGrid:
        """
        Returns a grid that is 1 in the oceans between specified latitudes.
        Typically used to mask areas observed by satellite altimetry.

        Args:
            latitude_min (float): Southern boundary in degrees.
            latitude_max (float): Northern boundary in degrees.
            value (float): Value to assign outside the projection.
            ice_threshold (float): Maximum ice thickness to still be considered ocean.
        """
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
        """
        Generates a regular grid of points returning only those in the valid oceans.

        Args:
            spacing_degrees (float): Spatial resolution of the points.
            latitude_min (float): Southern boundary.
            latitude_max (float): Northern boundary.
        """
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
        """
        Generates a regular grid of points returning only those over ice sheets.

        Args:
            spacing_degrees (float): Spatial resolution of the points.
            exclude_ice_shelves (bool): Exclude floating ice shelves.
            exclude_glaciers (bool): Exclude smaller mountain glaciers.
        """
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
        self, ice_thickness_change: SHGrid, /
    ) -> SHGrid:
        """Converts an ice thickness change into the associated surface mass load."""
        return (
            self.model.parameters.ice_density
            * self.one_minus_ocean_function
            * ice_thickness_change
        )

    def direct_load_from_sea_level_change(self, sea_level_change: SHGrid, /) -> SHGrid:
        """Converts a sea level change into the associated surface mass load."""
        return (
            self.model.parameters.water_density * self.ocean_function * sea_level_change
        )

    def direct_load_from_density_change(self, density_change: SHGrid, /) -> SHGrid:
        """Converts an ocean density change into the associated surface mass load."""
        return self.sea_level * self.ocean_function * density_change

    # ---------------------------------------------------------#
    #                 Convenience Loads                        #
    # ---------------------------------------------------------#

    def disk_load(
        self, delta: float, latitude: float, longitude: float, amplitude: float, /
    ) -> SHGrid:
        """
        Returns a circular disk load.

        Args:
            delta (float): Radius of the disk in degrees.
            latitude (float): Latitude of the disk center in degrees.
            longitude (float): Longitude of the disk center in degrees.
            amplitude (float): Amplitude of the load.
        """
        return amplitude * SHGrid.from_cap(
            delta,
            latitude,
            longitude,
            lmax=self.lmax,
            grid=self.grid,
            extend=self.extend,
            sampling=self.sampling,
        )

    def northern_hemisphere_load(self, /, *, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Northern Hemisphere ice."""
        ice_change = (
            -fraction
            * self.ice_thickness
            * self.northern_hemisphere_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def southern_hemisphere_load(self, /, *, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Southern Hemisphere ice."""
        ice_change = (
            -fraction
            * self.ice_thickness
            * self.southern_hemisphere_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def greenland_load(self, /, *, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the Greenland ice sheet."""
        ice_change = -fraction * self.ice_thickness * self.greenland_projection(value=0)
        return self.direct_load_from_ice_thickness_change(ice_change)

    def west_antarctic_load(self, /, *, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the West Antarctic ice sheet."""
        ice_change = (
            -fraction * self.ice_thickness * self.west_antarctic_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def east_antarctic_load(self, /, *, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of the East Antarctic ice sheet."""
        ice_change = (
            -fraction * self.ice_thickness * self.east_antarctic_projection(value=0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    # ---------------------------------------------------------#
    #                 Default Factory                          #
    # ---------------------------------------------------------#

    @classmethod
    def default(cls, lmax: int, /, *, exclude_caspian: bool = True) -> "EarthState":
        """
        Generates a standard Present-Day EarthState.

        Constructs an Earth model using the default PREM parameterization
        and loads present-day (0.0 ka) ice thickness and topography from ICE-7G.

        Args:
            lmax (int): Maximum spherical harmonic degree to build grids.
            exclude_caspian (bool): Whether to enforce the Caspian Sea mask.

        Returns:
            EarthState: A configured, Present-Day Earth snapshot.
        """
        from .core import EarthModel
        from .ice_ng import IceNG, IceModel

        # Initialize the standard physics model
        model = EarthModel.default(lmax)

        # Initialize the ICE-7G data provider, passing the required length scale
        ice_engine = IceNG(
            version=IceModel.ICE7G, length_scale=model.parameters.length_scale
        )

        # Fetch present day (0.0 ka) data natively fitted to the model scales
        ice_thickness, sea_level = ice_engine.get_ice_thickness_and_sea_level(
            0.0, lmax, grid="DH", sampling=1, extend=True
        )

        return cls(ice_thickness, sea_level, model, exclude_caspian=exclude_caspian)
