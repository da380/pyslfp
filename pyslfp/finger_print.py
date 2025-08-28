"""
Module for the FingerPrint class that allows for forward
and adjoint elastic fingerprint calculations.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union

import inspect

import numpy as np

import pyshtools as pysh
from pyshtools import SHGrid, SHCoeffs

from pygeoinf import (
    LinearOperator,
    HilbertSpaceDirectSum,
    EuclideanSpace,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from .ice_ng import IceNG, IceModel
from .physical_parameters import EarthModelParameters
from .love_numbers import LoveNumbers


from . import DATADIR


class FingerPrint(EarthModelParameters, LoveNumbers):
    """
    Computes elastic sea level "fingerprints" for a given surface load.

    A sea level fingerprint is the unique pattern of sea level change that
    results from a change in a surface load, such as the melting of an ice
    sheet. This class solves the sea level equation, accounting for Earth's
    elastic response, gravitational self-consistency, and rotational feedbacks.

    The background state (sea level and ice thickness) must be set via a
    method like `set_state_from_ice_ng` before calculations can be performed.
    """

    def __init__(
        self,
        /,
        *,
        lmax: int = 256,
        earth_model_parameters: Optional[EarthModelParameters] = None,
        grid: str = "DH",
        love_number_file: str = DATADIR + "/love_numbers/PREM_4096.dat",
    ) -> None:
        """
        Args:
            lmax: The truncation degree for spherical harmonic expansions.
            earth_model_parameters: Parameters for the Earth model. If None,
                default parameters are used.
            grid: The pyshtools grid type ('DH' or 'GLQ'). Defaults to 'DH'.
            extend: If True, the spatial grid is extended to include 360
                degrees longitude. Defaults to True.
            love_number_file: Path to the file containing the Love numbers.
        """
        # Set up the earth model parameters
        if earth_model_parameters is None:
            super().__init__()
        else:
            init_kwargs = EarthModelParameters._get_init_kwargs_from_instance(
                earth_model_parameters
            )
            super().__init__(**init_kwargs)

        # Set options.
        self._lmax: int = lmax

        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1

        self._love_number_file = love_number_file

        LoveNumbers.__init__(
            self,
            self.lmax,
            self,
            file=self._love_number_file,
        )

        # Internal parameters (do not change)
        self._extend: bool = True
        self._normalization: str = "ortho"
        self._csphase: int = 1

        self._normalization: str = "ortho"
        self._csphase: int = 1
        self._integration_factor: float = (
            np.sqrt((4 * np.pi)) * self._mean_sea_floor_radius**2
        )
        self._rotation_factor: float = (
            np.sqrt((4 * np.pi) / 15.0)
            * self.rotation_frequency
            * self.mean_sea_floor_radius**2
        )
        self._inertia_factor: float = (
            np.sqrt(5 / (12 * np.pi))
            * self.rotation_frequency
            * self.mean_sea_floor_radius**3
            / (
                self.gravitational_constant
                * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            )
        )

        # Background model not set.
        self._sea_level: Optional[SHGrid] = None
        self._ice_thickness: Optional[SHGrid] = None
        self._ocean_function: Optional[SHGrid] = None
        self._ocean_area: Optional[float] = None

        # Initialise the counter for number of solver calls.
        self._solver_counter: int = 0

    # -----------------------------------------------#
    #                    Properties                  #
    # -----------------------------------------------#

    @property
    def lmax(self) -> int:
        """Return truncation degree for expansions."""
        return self._lmax

    @property
    def normalization(self) -> str:
        """Return spherical harmonic normalisation convention."""
        return self._normalization

    @property
    def csphase(self) -> int:
        """Return Condon-Shortley phase option."""
        return self._csphase

    @property
    def grid(self) -> str:
        """Return spatial grid option."""
        return self._grid

    @property
    def extend(self) -> bool:
        """True if grid extended to include 360 degree longitude."""
        return self._extend

    @property
    def background_set(self) -> bool:
        """Returns true is background state has been set."""
        return self._sea_level is not None and self._ice_thickness is not None

    @property
    def sea_level(self) -> SHGrid:
        """Returns the background sea level grid."""
        if self._sea_level is None:
            raise AttributeError("Sea level has not been set.")
        return self._sea_level

    @sea_level.setter
    def sea_level(self, value: SHGrid) -> None:
        self.check_field(value)
        self._sea_level = value
        self._ocean_function = None

    @property
    def ice_thickness(self) -> SHGrid:
        """Returns the background ice thickness grid."""
        if self._ice_thickness is None:
            raise AttributeError("Ice thickness has not been set.")
        return self._ice_thickness

    @ice_thickness.setter
    def ice_thickness(self, value: SHGrid) -> None:
        self.check_field(value)
        self._ice_thickness = value
        self._ocean_function = None

    @property
    def ocean_function(self) -> SHGrid:
        """Returns the ocean function (1 over oceans, 0 elsewhere)."""
        if self._ocean_function is None:
            self._compute_ocean_function()
        return self._ocean_function

    @property
    def one_minus_ocean_function(self) -> SHGrid:
        """Returns 1 - C, where C is the ocean function."""
        tmp = self.ocean_function.copy()
        tmp.data = 1 - tmp.data
        return tmp

    @property
    def ocean_area(self) -> float:
        """Returns the total ocean area."""
        if self._ocean_area is None:
            self._compute_ocean_area()
        return self._ocean_area

    @property
    def love_number_file(self):
        """Returns the path to the Love number file."""
        return self._love_number_file

    @property
    def solver_counter(self) -> int:
        """Returns the number of times the __call__ solver method has been called."""
        return self._solver_counter

    # ---------------------------------------------------------#
    #                     Private methods                      #
    # ---------------------------------------------------------#

    def _grid_name(self):
        return self.grid if self._sampling == 1 else "DH2"

    def _compute_ocean_function(self) -> None:
        """Computes and stores the ocean function from sea level and ice."""
        if self._sea_level is None or self._ice_thickness is None:
            raise AttributeError(
                "Sea level and ice thickness must be set before computing ocean function."
            )
        self._ocean_function = SHGrid.from_array(
            np.where(
                self.water_density * self.sea_level.data
                - self.ice_density * self.ice_thickness.data
                > 0,
                1,
                0,
            ),
            grid=self.grid,
        )

    def _compute_ocean_area(self) -> None:
        """Computes and stores the ocean area."""
        if self._ocean_function is None:
            self._compute_ocean_function()
        self._ocean_area = self.integrate(self._ocean_function)

    # --------------------------------------------------------#
    #                       Public methods                    #
    # --------------------------------------------------------#

    def lats(self) -> np.ndarray:
        """Return the latitudes for the spatial grid."""
        return self.zero_grid().lats()

    def lons(self) -> np.ndarray:
        """Return the longitudes for the spatial grid."""
        return self.zero_grid().lons()

    def check_field(self, f: SHGrid) -> bool:
        """Checks if an SHGrid object is compatible with instance settings."""
        is_compatible = (
            f.lmax == self.lmax and f.grid == self.grid and f.extend == self.extend
        )
        if not is_compatible:
            raise ValueError(
                "Provided SHGrid object is not compatible with FingerPrint settings."
            )
        return True

    def check_coefficient(self, f: SHCoeffs) -> bool:
        """Checks if an SHCoeffs object is compatible with instance settings."""
        is_compatible = (
            f.lmax == self.lmax
            and f.normalization == self.normalization
            and f.csphase == self.csphase
        )
        if not is_compatible:
            raise ValueError(
                "Provided SHCoeffs object is not compatible with FingerPrint settings."
            )
        return True

    def expand_field(
        self, f: SHGrid, /, *, lmax_calc: Optional[int] = None
    ) -> SHCoeffs:
        """Expands an SHGrid object into spherical harmonic coefficients."""
        self.check_field(f)
        return f.expand(
            lmax_calc=lmax_calc, normalization=self.normalization, csphase=self.csphase
        )

    def expand_coefficient(self, f: SHCoeffs) -> SHGrid:
        """Expands spherical harmonic coefficients into an SHGrid object."""
        self.check_coefficient(f)
        grid = "DH2" if self._sampling == 2 else self.grid
        return f.expand(grid=grid, extend=self.extend)

    def load_response(self, load: SHGrid) -> Tuple[SHGrid, SHGrid]:
        """
        Returns the direct elastic deformation for a load.

        Note:
            This calculation does not account for the sea level equation (i.e., no
            gravitationally self-consistent sea level change or rotational feedbacks).

        Args:
            load: The applied surface load.

        Returns:
            A tuple containing the vertical displacement and the gravitational
            potential change as `SHGrid` objects.
        """
        self.check_field(load)
        displacement_lm = self.expand_field(load)
        gravitational_potential_change_lm = displacement_lm.copy()

        for l in range(0, self.lmax + 1):
            displacement_lm.coeffs[:, l, :] *= self._h[l]
            gravitational_potential_change_lm.coeffs[:, l, :] *= self._k[l]

        displacement = self.expand_coefficient(displacement_lm)
        gravitational_potential_change = self.expand_coefficient(
            gravitational_potential_change_lm
        )
        return displacement, gravitational_potential_change

    def integrate(self, f: SHGrid) -> float:
        """
        Integrate a function over the surface of the sphere.

        Args:
            f: The function to integrate, represented as an SHGrid object.

        Returns:
            The integral of the function over the surface.
        """
        return (
            self._integration_factor * self.expand_field(f, lmax_calc=0).coeffs[0, 0, 0]
        )

    def point_evaluation(
        self, f: SHGrid, latitude: float, longitude: float, degrees: bool = True
    ) -> float:
        """
        Evaluate a grid function at a specific point.

        Args:
            f: The function to evaluate.
            latitude: The latitude of the point.
            longitude: The longitude of the point.
            degrees: If True, latitude and longitude are in degrees. Defaults to True.

        Returns:
            The value of the function at the specified point.
        """
        f_lm = self.expand_field(f)
        return f_lm.expand(lat=[latitude], lon=[longitude], degrees=degrees)[0]

    def coefficient_evaluation(self, f: SHGrid, l: int, m: int) -> float:
        """
        Return the (l,m)-th spherical harmonic coefficient of a function.

        Args:
            f: The function to get the coefficient from.
            l: The degree.
            m: The order.

        Returns:
            The value of the (l,m) coefficient.
        """
        if not (0 <= l <= self.lmax and -l <= m <= l):
            raise ValueError(f"(l,m)=({l},{m}) is out of bounds for lmax={self.lmax}.")
        f_lm = self.expand_field(f)
        return f_lm.coeffs[0 if m >= 0 else 1, l, abs(m)]

    def zero_grid(self) -> SHGrid:
        """Return a grid of zeros with compatible dimensions."""
        return SHGrid.from_zeros(
            lmax=self.lmax, grid=self.grid, sampling=self._sampling, extend=self.extend
        )

    def constant_grid(self, value: float) -> SHGrid:
        """Return a grid of a constant value."""
        f = self.zero_grid()
        f.data[:, :] = value
        return f

    def zero_coefficients(self) -> SHCoeffs:
        """Return a set of zero spherical harmonic coefficients."""
        return SHCoeffs.from_zeros(
            lmax=self.lmax, normalization=self.normalization, csphase=self.csphase
        )

    def ocean_average(self, f: SHGrid) -> float:
        """Return the average of a function over the oceans."""
        return self.integrate(self.ocean_function * f) / self.ocean_area

    def set_state_from_ice_ng(
        self, /, *, version: IceModel = IceModel.ICE7G, date: float = 0.0
    ) -> None:
        """
        Sets the background state from an ICE-NG model.

        Args:
            version: The ice model version to use (e.g., `IceModel.ICE7G`).
            date: The date in thousands of years before present (ka).
        """
        ice_ng = IceNG(version=version)
        ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(
            date, self.lmax, grid=self.grid, sampling=self._sampling, extend=self.extend
        )
        self.ice_thickness = ice_thickness / self.length_scale
        self.sea_level = sea_level / self.length_scale

    def mean_sea_level_change(self, direct_load: SHGrid) -> float:
        """
        Returns the mean sea level change from a direct load (eustatic change).
        """
        self.check_field(direct_load)
        return -self.integrate(direct_load) / (self.water_density * self.ocean_area)

    def __call__(
        self,
        /,
        *,
        direct_load: Optional[SHGrid] = None,
        displacement_load: Optional[SHGrid] = None,
        gravitational_potential_load: Optional[SHGrid] = None,
        angular_momentum_change: Optional[np.ndarray] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the generalized sea level equation for a given load.

        Args:
            direct_load: The direct surface mass load (e.g., from ice melt).
            displacement_load: An externally imposed displacement load.
            gravitational_potential_load: An externally imposed gravitational potential load.
            angular_momentum_change: An externally imposed change in angular momentum.
            rotational_feedbacks: If True, include the effects of polar wander.
            rtol: The relative tolerance for the iterative solver to determine convergence.
            verbose: If True, print the relative error at each iteration.

        Returns:
            A tuple containing:
                - `sea_level_change` (SHGrid): The self-consistent sea level change.
                - `displacement` (SHGrid): The vertical surface displacement.
                - `gravity_potential_change` (SHGrid): Change in gravity potential.
                - `angular_velocity_change` (np.ndarray): Change in angular velocity `[ω_x, ω_y]`.
        """
        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            loads_present = True
            assert self.check_field(direct_load)
            mean_sea_level_change = -self.integrate(direct_load) / (
                self.water_density * self.ocean_area
            )
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0

        else:
            direct_load = self.zero_grid()
            mean_sea_level_change = 0

        if displacement_load is not None:
            loads_present = True
            assert self.check_field(displacement_load)
            displacement_load_lm = self.expand_field(displacement_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(displacement_load.data)) > 0

        if gravitational_potential_load is not None:
            loads_present = True
            assert self.check_field(gravitational_potential_load)
            gravitational_potential_load_lm = self.expand_field(
                gravitational_potential_load
            )
            non_zero_rhs = (
                non_zero_rhs or np.max(np.abs(gravitational_potential_load.data)) > 0
            )

        if angular_momentum_change is not None:
            loads_present = True
            non_zero_rhs = non_zero_rhs or np.max(np.abs(angular_momentum_change)) > 0

        if loads_present is False or not non_zero_rhs:
            return self.zero_grid(), self.zero_grid(), self.zero_grid(), np.zeros(2)

        self._solver_counter += 1

        load = (
            direct_load
            + self.water_density * self.ocean_function * mean_sea_level_change
        )

        angular_velocity_change = np.zeros(2)

        g = self.gravitational_acceleration
        r = self._rotation_factor
        i = self._inertia_factor
        m = 1 / (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
        ht = self._ht[2]
        kt = self._kt[2]

        err = 1
        count = 0
        count_print = 0
        while err > rtol:

            displacement_lm = self.expand_field(load)
            gravity_potential_change_lm = displacement_lm.copy()

            for l in range(self.lmax + 1):

                displacement_lm.coeffs[:, l, :] *= self._h[l]
                gravity_potential_change_lm.coeffs[:, l, :] *= self._k[l]

                if displacement_load is not None:

                    displacement_lm.coeffs[:, l, :] += (
                        self._h_u[l] * displacement_load_lm.coeffs[:, l, :]
                    )

                    gravity_potential_change_lm.coeffs[:, l, :] += (
                        self._k_u[l] * displacement_load_lm.coeffs[:, l, :]
                    )

                if gravitational_potential_load is not None:

                    displacement_lm.coeffs[:, l, :] += (
                        self._h_phi[l] * gravitational_potential_load_lm.coeffs[:, l, :]
                    )

                    gravity_potential_change_lm.coeffs[:, l, :] += (
                        self._k_phi[l] * gravitational_potential_load_lm.coeffs[:, l, :]
                    )

            if rotational_feedbacks:

                centrifugal_coeffs = r * angular_velocity_change

                displacement_lm.coeffs[:, 2, 1] += ht * centrifugal_coeffs
                gravity_potential_change_lm.coeffs[:, 2, 1] += kt * centrifugal_coeffs

                angular_velocity_change = (
                    i * gravity_potential_change_lm.coeffs[:, 2, 1]
                )

                if angular_momentum_change is not None:
                    angular_velocity_change -= m * angular_momentum_change

                gravity_potential_change_lm.coeffs[:, 2, 1] += (
                    r * angular_velocity_change
                )

            displacement = self.expand_coefficient(displacement_lm)
            gravity_potential_change = self.expand_coefficient(
                gravity_potential_change_lm
            )

            sea_level_change = (-1 / g) * (g * displacement + gravity_potential_change)
            sea_level_change.data += mean_sea_level_change - self.ocean_average(
                sea_level_change
            )

            load_new = (
                direct_load
                + self.water_density * self.ocean_function * sea_level_change
            )
            if count > 1 or mean_sea_level_change != 0:
                err = np.max(np.abs((load_new - load).data)) / np.max(np.abs(load.data))
                if verbose:
                    count_print += 1
                    print(f"Iteration = {count_print}, relative error = {err:6.4e}")

            load = load_new
            count += 1

        return (
            sea_level_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    def centrifugal_potential_change(
        self, angular_velocity_change: np.ndarray
    ) -> SHGrid:
        """Computes the centrifugal potential change from an angular velocity change."""
        centrifugal_potential_change_lm = self.zero_coefficients()
        centrifugal_potential_change_lm.coeffs[:, 2, 1] = (
            self._rotation_factor * angular_velocity_change
        )
        return self.expand_coefficient(centrifugal_potential_change_lm)

    def gravity_potential_change_to_gravitational_potential_change(
        self, gravity_potential_change: SHGrid, angular_velocity_change: np.ndarray
    ) -> SHGrid:
        """Subtracts the centrifugal potential to isolate the gravitational potential."""
        gravitational_potential_change_lm = self.expand_field(gravity_potential_change)
        gravitational_potential_change_lm.coeffs[:, 2, 1] -= (
            self._rotation_factor * angular_velocity_change
        )
        return self.expand_coefficient(gravitational_potential_change_lm)

    def gravitational_potential_change_to_gravity_potential_change(
        self,
        gravitational_potential_change: SHGrid,
        angular_velocity_change: np.ndarray,
    ) -> SHGrid:
        """Adds the centrifugal potential to get the total gravity potential."""
        gravity_potential_change_lm = self.expand_field(gravitational_potential_change)
        gravity_potential_change_lm.coeffs[:, 2, 1] += (
            self._rotation_factor * angular_velocity_change
        )
        return self.expand_coefficient(gravity_potential_change_lm)

    def ocean_projection(self, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 over oceans and `value` elsewhere."""
        return SHGrid.from_array(
            np.where(self.ocean_function.data > 0, 1, value), grid=self.grid
        )

    def ice_projection(self, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 over ice sheets and `value` elsewhere."""
        return SHGrid.from_array(
            np.where(self.ice_thickness.data > 0, 1, value), grid=self.grid
        )

    def land_projection(self, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 over land and `value` elsewhere."""
        return SHGrid.from_array(
            np.where(self.ocean_function.data == 0, 1, value), grid=self.grid
        )

    def northern_hemisphere_projection(self, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 in the Northern Hemisphere and `value` elsewhere."""
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid=self.grid)

    def southern_hemisphere_projection(self, value: float = np.nan) -> SHGrid:
        """Returns a grid that is 1 in the Southern Hemisphere and `value` elsewhere."""
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid=self.grid)

    def altimetry_projection(
        self, latitude1: float = -66.0, latitude2: float = 66.0, value: float = np.nan
    ) -> SHGrid:
        """
        Returns a grid that is 1 in the oceans between specified latitudes
        (typical for satellite altimetry) and `value` elsewhere.
        """
        lats, _ = np.meshgrid(self.lats(), self.lons(), indexing="ij")
        ocean_mask = self.ocean_function.data > 0
        lat_mask = np.logical_and(lats > latitude1, lats < latitude2)
        return SHGrid.from_array(
            np.where(np.logical_and(ocean_mask, lat_mask), 1, value), grid=self.grid
        )

    def disk_load(
        self, delta: float, latitude: float, longitude: float, amplitude: float
    ) -> SHGrid:
        """
        Return a circular disk load.

        Args:
            delta: Radius of the disk in degrees.
            latitude: Latitude of the disk's center in degrees.
            longitude: Longitude of the disk's center in degrees.
            amplitude: Amplitude of the load.
        """
        return amplitude * SHGrid.from_cap(
            delta,
            latitude,
            longitude,
            lmax=self.lmax,
            grid=self.grid,
            extend=self._extend,
            sampling=self._sampling,
        )

    def point_load(
        self,
        latitude: float,
        longitude: float,
        amplitude: float = 1.0,
        smoothing_angle: Optional[float] = None,
    ) -> SHGrid:
        """
        Return a point load, optionally smoothed with a Gaussian kernel.

        Args:
            latitude: Latitude of the point load in degrees.
            longitude: Longitude of the point load in degrees.
            amplitude: Amplitude of the load.
            smoothing_angle: Angle in degrees for Gaussian smoothing.
        """
        theta = 90.0 - latitude
        point_load_lm = self.zero_coefficients()
        ylm = pysh.expand.spharm(
            point_load_lm.lmax, theta, longitude, normalization=self.normalization
        )
        for l in range(point_load_lm.lmax + 1):
            point_load_lm.coeffs[0, l, 0] += ylm[0, l, 0]
            for m in range(1, l + 1):
                point_load_lm.coeffs[0, l, m] += ylm[0, l, m]
                point_load_lm.coeffs[1, l, m] += ylm[1, l, m]
        if smoothing_angle is not None:
            th = 0.5 * smoothing_angle * np.pi / 180.0
            t = th * th
            for l in range(point_load_lm.lmax + 1):
                fac = np.exp(-l * (l + 1) * t)
                point_load_lm.coeffs[:, l, :] *= fac
        point_load_lm *= 1 / self.mean_sea_floor_radius**2
        return amplitude * self.expand_coefficient(point_load_lm)

    def direct_load_from_ice_thickness_change(
        self, ice_thickness_change: SHGrid
    ) -> SHGrid:
        """Converts an ice thickness change into the associated surface mass load."""
        self.check_field(ice_thickness_change)
        return self.ice_density * self.one_minus_ocean_function * ice_thickness_change

    def northern_hemisphere_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Northern Hemisphere ice."""
        ice_change = (
            -fraction * self.ice_thickness * self.northern_hemisphere_projection(0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def southern_hemisphere_load(self, fraction: float = 1.0) -> SHGrid:
        """Returns a load from melting a fraction of Southern Hemisphere ice."""
        ice_change = (
            -fraction * self.ice_thickness * self.southern_hemisphere_projection(0)
        )
        return self.direct_load_from_ice_thickness_change(ice_change)

    def adjoint_loads_for_sea_level_point_measurement(
        self, latitude: float, longitude: float, smoothing_angle: Optional[float] = None
    ) -> Tuple[SHGrid, None, None, None]:
        """Returns the adjoint loads for a sea level point measurement."""
        direct_load = self.point_load(
            latitude, longitude, smoothing_angle=smoothing_angle
        )
        return direct_load, None, None, None

    def adjoint_loads_for_displacement_point_measurement(
        self, latitude: float, longitude: float, smoothing_angle: Optional[float] = None
    ) -> Tuple[None, SHGrid, None, None]:
        """Returns the adjoint loads for a displacement point measurement."""
        displacement_load = -1.0 * self.point_load(
            latitude, longitude, smoothing_angle=smoothing_angle
        )
        return None, displacement_load, None, None

    def adjoint_loads_for_gravity_potential_coefficient(
        self, l: int, m: int
    ) -> Tuple[None, None, SHGrid, None]:
        """Returns adjoint loads for an observation of a gravity potential coefficient."""
        if not (0 <= l <= self.lmax and -l <= m <= l):
            raise ValueError(f"(l,m)=({l},{m}) is out of bounds for lmax={self.lmax}.")
        g = self.gravitational_acceleration
        b = self.mean_sea_floor_radius
        adjoint_load_lm = self.zero_coefficients()
        adjoint_load_lm.coeffs[0 if m >= 0 else 1, l, abs(m)] = -g / b**2
        adjoint_load = self.expand_coefficient(adjoint_load_lm)
        return None, None, adjoint_load, None

    def adjoint_loads_for_gravitational_potential_coefficient(
        self, l: int, m: int
    ) -> Tuple[None, None, SHGrid, np.ndarray]:
        """Returns adjoint loads for an observation of a gravitational potential coefficient."""
        _, _, adjoint_load, _ = self.adjoint_loads_for_gravity_potential_coefficient(
            l, m
        )
        angular_momentum_change = self.angular_momentum_change_from_potential(
            adjoint_load
        )
        return None, None, adjoint_load, angular_momentum_change

    def angular_momentum_change_from_potential(
        self, gravitational_potential_load: SHGrid
    ) -> np.ndarray:
        """Returns the adjoint angular momentum change for a given gravitational potential load."""
        gravitational_potential_load_lm = self.expand_field(
            gravitational_potential_load, lmax_calc=2
        )
        r = self._rotation_factor
        b = self.mean_sea_floor_radius
        return -r * b * b * gravitational_potential_load_lm.coeffs[:, 2, 1]

    def gaussian_averaging_function(
        self, r: float, latitude: float, longitude: float, cut: bool = False
    ) -> SHGrid:
        """
        Returns a Gaussian averaging function centered at a point.

        Args:
            r: Radius of the averaging function in kilometers.
            latitude: Latitude of the center in degrees.
            longitude: Longitude of the center in degrees.
            cut: If true, the function is high-pass filtered by removing l<2.
        """
        th0 = (90.0 - latitude) * np.pi / 180.0
        ph0 = longitude * np.pi / 180.0
        c = np.log(2) / (1 - np.cos(1000 * r / self.mean_sea_floor_radius))
        fac = 2 * np.pi * (1 - np.exp(-2 * c))
        fac = c / (self.mean_sea_floor_radius**2 * fac)
        w = self.zero_grid()
        lats, lons = self.lats(), self.lons()
        for ilat, lat in enumerate(lats):
            th = (90.0 - lat) * np.pi / 180.0
            fac1 = np.cos(th) * np.cos(th0)
            fac2 = np.sin(th) * np.sin(th0)
            for ilon, lon in enumerate(lons):
                ph = lon * np.pi / 180.0
                calpha = fac1 + fac2 * np.cos(ph - ph0)
                w.data[ilat, ilon] = fac * np.exp(-c * (1 - calpha))
        if cut:
            w_lm = self.expand_field(w)
            w_lm.coeffs[:, :2, :] = 0.0
            w = self.expand_coefficient(w_lm)
        return w

    def lebesgue_load_space(self) -> Lebesgue:
        """
        Defines the mathematical space for square-integrable surface mass loads.

        This method returns an instance of a pygeoinf Lebesgue space ($L^2$),
        which represents scalar functions on the sphere with a finite integral
        of their square. It is the natural function space for defining loads
        without imposing any prior assumptions about their smoothness.

        Returns:
            An instance of `pygeoinf.symmetric_space.sphere.Lebesgue`.
        """
        return Lebesgue(
            self.lmax,
            radius=self.mean_sea_floor_radius,
            grid=self._grid_name(),
        )

    def lebesgue_response_space(self) -> HilbertSpaceDirectSum:
        """
        Defines the mathematical space for the physical response fields.

        This method returns a direct sum of Hilbert spaces, where each component
        corresponds to a distinct physical field resulting from the sea-level
        calculation. This composite space serves as the codomain for the
        Lebesgue-space sea-level operator.

        The components of the direct sum are:
        1. Sea-level change (a Lebesgue $L^2$ space).
        2. Vertical surface displacement (a Lebesgue $L^2$ space).
        3. Gravitational potential change (a Lebesgue $L^2$ space).
        4. Angular velocity change (a 2D Euclidean space for [ω_x, ω_y]).

        Returns:
            An instance of `pygeoinf.HilbertSpaceDirectSum`.
        """
        field_space = self.lebesgue_load_space()
        return HilbertSpaceDirectSum(
            [field_space, field_space, field_space, EuclideanSpace(2)]
        )

    def sobolev_load_space(self, order: float, scale: float) -> Sobolev:
        """
        Defines a Sobolev space for surface mass loads with smoothness constraints.

        This space is primarily used to regularize inverse problems. By seeking
        a solution within a Sobolev space, we implicitly penalize roughness,
        leading to a more physically plausible (i.e., smoother) estimate of
        the unknown surface load.

        Args:
            order: The Sobolev order (s > 0), which controls the degree of
                smoothness. Higher orders enforce greater smoothness.
            scale: The Sobolev scale (λ > 0), a characteristic length scale.

        Returns:
            An instance of `pygeoinf.symmetric_space.sphere.Sobolev`.
        """
        return Sobolev(
            self.lmax,
            order,
            scale,
            radius=self.mean_sea_floor_radius,
            grid=self._grid_name(),
        )

    def sobolev_response_space(
        self, order: float, scale: float
    ) -> HilbertSpaceDirectSum:
        """
        Defines the response space corresponding to a Sobolev load space.

        When the input load is defined in a Sobolev space of order `s`, the
        resulting physical fields have order `s+1`, this following from
        elliptic regularity.

        Args:
            order: The Sobolev order `s` of the corresponding load space.
            scale: The Sobolev scale `λ` of the corresponding load space.

        Returns:
            An instance of `pygeoinf.HilbertSpaceDirectSum` where the field
            components are Sobolev spaces of order `s+1`.
        """
        field_space = Sobolev(
            self.lmax,
            order + 1,
            scale,
            radius=self.mean_sea_floor_radius,
            grid=self._grid_name(),
        )
        return HilbertSpaceDirectSum(
            [field_space, field_space, field_space, EuclideanSpace(2)]
        )

    def as_lebesgue_linear_operator(
        self, /, *, rotational_feedbacks: bool = True, rtol: float = 1e-6
    ) -> LinearOperator:
        """
        Wraps the physical model as a LinearOperator between Lebesgue spaces.

        This method provides the fundamental mathematical representation of the
        sea-level fingerprint problem. It encapsulates the forward mapping
        (from a surface mass load to the response fields) and its corresponding
        adjoint mapping into a single `pygeoinf.LinearOperator` object. This
        operator acts on simple, square-integrable functions ($L^2$) without
        any smoothness priors.

        Args:
            rotational_feedbacks: If True, include polar wander effects in the
                forward and adjoint calculations. Defaults to True.
            rtol: The relative tolerance for the underlying iterative solver.
                Defaults to 1e-6.

        Returns:
            A `pygeoinf.LinearOperator` that maps from the Lebesgue load
            space to the Lebesgue response space.
        """

        domain = self.lebesgue_load_space()
        codomain = self.lebesgue_response_space()

        def mapping(u: SHGrid) -> List[Union[SHGrid, np.ndarray]]:
            """The forward mapping from a load to the sea level response fields."""
            (
                sea_level_change,
                vertical_displacement,
                gravity_potential_change,
                angular_velocity_change,
            ) = self(
                direct_load=u, rotational_feedbacks=rotational_feedbacks, rtol=rtol
            )

            if rotational_feedbacks:
                gravitational_potential_change = (
                    self.gravity_potential_change_to_gravitational_potential_change(
                        gravity_potential_change, angular_velocity_change
                    )
                )
            else:
                gravitational_potential_change = gravity_potential_change

            return [
                sea_level_change,
                vertical_displacement,
                gravitational_potential_change,
                angular_velocity_change,
            ]

        def adjoint_mapping(response: List[Union[SHGrid, np.ndarray]]) -> SHGrid:
            """The adjoint mapping from response fields to the adjoint load."""
            g = self.gravitational_acceleration
            adjoint_direct_load = response[0]
            adjoint_displacement_load = -1 * response[1]
            adjoint_gravitational_potential_load = -g * response[2]

            if rotational_feedbacks:
                adjoint_angular_momentum_change = -g * (
                    response[3]
                    + self.angular_momentum_change_from_potential(response[2])
                )
            else:
                adjoint_angular_momentum_change = None

            adjoint_sea_level, _, _, _ = self(
                direct_load=adjoint_direct_load,
                displacement_load=adjoint_displacement_load,
                gravitational_potential_load=adjoint_gravitational_potential_load,
                angular_momentum_change=adjoint_angular_momentum_change,
                rotational_feedbacks=rotational_feedbacks,
                rtol=rtol,
            )
            return adjoint_sea_level

        return LinearOperator(
            domain, codomain, mapping, adjoint_mapping=adjoint_mapping
        )

    def as_sobolev_linear_operator(
        self,
        order: float,
        scale: float,
        /,
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-6,
    ) -> LinearOperator:
        """
        Constructs the sea-level model as a LinearOperator between Sobolev spaces.

        This is the primary tool for solving regularized inverse problems. By
        defining the operator on Sobolev spaces, we frame the problem to seek a
        spatially smooth surface load, which is often a necessary physical
        constraint to obtain a unique and stable solution from noisy or
        incomplete data. This approach is equivalent to a form of Tikhonov
        regularization.

        The operator correctly handles the transformation between the underlying
        Lebesgue representation (where the physics is calculated) and the
        Sobolev representation (where the problem is posed) by internally
        managing the "mass matrices" that define the Sobolev inner products.

        Args:
            order: The Sobolev order (s > 0), controlling the degree of
                smoothness of the input load space.
            scale: The Sobolev scale (λ > 0), a characteristic length scale
                that defines the spatial scale at which the smoothness is enforced.
            rotational_feedbacks: If True, include polar wander effects. Defaults to True.
            rtol: Relative tolerance for the underlying iterative solver. Defaults to 1e-6.

            Returns:
                A `pygeoinf.LinearOperator` that maps from the Sobolev load
                space to the Sobolev response space.
        """

        domain = self.sobolev_load_space(order, scale)
        codomain = self.sobolev_response_space(order, scale)

        lebesgue_operator = self.as_lebesgue_linear_operator(
            rotational_feedbacks=rotational_feedbacks, rtol=rtol
        )

        return LinearOperator.from_formal_adjoint(domain, codomain, lebesgue_operator)
