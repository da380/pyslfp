"""
Module for the FingerPrint class that allows for forward 
and adjoint elastic fingerprint calculations. 
"""

import numpy as np
import pyshtools as pysh
from pyshtools import SHGrid, SHCoeffs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# from pyslfp.fields import ResponseFields, ResponseCoefficients
from pyslfp.ice_ng import IceNG
from pyslfp.physical_parameters import EarthModelParameters

from . import DATADIR


if __name__ == "__main__":
    pass


class FingerPrint(EarthModelParameters):
    """
    Class for computing elastic sea level fingerprints.

    Initialisation of the class sets up various computational options,
    but the backgroud sea level and ice thickness are not set. The latter
    fields must be set separately, and until this is done fingerprint
    calculations can not be performed.
    """

    def __init__(
        self,
        /,
        *,
        lmax=256,
        earth_model_parameters=None,
        grid="DH",
        extend=True,
        love_number_file=DATADIR + "/love_numbers/PREM_4096.dat",
    ):
        """
        Args:
            lmax (int): Truncation degree for spherical harmonic expansions.
            earth_model_parameters (EarthModelParameters): Parameters for the Earth model.
            grid (str): pyshtools grid option.
            extend (bool): If True, spatial grid extended to inlcude 360 degrees. Default is True.
            love_number_file (str): Path to file containing the Love numbers.
        """

        # Set up the earth model parameters
        if earth_model_parameters is None:
            super().__init__()
        else:
            super().__init__(
                length_scale=earth_model_parameters.length_scale,
                density_scale=earth_model_parameters.density_scale,
                time_scale=earth_model_parameters.time_scale,
                equatorial_radius=earth_model_parameters.equatorial_radius,
                polar_radius=earth_model_parameters.polar_radius,
                mean_radius=earth_model_parameters.mean_radius,
                mean_sea_floor_radius=earth_model_parameters.mean_sea_floor_radius,
                mass=earth_model_parameters.mass,
                gravitational_acceleration=earth_model_parameters.gravitational_acceleration,
                equatorial_moment_of_inertia=earth_model_parameters.equatorial_moment_of_inertia,
                polar_moment_of_inertia=earth_model_parameters.polar_moment_of_inertia,
                rotation_frequency=earth_model_parameters.rotation_frequency,
                water_density=earth_model_parameters.water_density,
                ice_density=earth_model_parameters.ice_density,
            )

        # Set some options.
        self._lmax = lmax
        self._extend = extend
        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1

        # Do not change these parameters!
        self._normalization = "ortho"
        self._csphase = 1

        # Scaling of degree-zero coefficient to set surface integral.
        self._integration_factor = np.sqrt((4 * np.pi)) * self._mean_sea_floor_radius**2

        # Scaling of angular velocity perturbtion to get (2,\pm 1)-coefficients for
        # centrifugal potential.
        self._rotation_factor = (
            np.sqrt((4 * np.pi) / 15.0)
            * self.rotation_frequency
            * self.mean_sea_floor_radius**2
        )

        # Scaling of gravitational potential (2,\pm 1)-coefficients to get
        # angular velocity perturbations.
        self._inertia_factor = (
            np.sqrt(5 / (12 * np.pi))
            * self.rotation_frequency
            * self.mean_sea_floor_radius**3
            / (
                self.gravitational_constant
                * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            )
        )

        # Read in the Love numbers.
        self._love_number_file = love_number_file
        self._read_love_numbers(love_number_file)

        # Background model not set.
        self._sea_level = None
        self._ice_thickness = None
        self._ocean_function = None
        self._ocean_area = None

        # Initialise the counter for number of solver calls.
        self._solver_counter = 0

    # -----------------------------------------------#
    #                    Properties                  #
    # -----------------------------------------------#

    @property
    def lmax(self):
        """Return truncation degree for expansions."""
        return self._lmax

    @property
    def normalization(self):
        """Return spherical harmonic normalisation convention."""
        return self._normalization

    @property
    def csphase(self):
        """Return Condon-Shortley phase option."""
        return self._csphase

    @property
    def grid(self):
        """Return spatial grid option."""
        return self._grid

    @property
    def extend(self):
        """True if grid extended to include 360 degree longitude."""
        return self._extend

    @property
    def background_set(self):
        """
        Returns true is background state has been set.
        """
        return self._sea_level is not None and self._ice_thickness is not None

    @property
    def sea_level(self):
        """Returns the backgroud sea level."""
        if self._sea_level is None:
            raise NotImplementedError("Sea level not set.")
        else:
            return self._sea_level

    @sea_level.setter
    def sea_level(self, value):
        self._check_field(value)
        self._sea_level = value
        self._ocean_function = None

    @property
    def ice_thickness(self):
        """Returns the backgroud ice thickness."""
        if self._ice_thickness is None:
            raise NotImplementedError("Ice thickness not set.")
        else:
            return self._ice_thickness

    @ice_thickness.setter
    def ice_thickness(self, value):
        self._check_field(value)
        self._ice_thickness = value
        self._ocean_function = None

    @property
    def ocean_function(self):
        """Returns the ocean function."""
        if self._ocean_function is None:
            self._compute_ocean_function()
        return self._ocean_function

    @property
    def one_minus_ocean_function(self):
        """Returns 1 - C, with C the ocean function."""
        tmp = self.ocean_function.copy()
        tmp.data = 1 - tmp.data
        return tmp

    @property
    def ocean_area(self):
        """Returns the ocean area."""
        if self._ocean_area is None:
            self._compute_ocean_area()
        return self._ocean_area

    @property
    def solver_counter(self):
        """
        Returns the number of times the solver method
        has been called.
        """
        return self._solver_counter

    # ---------------------------------------------------------#
    #                     Private methods                     #
    # ---------------------------------------------------------#

    def _read_love_numbers(self, file):
        # Read in the Love numbers from a given file and non-dimensionalise.

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if self.lmax > data_degree:
            raise ValueError("maximum degree is larger than present in data file")

        self._h_u = data[: self.lmax + 1, 1] * self.load_scale / self.length_scale
        self._k_u = (
            data[: self.lmax + 1, 2]
            * self.load_scale
            / self.gravitational_potential_scale
        )
        self._h_phi = data[: self.lmax + 1, 3] * self.load_scale / self.length_scale
        self._k_phi = (
            data[: self.lmax + 1, 4]
            * self.load_scale
            / self.gravitational_potential_scale
        )

        self._h = self._h_u + self._h_phi
        self._k = self._k_u + self._k_phi

        self._ht = (
            data[: self.lmax + 1, 5]
            * self.gravitational_potential_scale
            / self.length_scale
        )
        self._kt = data[: self.lmax + 1, 6]

    def _check_field(self, f):
        # Check SHGrid object is compatible with options.
        return f.lmax == self.lmax and f.grid == self.grid and f.extend == self.extend

    def _check_coefficient(self, f):
        # Check SHCoeff object is compatible with options.
        return (
            f.lmax == self.lmax
            and f.normalization == self.normalization
            and f.csphase == self.csphase
        )

    def _expand_field(self, f, /, *, lmax_calc=None):
        # Expand a SHGrid object using stored parameters.
        assert self._check_field(f)
        if lmax_calc is None:
            return f.expand(normalization=self.normalization, csphase=self.csphase)
        else:
            return f.expand(
                lmax_calc=lmax_calc,
                normalization=self.normalization,
                csphase=self.csphase,
            )

    def _expand_coefficient(self, f):
        # Expand a SHCoeff object using stored parameters.
        assert self._check_coefficient(f)
        if self._sampling == 2:
            grid = "DH2"
        else:
            grid = self.grid
        return f.expand(grid=grid, extend=self.extend)

    def _compute_ocean_function(self):
        # Computes and stores the ocean function.
        if self._sea_level is None or self._ice_thickness is None:
            raise NotImplementedError("Sea level and/or ice thickness not set")
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

    def _compute_ocean_area(self):
        # Computes and stores the ocean area.
        if self._ocean_function is None:
            self._compute_ocean_function()
        self._ocean_area = self.integrate(self._ocean_function)

    # --------------------------------------------------------#
    #                       Public methods                    #
    # --------------------------------------------------------#

    def lats(self):
        """
        Return the latitudes for the spatial grid.
        """
        return self.zero_grid().lats()

    def lons(self):
        """
        Return the longitudes for the spatial grid.
        """
        return self.zero_grid().lons()

    def displacement_love_numbers(self):
        """
        Return the displacement Love numbers.
        """
        return self._h

    def gravitational_love_numbers(self):
        """
        Return the gravitational Love numbers.
        """
        return self._k

    def load_response(self, load):
        """
        Returns the deformation fields associated with a load. This
        calculation does not account for gravitationally self-consistent
        sea level change nor rotational feedbacks.

        Args:

            load (SHGrid): The applied load.

        Returns:
            SHGrid: Vertical displacement.
            SHGrid: Gravitational potential change.

        """

        self._check_field(load)
        displacement_lm = self._expand_field(load)
        gravitational_potential_change_lm = displacement_lm.copy()

        for l in range(0, self.lmax + 1):
            displacement_lm.coeffs[:, l, :] *= self._h[l]
            gravitational_potential_change_lm.coeffs[:, l, :] *= self._k[l]

        displacement = self._expand_coefficient(displacement_lm)
        gravitational_potential_change = self._expand_coefficient(
            gravitational_potential_change_lm
        )
        return displacement, gravitational_potential_change

    def integrate(self, f):
        """Integrate function over the surface.

        Args:
            f (SHGrid): Function to integrate.

        Returns:
            float: Integral of the function over the surface.
        """
        return (
            self._integration_factor
            * self._expand_field(f, lmax_calc=0).coeffs[0, 0, 0]
        )

    def point_evaulation(self, f, latitude, longitude, degrees=True):
        """Evaluate a function at a given point.

        Args:
            f (SHGrid): Function to evaluate.
            latitude (float): Latitude of the point.
            longitude (float): Longitude of the point.

        Returns:
            float: Value of the function at the point.
        """
        f_lm = self._expand_field(f)
        return f_lm.expand(lat=[latitude], lon=[longitude], degrees=degrees)[0]

    def coefficient_evaluation(self, f, l, m):
        """Return the (l,m)th spherical harmonic coefficient of a function.

        Args:
            f (SHGrid): Function to evaluate.
            l (int): The degree.
            m (int): The order.

        Returns:
            float: Value of the coefficient.
        """
        assert 0 <= l <= self.lmax
        assert -l <= m <= l
        f_lm = self._expand_field(f)
        return f_lm.coeffs[0 if m >= 0 else 1, l, abs(m)]

    def zero_grid(self):
        """Return a grid of zeros."""
        return SHGrid.from_zeros(
            lmax=self.lmax,
            grid=self.grid,
            sampling=self._sampling,
            extend=self.extend,
        )

    def constant_grid(self, value):
        """Return a grid of constant values"""
        f = SHGrid.from_zeros(
            lmax=self.lmax,
            grid=self.grid,
            sampling=self._sampling,
            extend=self.extend,
        )
        f.data[:, :] = value
        return f

    def zero_coefficients(self):
        """Return coefficients of zeros."""
        return SHCoeffs.from_zeros(
            lmax=self.lmax,
            normalization=self.normalization,
            csphase=self.csphase,
        )

    def ocean_average(self, f):
        """Return average of a function over the oceans."""
        return self.integrate(self.ocean_function * f) / self.ocean_area

    def set_state_from_ice_ng(self, /, *, version=7, date=0):
        """
        Sets background state from ice_7g, ice_6g, or ice_5g.

        Args:
            version (int): Selects the model to use.
            data (float): Selects the date from which values are taken.

        Notes:
            To detemrine the fields, linear interpolation between
            model values is applied. If the date is out of range,
            constant extrapolation of the boundary values is used.
        """
        ice_ng = IceNG(version=version)
        ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(
            date,
            self.lmax,
            grid=self.grid,
            sampling=self._sampling,
            extend=self.extend,
        )
        self.ice_thickness = ice_thickness / self.length_scale
        self.sea_level = sea_level / self.length_scale

    def mean_sea_level_change(self, direct_load):
        """
        Returns the mean sea level change associated with a direct load.
        """
        assert self._check_field(direct_load)
        return -self.integrate(direct_load) / (self.water_density * self.ocean_area)

    def __call__(
        self,
        /,
        *,
        direct_load=None,
        displacement_load=None,
        gravitational_potential_load=None,
        angular_momentum_change=None,
        rotational_feedbacks=True,
        rtol=1.0e-6,
        verbose=False,
    ):
        """
        Returns the solution to the generalised fingerprint problem for a given generalised load.
        If only a non-zero direct load is input, then the calculation reduces to the standard
        fingerprint problem.

        Args:
            direct_load (SHGrid): The direct load applied in the problem. Default is None.
            displacement_load (SHGrid): The displacement load applied in the problem. Default is None.
            gravitational_potential_load (SHGrid): The gravitational potential load applied in the
                 problem. The default is None.
            angular_momentum_change (numpy vector): The angular momentum change. Default is None.
            rtol (float): Relative tolerance for iterative solution. Default is 1e-6.
            verbose (bool): If True, relative errors printed during iterations.

        Returns:
            SHGrid: sea_level_change.
            SHGrid: displacement.
            SHGrid: Gravity potential change.
            numpy vector: angular velocity change.
        """

        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            loads_present = True
            assert self._check_field(direct_load)
            mean_sea_level_change = -self.integrate(direct_load) / (
                self.water_density * self.ocean_area
            )
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0

        else:
            direct_load = self.zero_grid()
            mean_sea_level_change = 0

        if displacement_load is not None:
            loads_present = True
            assert self._check_field(displacement_load)
            displacement_load_lm = self._expand_field(displacement_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(displacement_load.data)) > 0

        if gravitational_potential_load is not None:
            loads_present = True
            assert self._check_field(gravitational_potential_load)
            gravitational_potential_load_lm = self._expand_field(
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

            displacement_lm = self._expand_field(load)
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

            displacement = self._expand_coefficient(displacement_lm)
            gravity_potential_change = self._expand_coefficient(
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

    def centrifugal_potential_change(self, angular_velocity_change):
        """
        Returns the centrifugal potential change associated with a given
        angular velocity change.
        """
        centrifugal_potential_change_lm = self.zero_coefficients()
        centrifugal_potential_change_lm.coeffs[:, 2, 1] = (
            self._rotation_factor * angular_velocity_change
        )
        return self._expand_coefficient(centrifugal_potential_change_lm)

    def gravity_potential_change_to_gravitational_potential_change(
        self, gravity_potential_change, angular_velocity_change
    ):
        """
        Subtracts the centrifugal potential perturbation from the
        gravity potential change to isolate the gravitational potential change.

        Args:
            gravity_potential_change (SHGrid): The gravity potential change.
            angular_velocity_change (numpy vector): The angular velocity change.

        Returns:
            (SHGrid): The gravitational potential change.
        """
        gravitational_potential_change_lm = self._expand_field(gravity_potential_change)
        gravitational_potential_change_lm.coeffs[:, 2, 1] -= (
            self._rotation_factor * angular_velocity_change
        )
        return self._expand_coefficient(gravitational_potential_change_lm)

    def gravitational_potential_change_to_gravity_potential_change(
        self, gravitational_potential_change, angular_velocity_change
    ):
        """
        Adds the centrifugal potential perturbation from the
        gravitational potential change to return the gravity potential change.

        Args:
            gravitational_potential_change (SHGrid): The gravitational potential change.
            angular_velocity_change (numpy vector): The angular velocity change.

        Returns:
            (SHGrid): The gravitaty potential change.
        """
        gravity_potential_change_lm = self._expand_field(gravitational_potential_change)
        gravity_potential_change_lm.coeffs[:, 2, 1] += (
            self._rotation_factor * angular_velocity_change
        )
        return self._expand_coefficient(gravity_potential_change_lm)

    def ocean_projection(self, value=np.nan):
        """
        Returns a field that is 1 over the oceans and equal to "value" elsewhere.
        The defult value is NaN.
        """
        return SHGrid.from_array(
            np.where(self.ocean_function.data > 0, 1, value), grid=self.grid
        )

    def ice_projection(self, value=np.nan):
        """
        Returns a field that is 1 over the ice sheets and equal to "value" elsewhere.
        The defult value is NaN.
        """
        return SHGrid.from_array(
            np.where(self.ice_thickness.data > 0, 1, value), grid=self.grid
        )

    def land_projection(self, value=np.nan):
        """
        Returns a field that is 1 over the land and equal to "value" elsewhere.
        The defult value is NaN.
        """

        return SHGrid.from_array(
            np.where(self.ocean_function.data == 0, 1, value), grid=self.grid
        )

    def northern_hemisphere_projection(self, value=np.nan):
        """
        Returns a field that is 1 over the northern hemisphere and equal to "value" elsewhere.
        The defult value is NaN.
        """
        lats, _ = np.meshgrid(
            self.ice_thickness.lats(),
            self.ice_thickness.lons(),
            indexing="ij",
        )
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid=self.grid)

    def southern_hemisphere_projection(self, value=np.nan):
        """
        Returns a field that is 1 over the southern hemisphere and equal to "value" elsewhere.
        The defult value is NaN.
        """
        lats, _ = np.meshgrid(
            self.ice_thickness.lats(),
            self.ice_thickness.lons(),
            indexing="ij",
        )
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid=self.grid)

    def altimetery_projection(self, latitude1=-66.0, latitude2=66.0, value=np.nan):
        """
        Returns a function that is equal to 1 in the oceans between the specified
        latitudes, and elsewhere equal to a given value.

        Args:
            latitude1 (float): Latitude below which the field equals the chosen value.
                Default is -66 degrees.
            latitude2 (float): Latitude above which the field equals the chosen value.
                Default is +66 degrees.
            value (float): Value of the function outside of the latitude range. Default
                value is NaN
        """
        lats, _ = np.meshgrid(
            self.ice_thickness.lats(),
            self.ice_thickness.lons(),
            indexing="ij",
        )
        return SHGrid.from_array(
            np.where(
                np.logical_and(
                    np.logical_and(lats > latitude1, lats < latitude2),
                    self.ocean_function.data == 0,
                ),
                1,
                value,
            ),
            grid=self.grid,
        )

    def disk_load(self, delta, latitutude, longitude, amplitude):
        """Return a disk load.

        Args:
            delta (float): Radius of the disk.
            latitutude (float): Latitude of the centre of the disk.
            longitude (float): Longitude of the centre of the disk.
            amplitude (float): Amplitude of the load.

        Returns:
            SHGrid: Load associated with the disk.
        """
        return amplitude * SHGrid.from_cap(
            delta,
            latitutude,
            longitude,
            lmax=self.lmax,
            grid=self.grid,
            extend=self._extend,
            sampling=self._sampling,
        )

    def point_load(self, latitude, longitude, amplitude=1, smoothing_angle=None):
        """Return a point load.

        Args:
            latitude (float): Latitude of the point load in degrees.
            longitude (float): Longitude of the point load in degrees.
            amplitude (float): Amplitude of the load.
            smoothing_angle (float): Angle over which point load
                 is smoothed. Default is None

        Returns:
            SHGrid: Load associated with the point load.
        """
        theta = 90.0 - latitude
        point_load_lm = self.zero_coefficients()
        ylm = pysh.expand.spharm(
            point_load_lm.lmax, theta, longitude, normalization=self.normalization
        )

        for l in range(0, point_load_lm.lmax + 1):
            point_load_lm.coeffs[0, l, 0] += ylm[0, l, 0]
            for m in range(1, l + 1):
                point_load_lm.coeffs[0, l, m] += ylm[0, l, m]
                point_load_lm.coeffs[1, l, m] += ylm[1, l, m]

        if smoothing_angle is not None:
            th = 0.5 * smoothing_angle * np.pi / 180
            t = th * th
            for l in range(0, point_load_lm.lmax + 1):
                fac = np.exp(-l * (l + 1) * t)
                point_load_lm.coeffs[:, l, :] *= fac

        point_load_lm = (1 / self.mean_sea_floor_radius**2) * point_load_lm
        point_load = amplitude * self._expand_coefficient(point_load_lm)

        return point_load

    def direct_load_from_ice_thickness_change(self, ice_thickness_change):
        """Converts an ice thickness change into the associated load.

        Args:
            ice_thickness_change (SHGrid): Ice thickness change.

        Returns:
            SHGrid: Load associated with the ice thickness change.
        """
        self._check_field(ice_thickness_change)
        return self.ice_density * self.one_minus_ocean_function * ice_thickness_change

    def northern_hemisphere_load(self, fraction=1):
        """Returns a load associated with melting the given fraction of ice in the northern hemisphere.

        Args:
            fraction (float): Fraction of ice to melt.

        Returns:
            SHGrid: Load associated with melting the given fraction of ice in the northern hemisphere.
        """
        ice_thickness_change = (
            -fraction * self.ice_thickness * self.northern_hemisphere_projection(0)
        )
        return self.direct_load_from_ice_thickness_change(ice_thickness_change)

    def southern_hemisphere_load(self, fraction=1):
        """Returns a load associated with melting the given fraction of ice in the northern hemisphere.

        Args:
            fraction (float): Fraction of ice to melt.

        Returns:
            SHGrid: Load associated with melting the given fraction of ice in the northern hemisphere.
        """
        ice_thickness_change = (
            -fraction * self.ice_thickness * self.southern_hemisphere_projection(0)
        )
        return self.direct_load_from_ice_thickness_change(ice_thickness_change)

    def adjoint_loads_for_sea_level_point_measurement(
        self, latitude, longitude, smoothing_angle=None
    ):
        """Returns the adjoint loads for a sea level measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields:
        """
        direct_load = self.point_load(
            latitude, longitude, smoothing_angle=smoothing_angle
        )
        return direct_load, None, None, None

    def adjoint_loads_for_displacement_point_measurement(
        self, latitude, longitude, smoothing_angle=None
    ):
        """Returns the adjoint loads for a displacement measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields: Adjoint loads for the displacement measurement.
        """
        displacement_load = -1 * self.point_load(
            latitude, longitude, smoothing_angle=smoothing_angle
        )
        return None, displacement_load, None, None

    def adjoint_loads_for_gravity_potential_coefficient(self, l, m):
        """
        Returns the adjoint loads for an observation of the gravity potential change
        at the given degree and order.
        """
        assert 0 <= l <= self.lmax
        assert -l <= m <= l
        g = self.gravitational_acceleration
        b = self.mean_sea_floor_radius
        adjoint_load_lm = self.zero_coefficients()
        adjoint_load_lm.coeffs[0 if m >= 0 else 1, l, abs(m)] = -g / b**2
        adjoint_load = self._expand_coefficient(adjoint_load_lm)
        return None, None, adjoint_load, None

    def adjoint_loads_for_gravitational_potential_coefficient(self, l, m):
        """
        Returns the adjoint loads for an observation of the gravitational potential change
        at the given degree and order.
        """
        _, _, adjoint_load, _ = self.adjoint_loads_for_gravity_potential_coefficient(
            l, m
        )
        angular_momentum_change = self.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(
            adjoint_load
        )
        return None, None, adjoint_load, angular_momentum_change

    def adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(
        self, gravitational_potential_load
    ):
        """
        Returns the angular momentum change for a given gravitational
        potential load. This method is used to remove the centrifugal contribution
        from the associated measurement.
        """
        gravitational_potential_load_lm = self._expand_field(
            gravitational_potential_load, lmax_calc=2
        )
        r = self._rotation_factor
        b = self.mean_sea_floor_radius
        return -r * b * b * gravitational_potential_load_lm.coeffs[:, 2, 1]

    def gaussian_averaging_function(self, r, latitude, longitude, cut=False):
        """
        Returns a Gaussian averaging function.

        Args:
            r (float): Radius of the averaging function.
            latitude (float): Latitude of the centre of the averaging function.
            longitude (float): Longitude of the centre of the averaging function.
            cut (bool): If true, the averaging function is cut at the truncation degree.

        Returns:
            SHGrid: Gaussian averaging function.
        """
        th0 = (90 - latitude) * np.pi / 180
        ph0 = (longitude) * np.pi / 180
        c = np.log(2) / (1 - np.cos(1000 * r / self.mean_sea_floor_radius))
        fac = 2 * np.pi * (1 - np.exp(-2 * c))
        fac = c / (self.mean_sea_floor_radius**2 * fac)
        w = self.zero_grid()
        for ilat, lat in enumerate(w.lats()):
            th = (90 - lat) * np.pi / 180
            fac1 = np.cos(th) * np.cos(th0)
            fac2 = np.sin(th) * np.sin(th0)
            for ilon, lon in enumerate(w.lons()):
                ph = lon * np.pi / 180
                calpha = fac1 + fac2 * np.cos(ph - ph0)
                w.data[ilat, ilon] = fac * np.exp(-c * (1 - calpha))
        if cut:
            w_lm = self._expand_field(w)
            w_lm.coeffs[:, :2, :] = 0.0
            w = self._expand_coefficient(w_lm)
        return w
