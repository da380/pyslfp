"""
Module for the FingerPrint class that allows for forward 
and adjoint elastic fingerprint calculations. 
"""

import numpy as np
import pyshtools as pysh
from pyshtools import SHGrid, SHCoeffs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# from pyslfp.fields import ResponseFields, ResponseCoefficients
from pyslfp.ice_ng import IceNG

from . import DATADIR


if __name__ == "__main__":
    pass


class FingerPrint:
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
        length_scale=1,
        mass_scale=1,
        time_scale=1,
        grid="DH",
        love_number_file=DATADIR + "/love_numbers/PREM_4096.dat",
    ):
        """
        Args:
            lmax (int): Truncation degree for spherical harmonic expansions.
            length_scale (float): Length used for non-dimensionalisation.
            mass_scale (float): Mass used for non-dimensionalisation.
            time_scale (float): Time used for non-dimensionalisation.
            grid (str): pyshtools grid option.
            love_number_file (str): Path to file containing the Love numbers.
        """

        # Set the base units.
        self._length_scale = length_scale
        self._mass_scale = mass_scale
        self._time_scale = time_scale

        # Set the derived units.
        self._frequency_scale = 1 / self.time_scale
        self._density_scale = self.mass_scale / self.length_scale**3
        self._load_scale = self.mass_scale / self.length_scale**2
        self._velocity_scale = self.length_scale / self.time_scale
        self._acceleration_scale = self.velocity_scale / self.time_scale
        self._gravitational_potential_scale = (
            self.acceleration_scale * self.length_scale
        )
        self._moment_of_inertia_scale = self.mass_scale * self.length_scale**2

        # Set the physical constants.
        self._equatorial_radius = 6378137 / self.length_scale
        self._polar_radius = 6356752 / self.length_scale
        self._mean_radius = 6371000 / self.length_scale
        self._mean_sea_floor_radius = 6368000 / self.length_scale
        self._mass = 5.974e24 / self.mass_scale
        self._gravitational_acceleration = 9.825652323 / self.acceleration_scale
        self._gravitational_constant = (
            6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3
        )
        self._equatorial_moment_of_inertia = 8.0096e37 / self.moment_of_inertia_scale
        self._polar_moment_of_inertia = 8.0359e37 / self.moment_of_inertia_scale
        self._rotation_frequency = 7.27220521664304e-05 / self.frequency_scale
        self._water_density = 1000 / self.density_scale
        self._ice_density = 917 / self.density_scale
        self._solid_earth_surface_density = 2600.0 / self.density_scale

        # Set some options.
        self._lmax = lmax
        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1
        self._extend = True
        self._normalization = "ortho"
        self._csphase = 1

        # Read in the Love numbers.
        self._read_love_numbers(love_number_file)

        # Pre-compute some constants.
        self._integration_factor = np.sqrt((4 * np.pi)) * self._mean_sea_floor_radius**2
        self._rotation_factor = (
            np.sqrt((4 * np.pi) / 15.0)
            * self.rotation_frequency
            * self.mean_sea_floor_radius**2
        )
        self._inertia_factor = (
            np.sqrt(5 / (12 * np.pi))
            * self.rotation_frequency
            * self.mean_sea_floor_radius**3
            / (
                self.gravitational_constant
                * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            )
        )

        # Background model not set.
        self._background_sea_level = None
        self._background_ice_thickness = None
        self._ocean_function = None
        self._ocean_area = None

    # ------------------------------------------------#
    #          Properties related to units           #
    # ------------------------------------------------#

    @property
    def length_scale(self):
        """Return length for non-dimensionalisation."""
        return self._length_scale

    @property
    def mass_scale(self):
        """Return mass for non-dimensionalisation."""
        return self._mass_scale

    @property
    def time_scale(self):
        """Return time for non-dimensionalisation."""
        return self._time_scale

    @property
    def frequency_scale(self):
        """Return frequency for non-dimensionalisation."""
        return self._frequency_scale

    @property
    def density_scale(self):
        """Return density for non-dimensionalisation."""
        return self._density_scale

    @property
    def load_scale(self):
        """Return load for non-dimensionalisation."""
        return self._load_scale

    @property
    def velocity_scale(self):
        """Return velocity for non-dimensionalisation."""
        return self._velocity_scale

    @property
    def acceleration_scale(self):
        """Return acceleration for non-dimensionalisation."""
        return self._acceleration_scale

    @property
    def gravitational_potential_scale(self):
        """Return gravitational potential for non-dimensionalisation."""
        return self._gravitational_potential_scale

    @property
    def moment_of_inertia_scale(self):
        """Return moment of intertia for non-dimensionalisation."""
        return self._moment_of_inertia_scale

    # -----------------------------------------------------#
    #      Properties related to physical constants       #
    # -----------------------------------------------------#

    @property
    def equatorial_radius(self):
        """Return Earth's equatorial radius."""
        return self._equatorial_radius

    @property
    def polar_radius(self):
        """Return Earth's polar radius."""
        return self._polar_radius

    @property
    def mean_radius(self):
        """Return Earth's mean radius."""
        return self._mean_radius

    @property
    def mean_sea_floor_radius(self):
        """Return Earth's mean sea floor radius."""
        return self._mean_sea_floor_radius

    @property
    def mass(self):
        """Return Earth's mass."""
        return self._mass

    @property
    def gravitational_acceleration(self):
        """Return Earth's surface gravitational acceleration."""
        return self._gravitational_acceleration

    @property
    def gravitational_constant(self):
        """Return Gravitational constant."""
        return self._gravitational_constant

    @property
    def equatorial_moment_of_inertia(self):
        """Return Earth's equatorial moment of inertia."""
        return self._equatorial_moment_of_inertia

    @property
    def polar_moment_of_inertia(self):
        """Return Earth's polar moment of inertia."""
        return self._polar_moment_of_inertia

    @property
    def rotation_frequency(self):
        """Return Earth's rotational frequency."""
        return self._rotation_frequency

    @property
    def water_density(self):
        """Return density of water."""
        return self._water_density

    @property
    def ice_density(self):
        """Return density of ice."""
        return self._ice_density

    @property
    def solid_earth_surface_density(self):
        """Return density of the solid Earth's surface."""
        return self._solid_earth_surface_density

    # -----------------------------------------------#
    #       Properties related to grid options       #
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

    # ----------------------------------------------------#
    #     Properties related to the background state     #
    # ----------------------------------------------------#

    @property
    def background_sea_level(self):
        """Returns the backgroud sea level."""
        if self._background_sea_level is None:
            raise NotImplementedError("Sea level not set.")
        else:
            return self._background_sea_level

    @background_sea_level.setter
    def background_sea_level(self, value):
        self._check_field(value)
        self._background_sea_level = value

    @property
    def background_ice_thickness(self):
        """Returns the backgroud ice thickness."""
        if self._background_ice_thickness is None:
            raise NotImplementedError("Ice thickness not set.")
        else:
            return self._background_ice_thickness

    @background_ice_thickness.setter
    def background_ice_thickness(self, value):
        self._check_field(value)
        self._background_ice_thickness = value

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

    # ---------------------------------------------------------#
    #                     Private methods                     #
    # ---------------------------------------------------------#

    def _read_love_numbers(self, file):
        # Read in the Love numbers from a given file and non-dimensionalise.

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if self.lmax > data_degree:
            raise ValueError("maximum degree is larger than present in data file")

        self._vertical_displacement_to_vertical_displacement_love_number = (
            data[: self.lmax + 1, 1] * self.load_scale / self.length_scale
        )
        self._vertical_displacement_to_gravitational_potential_love_number = (
            data[: self.lmax + 1, 2]
            * self.load_scale
            / self.gravitational_potential_scale
        )
        self._gravitational_potential_to_vertical_displacement_love_number = (
            data[: self.lmax + 1, 3] * self.load_scale / self.length_scale
        )
        self._gravitational_potential_to_gravitational_potential_love_number = (
            data[: self.lmax + 1, 4]
            * self.load_scale
            / self.gravitational_potential_scale
        )

        self._displacement_love_number = (
            self._vertical_displacement_to_vertical_displacement_love_number
            + self._gravitational_potential_to_vertical_displacement_love_number
        )
        self._gravitational_potential_love_number = (
            self._vertical_displacement_to_gravitational_potential_love_number
            + self._gravitational_potential_to_gravitational_potential_love_number
        )

        self._vertical_displacement_tidal_love_number = (
            data[: self.lmax + 1, 5]
            * self.gravitational_potential_scale
            / self.length_scale
        )
        self._gravitational_potential_tidal_love_number = data[: self.lmax + 1, 6]

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
        if self._background_sea_level is None or self._background_ice_thickness is None:
            raise NotImplementedError("Sea level and/or ice thickness not set")
        self._ocean_function = SHGrid.from_array(
            np.where(
                self.water_density * self.background_sea_level.data
                - self.ice_density * self.background_ice_thickness.data
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

    def _iterate_solver(
        self, load, mean_sea_level_change, /, *, rotational_feedbacks=True
    ):
        # Given a load, returns the solid earth deformation and associated sea level

        assert self._check_field(load)
        load_lm = self._expand_field(load)
        vertical_displacement_coefficients = load_lm.copy()
        gravity_potential_change_coefficients = load_lm.copy()
        for l in range(self.lmax + 1):
            vertical_displacement_coefficients.coeffs[
                :, l, :
            ] *= self._displacement_love_number[l]
            gravity_potential_change_coefficients.coeffs[
                :, l, :
            ] *= self._gravitational_potential_love_number[l]

        if rotational_feedbacks:
            r = self._rotation_factor
            i = self._inertia_factor
            kt = self._gravitational_potential_tidal_love_number[2]
            ht = self._vertical_displacement_tidal_love_number[2]
            f = r * i / (1 - r * i * kt)
            vertical_displacement_coefficients.coeffs[:, 2, 1] += (
                ht * f * gravity_potential_change_coefficients.coeffs[:, 2, 1]
            )
            gravity_potential_change_coefficients.coeffs[:, 2, 1] += (
                kt * f * gravity_potential_change_coefficients.coeffs[:, 2, 1]
            )
            angular_velocity_change = (
                i * gravity_potential_change_coefficients.coeffs[:, 2, 1]
            )
            gravity_potential_change_coefficients.coeffs[:, 2, 1] += (
                r * angular_velocity_change
            )
        else:
            angular_velocity_change = np.zeros(2)

        g = self.gravitational_acceleration
        vertical_displacement = self._expand_coefficient(
            vertical_displacement_coefficients
        )
        gravity_potential_change = self._expand_coefficient(
            gravity_potential_change_coefficients
        )
        sea_level = (-1 / g) * (g * vertical_displacement + gravity_potential_change)
        sea_level.data[:, :] += mean_sea_level_change - self.ocean_average(sea_level)

        return (
            sea_level,
            vertical_displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    def _iterate_generalised_solver(
        self,
        load,
        mean_sea_level_change,
        /,
        *,
        vertical_displacement_load=None,
        gravitational_potential_load=None,
        angular_velocity_load=None,
        rotational_feedbacks=True,
    ):
        # Given a generalised load, returns the response.

        load_coefficient = self._expand_field(load)

        if vertical_displacement_load is not None:
            vertical_displacement_load_coefficient = self._expand_field(
                vertical_displacement_load
            )
        if gravitational_potential_load is not None:
            gravitational_potential_load_coefficient = self._expand_field(
                gravitational_potential_load
            )

        vertical_displacement_coefficient = load_coefficient.copy()
        gravity_potential_change_coefficient = load_coefficient.copy()

        for l in range(self.lmax + 1):

            vertical_displacement_coefficient.coeffs[
                :, l, :
            ] *= self._displacement_love_number[l]

            gravity_potential_change_coefficient.coeffs[
                :, l, :
            ] *= self._gravitational_potential_love_number[l]

            if vertical_displacement_load is not None:
                vertical_displacement_coefficient.coeffs[:, l, :] += (
                    self._vertical_displacement_to_vertical_displacement_love_number[l]
                    * vertical_displacement_load_coefficient.coeffs[:, l, :]
                )

                gravity_potential_change_coefficient.coeffs[:, l, :] += (
                    self._vertical_displacement_to_gravitational_potential_love_number[
                        l
                    ]
                    * vertical_displacement_load_coefficient.coeffs[:, l, :]
                )

            if gravitational_potential_load is not None:
                vertical_displacement_coefficient.coeffs[:, l, :] += (
                    self._gravitational_potential_to_vertical_displacement_love_number[
                        l
                    ]
                    * gravitational_potential_load_coefficient.coeffs[:, l, :]
                )

                gravity_potential_change_coefficient.coeffs[:, l, :] += (
                    self._gravitational_potential_to_gravitational_potential_love_number[
                        l
                    ]
                    * gravitational_potential_load_coefficient.coeffs[:, l, :]
                )

        if rotational_feedbacks:
            r = self._rotation_factor
            i = self._inertia_factor
            kt = self._gravitational_potential_tidal_love_number[2]
            ht = self._vertical_displacement_tidal_love_number[2]
            g = r / (1 - r * i * kt)
            h = 1 / (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            vertical_displacement_coefficient.coeffs[:, 2, 1] += (
                ht * g * i * gravity_potential_change_coefficient.coeffs[:, 2, 1]
            )

            if angular_velocity_load is not None:
                vertical_displacement_coefficient.coeffs[:, 2, 1] += (
                    ht * r * h * angular_velocity_load[:]
                    + kt * g * h * angular_velocity_load[:]
                )

            gravity_potential_change_coefficient.coeffs[:, 2, 1] += (
                kt * g * i * gravity_potential_change_coefficient.coeffs[:, 2, 1]
            )

            if angular_velocity_load is not None:
                gravity_potential_change_coefficient.coeffs[:, 2, 1] += (
                    kt * g * h * angular_velocity_load[:]
                )

            angular_velocity_change = (
                i * gravity_potential_change_coefficient.coeffs[:, 2, 1]
            )

            if angular_velocity_load is not None:
                angular_velocity_change += h * angular_velocity_load[:]

            gravity_potential_change_coefficient.coeffs[:, 2, 1] += (
                r * angular_velocity_change
            )
        else:
            angular_velocity_change = np.zeros(2)

        g = self.gravitational_acceleration
        vertical_displacement = self._expand_coefficient(
            vertical_displacement_coefficient
        )
        gravity_potential_change = self._expand_coefficient(
            gravity_potential_change_coefficient
        )
        sea_level_change = (-1 / g) * (
            g * vertical_displacement + gravity_potential_change
        )
        sea_level_change.data[:, :] += mean_sea_level_change - self.ocean_average(
            sea_level_change
        )

        return (
            sea_level_change,
            vertical_displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    # --------------------------------------------------------#
    #                       Public methods                    #
    # --------------------------------------------------------#

    def plot(self, f, *args, **kwargs):
        """
        Plot a field showing coastlines.
        """
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        plt.pcolor(f.lons(), f.lats(), f.data, *args, **kwargs)
        fmax = 1.2 * np.nanmax(np.abs(f.data))
        plt.clim([-fmax, fmax])
        return ax

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

    def evaluate_point(self, f, latitude, longitude):
        """Evaluate a function at a given point.

        Args:
            f (SHGrid): Function to evaluate.
            latitude (float): Latitude of the point.
            longitude (float): Longitude of the point.

        Returns:
            float: Value of the function at the point.
        """
        f_lm = self._expand_field(f)
        return pysh.expand.MakeGridPoint(
            f_lm.coeffs, latitude, longitude + 180
        ).flatten()[0]

    def zero_grid(self):
        """Return a grid of zeros."""
        return SHGrid.from_zeros(
            lmax=self.lmax,
            grid=self.grid,
            sampling=self._sampling,
            extend=self.extend,
        )

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

    def set_background_state_from_ice_ng(self, /, *, version=7, date=0):
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
        background_ice_thickness, background_sea_level = (
            ice_ng.get_ice_thickness_and_sea_level(
                date,
                self.lmax,
                grid=self.grid,
                sampling=self._sampling,
                extend=self.extend,
            )
        )
        self.background_ice_thickness = background_ice_thickness / self.length_scale
        self.background_sea_level = background_sea_level / self.length_scale

    def solver(
        self, direct_load, /, *, rotational_feedbacks=True, rtol=1.0e-6, verbose=False
    ):
        """
        Returns the solution to the fingerprint problem for a given direct load.

        Args:
            direct_load (SHGrid): The direct load.
            rotational_feedbacks (bool): If true, rotational feedbacks included.
            rtol (float): Relative tolerance used in assessing convergence of iterations.
            verbose (bool): If true, information on iterations printed.

        Returns:
            ResponseField: Instance of the class containing the vertical_displacement,
                gravity potential perturbation, rotational perturbation, and
                sea level change.

        Notes:
            If rotational feedbacks are included, the potential perturbation
            is that of gravity, this being a sum of the gravitational and
            centrifugal perturbations.
        """
        assert self._check_field(direct_load)
        mean_sea_level_change = -self.integrate(direct_load) / (
            self.water_density * self.ocean_area
        )
        load = (
            direct_load
            + self.water_density * self.ocean_function * mean_sea_level_change
        )

        err = 1
        count = 0
        while err > rtol:
            (
                sea_level,
                vertical_displacement,
                gravity_potential_change,
                angular_velocity_change,
            ) = self._iterate_solver(
                load,
                mean_sea_level_change,
                rotational_feedbacks=rotational_feedbacks,
            )
            load_new = (
                direct_load + self.water_density * self.ocean_function * sea_level
            )
            err = np.max(np.abs((load_new - load).data)) / np.max(np.abs(load.data))
            load = load_new
            if verbose:
                count += 1
                print(f"Iteration = {count}, relative error = {err:6.4e}")

        return (
            sea_level,
            vertical_displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    def generalised_solver(
        self,
        direct_load,
        /,
        *,
        vertical_displacement_load=None,
        gravitational_potential_load=None,
        angular_velocity_load=None,
        rotational_feedbacks=True,
        rtol=1.0e-6,
        verbose=False,
    ):
        """
        Returns the solution to the generalised fingerprint problem for a given generalised load.

        Args:
            generalised_load (ResponseFields object): The generalised load as an instance of ResponseFields (for now)
            rotational_feedbacks (bool): If true, rotational feedbacks included.
            rtol (float): Relative tolerance used in assessing convergence of iterations.
            verbose (bool): If true, information on iterations printed.

        Returns:
            ResponseField: Instance of the class containing the vertical_displacement,
                gravity potential perturbation, rotational perturbation, and
                sea level change.

        Notes:
            If rotational feedbacks are included, the potential perturbation
            is that of gravity, this being a sum of the gravitational and
            centrifugal perturbations.
        """

        assert self._check_field(direct_load)

        if vertical_displacement_load is not None:
            assert self._check_field(vertical_displacement_load)

        if gravitational_potential_load is not None:
            assert self._check_field(gravitational_potential_load)

        mean_sea_level_change = -self.integrate(direct_load) / (
            self.water_density * self.ocean_area
        )
        load = (
            direct_load
            + self.water_density * self.ocean_function * mean_sea_level_change
        )

        err = 1
        count = 0
        while err > rtol:
            (
                sea_level_change,
                vertical_displacement,
                gravity_potential_change,
                angular_velocity_change,
            ) = self._iterate_generalised_solver(
                load,
                mean_sea_level_change,
                vertical_displacement_load=vertical_displacement_load,
                gravitational_potential_load=gravitational_potential_load,
                angular_velocity_load=angular_velocity_load,
                rotational_feedbacks=rotational_feedbacks,
            )
            load_new = (
                direct_load
                + self.water_density * self.ocean_function * sea_level_change
            )
            err = np.max(np.abs((load_new - load).data)) / np.max(np.abs(load.data))
            if verbose:
                count += 1
                print(f"Iteration = {count}, relative error = {err:6.4e}")
            load = load_new

        return (
            sea_level_change,
            vertical_displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    def gravity_potential_change_to_gravitational_potential(
        self, response, rotational_feedbacks=True
    ):
        """Converts the gravity potential within a ResponseField to the gravitational potential.

        Args:
            response (ResponseField): The response field to convert.

        Returns:
            phi (SHGrid): The gravitational potential.
        """
        if not rotational_feedbacks:
            return response.phi
        phi_lm = self._expand_field(response.phi).coeffs
        phi_lm[:, 2, 1] -= self._rotation_factor * response.omega
        phi = self._expand_coefficient(
            SHCoeffs.from_array(
                phi_lm, normalization=self.normalization, csphase=self.csphase
            )
        )
        return phi

    def gravitational_potential_to_gravity_potential_change(
        self, response, rotational_feedbacks=True
    ):
        """Converts the gravitational potential within a ResponseField to the gravity potential.

        Args:
            response (ResponseField): The response field to convert.

        Returns:
            gamma (SHGrid): The gravity potential.
        """
        if not rotational_feedbacks:
            return response.phi
        phi_lm = self._expand_field(response.phi).coeffs
        phi_lm[:, 2, 1] += self._rotation_factor * response.omega
        gamma = self._expand_coefficient(
            SHCoeffs.from_array(
                phi_lm, normalization=self.normalization, csphase=self.csphase
            )
        )
        return gamma

    def ocean_mask(self, value=np.nan):
        """Return a mask over the oceans.

        Args:
            value (float): Value to set the mask over the oceans.

        Returns:
            SHGrid: Mask over the oceans.
        """
        return SHGrid.from_array(
            np.where(self.ocean_function.data > 0, 1, value), grid=self.grid
        )

    def ice_mask(self, value=np.nan):
        """Return a mask over the ice.

        Args:
            value (float): Value to set the mask over the ice.

        Returns:
            SHGrid: Mask over the ice.
        """
        return SHGrid.from_array(
            np.where(self.background_ice_thickness.data > 0, 1, value), grid=self.grid
        )

    def land_mask(self, value=np.nan):
        """Return mask over the land.

        Args:
            value (float): Value to set the mask over the land.

        Returns:
            SHGrid: Mask over the land.
        """
        return SHGrid.from_array(
            np.where(self.ocean_function.data == 0, 1, value), grid=self.grid
        )

    def northern_hemisphere_mask(self, value=np.nan):
        """Return mask over the northern hemisphere.

        Args:
            value (float): Value to set the mask over the northern hemisphere.

        Returns:
            SHGrid: Mask over the northern hemisphere.
        """
        lats, _ = np.meshgrid(
            self.background_ice_thickness.lats(),
            self.background_ice_thickness.lons(),
            indexing="ij",
        )
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid=self.grid)

    def southern_hemisphere_mask(self, value=np.nan):
        """Return mask over the southern hemisphere.

        Args:
            value (float): Value to set the mask over the southern hemisphere.

        Returns:
            SHGrid: Mask over the southern hemisphere.
        """
        lats, _ = np.meshgrid(
            self.background_ice_thickness.lats(),
            self.background_ice_thickness.lons(),
            indexing="ij",
        )
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid=self.grid)

    def altimetery_mask(self, latitude1=-66.0, latitude2=66.0, value=np.nan):
        """Return mask over the altimetry region.

        Args:
            latitude1 (float): Southern latitude of the altimetry region.
            latitude2 (float): Northern latitude of the altimetry region.
            value (float): Value to set the mask over the altimetry region.

        Returns:
            SHGrid: Mask over the altimetry region.
        """
        lats, _ = np.meshgrid(
            self.background_ice_thickness.lats(),
            self.background_ice_thickness.lons(),
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

    def point_load(self, latitude, longitude, amplitude=1):
        """Return a point load with inverse Laplacian smoothing.

        Args:
            latitude (float): Latitude of the point load.
            longitude (float): Longitude of the point load.
            amplitude (float): Amplitude of the load.

        Returns:
            SHGrid: Load associated with the point load.
        """
        theta = 90.0 - latitude
        phi = longitude + 180.0
        point_load_lm = self.zero_coefficients()
        ylm = pysh.expand.spharm(
            point_load_lm.lmax, theta, phi, normalization=self.normalization
        )

        for l in range(0, point_load_lm.lmax + 1):
            point_load_lm.coeffs[0, l, 0] += ylm[0, l, 0]
            for m in range(1, l + 1):
                point_load_lm.coeffs[0, l, m] += ylm[0, l, m]
                point_load_lm.coeffs[1, l, m] += ylm[1, l, m]

        point_load_lm = (1 / self.mean_sea_floor_radius**2) * point_load_lm
        point_load = amplitude * self._expand_coefficient(point_load_lm)

        return point_load

    def load_from_ice_thickness_change(self, ice_thickness_change):
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
            -fraction * self.background_ice_thickness * self.northern_hemisphere_mask(0)
        )
        return self.load_from_ice_thickness_change(ice_thickness_change)

    def southern_hemisphere_load(self, fraction=1):
        """Returns a load associated with melting the given fraction of ice in the northern hemisphere.

        Args:
            fraction (float): Fraction of ice to melt.

        Returns:
            SHGrid: Load associated with melting the given fraction of ice in the northern hemisphere.
        """
        ice_thickness_change = (
            -fraction * self.background_ice_thickness * self.southern_hemisphere_mask(0)
        )
        return self.load_from_ice_thickness_change(ice_thickness_change)

    def sea_level_adjoint_load(self, latitude, longitude):
        """Returns the adjoint loads for a sea level measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields: Adjoint loads for the sea level measurement.
        """
        direct_load = self.point_load(latitude, longitude)
        direct_load_u = self.zero_grid()
        direct_load_phi = self.zero_grid()
        kk = np.zeros(2)
        return ResponseFields(direct_load_u, direct_load_phi, kk, direct_load)

    def vertical_displacement_load(self, latitude, longitude):
        """Returns the adjoint loads for a vertical_displacement measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields: Adjoint loads for the vertical_displacement measurement.
        """
        direct_load = self.zero_grid()
        direct_load_u = -1 * self.point_load(latitude, longitude)
        direct_load_phi = self.zero_grid()
        kk = np.zeros(2)
        return ResponseFields(direct_load_u, direct_load_phi, kk, direct_load)

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
        ph0 = (longitude - 180) * np.pi / 180
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

    def gaussian_averaging_function_components(self, r, latitude, longitude, cut=False):

        pass
