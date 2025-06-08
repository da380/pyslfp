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
from pyslfp.physical_parameters import EarthModelParamters

from . import DATADIR


if __name__ == "__main__":
    pass


class FingerPrint(EarthModelParamters):
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
                mass_scale=earth_model_parameters.mass_scale,
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
        self._sea_level = None
        self._ice_thickness = None
        self._ocean_function = None
        self._ocean_area = None

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

    def _iterate_solver(
        self,
        load,
        mean_sea_level_change,
        /,
        *,
        displacement_load=None,
        gravitational_potential_load=None,
        angular_momentum_change=None,
        rotational_feedbacks=True,
    ):
        # Given a set of generalised loads, returns the solid earth deformation
        # and associated sea level change.

        if displacement_load is not None:
            displacement_load_coefficient = self._expand_field(displacement_load)
        if gravitational_potential_load is not None:
            gravitational_potential_load_coefficient = self._expand_field(
                gravitational_potential_load
            )

        displacement_lm = self._expand_field(load)
        gravity_potential_change_lm = displacement_lm.copy()

        for l in range(self.lmax + 1):

            displacement_lm.coeffs[:, l, :] *= self._h[l]
            gravity_potential_change_lm.coeffs[:, l, :] *= self._k[l]

            if displacement_load is not None:
                displacement_lm.coeffs[:, l, :] += (
                    self._h_u[l] * displacement_load_coefficient.coeffs[:, l, :]
                )

                gravity_potential_change_lm.coeffs[:, l, :] += (
                    self._k_u[l] * displacement_load_coefficient.coeffs[:, l, :]
                )

            if gravitational_potential_load is not None:
                displacement_lm.coeffs[:, l, :] += (
                    self._h_phi[l]
                    * gravitational_potential_load_coefficient.coeffs[:, l, :]
                )

                gravity_potential_change_lm.coeffs[:, l, :] += (
                    self._k_phi[l]
                    * gravitational_potential_load_coefficient.coeffs[:, l, :]
                )

        if rotational_feedbacks:
            r = self._rotation_factor
            i = self._inertia_factor
            kt = self._kt[2]
            ht = self._ht[2]
            g = r / (1 - r * i * kt)
            h = 1 / (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            displacement_lm.coeffs[:, 2, 1] += (
                ht * g * i * gravity_potential_change_lm.coeffs[:, 2, 1]
            )

            if angular_momentum_change is not None:
                displacement_lm.coeffs[:, 2, 1] += (
                    ht * r * h * angular_momentum_change[:]
                    + kt * g * h * angular_momentum_change[:]
                )

            gravity_potential_change_lm.coeffs[:, 2, 1] += (
                kt * g * i * gravity_potential_change_lm.coeffs[:, 2, 1]
            )

            if angular_momentum_change is not None:
                gravity_potential_change_lm.coeffs[:, 2, 1] += (
                    kt * g * h * angular_momentum_change[:]
                )

            angular_velocity_change = i * gravity_potential_change_lm.coeffs[:, 2, 1]

            if angular_momentum_change is not None:
                angular_velocity_change += h * angular_momentum_change[:]

            gravity_potential_change_lm.coeffs[:, 2, 1] += r * angular_velocity_change
        else:
            angular_velocity_change = np.zeros(2)

        g = self.gravitational_acceleration
        displacement = self._expand_coefficient(displacement_lm)
        gravity_potential_change = self._expand_coefficient(gravity_potential_change_lm)
        sea_level_change = (-1 / g) * (g * displacement + gravity_potential_change)
        sea_level_change.data[:, :] += mean_sea_level_change - self.ocean_average(
            sea_level_change
        )

        return (
            sea_level_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    # --------------------------------------------------------#
    #                       Public methods                    #
    # --------------------------------------------------------#

    def plot(
        self,
        f,
        /,
        *,
        projection=ccrs.Robinson(),
        contour=False,
        cmap="RdBu",
        coasts=True,
        rivers=False,
        borders=False,
        ocean_projection=False,
        land_projection=False,
        ice_projection=False,
        gridlines=True,
        symmetric=False,
        **kwargs,
    ):
        """
        Return a plot of a scalar field on the spatial grid.

        Args:
            f (SHGrid): Scalar field to be plotted.
            projection: cartopy projection to be used. Default is Robinson.
            contour (bool): If True, a contour plot is made, otherwise a pcolor plot.
            cmap (string): colormap. Default is RdBu.
            coasts (bool): If True, coast lines plotted. Default is True.
            rivers (bool): If True, major rivers plotted. Default is False.
            borders (bool): If True, country borders are plotted. Default is False.
            ocean_projection (bool): If True, values plotted only in oceans. Default is False.
            land_projection (bool): If True, values plotted only on land. Default is False.
            ice_projection (bool): If True, values plotted only over ice sheets. Default is False.
            gridlines (bool): If True, gridlines are included. Default is True.
            symmetric (bool): If True, clim values set symmetrically based on the fields maximum absolute value.
                Option overridden if vmin or vmax are set.
            kwargs: Keyword arguments for forwarding to the plotting functions.


        Raises:
            ValueError: If field is not defined the grid used by the class instance.



        """

        if not self._check_field(f):
            raise ValueError("Field not compatible with fingerprint grid")

        lons = f.lons()
        lats = f.lats()
        data = f.data.copy()

        if ocean_projection:
            data *= self.ocean_projection().data

        if land_projection:
            data *= self.land_projection().data

        if ice_projection:
            data *= self.ice_projection().data

        figsize = kwargs.pop("figsize", (10, 8))
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

        if coasts:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

        if rivers:
            ax.add_feature(cfeature.RIVERS, linewidth=0.8)

        if borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)

        kwargs.setdefault("cmap", cmap)

        lat_interval = kwargs.pop("lat_interval", 30)
        lon_interval = kwargs.pop("lon_interval", 30)

        if symmetric:
            data_max = 1.2 * np.nanmax(np.abs(data))
            kwargs.setdefault("vmin", -data_max)
            kwargs.setdefault("vmax", data_max)

        levels = kwargs.pop("levels", 10)

        if contour:

            im = ax.contourf(
                lons,
                lats,
                data,
                transform=ccrs.PlateCarree(),
                levels=levels,
                **kwargs,
            )

        else:

            im = ax.pcolormesh(
                lons,
                lats,
                data,
                transform=ccrs.PlateCarree(),
                **kwargs,
            )

        if gridlines:
            gl = ax.gridlines(
                linestyle="--",
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
            )

            gl.xlocator = mticker.MultipleLocator(lon_interval)
            gl.ylocator = mticker.MultipleLocator(lat_interval)
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

        return fig, ax, im

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

        Args:
            direct_load (SHGrid): The direct load applied in the problem. Default is None.
            displacement_load (SHGrid): The displacement load applied in the problem. Default is None.
            gravitational_potential_load (SHGrid): The gravitational potential load applied in the
                 problem. The default is None.
            angular_momentum_change (numpy vector): The angular velocity

        Returns:


        Notes:
            If rotational feedbacks are included, the potential perturbation
            is that of gravity, this being a sum of the gravitational and
            centrifugal perturbations.
        """

        loads_present = False

        if direct_load is not None:
            loads_present = True
            assert self._check_field(direct_load)
            mean_sea_level_change = -self.integrate(direct_load) / (
                self.water_density * self.ocean_area
            )
        else:
            direct_load = self.zero_grid()
            mean_sea_level_change = 0

        if displacement_load is not None:
            loads_present = True
            assert self._check_field(displacement_load)

        if gravitational_potential_load is not None:
            loads_present = True
            assert self._check_field(gravitational_potential_load)

        if angular_momentum_change is not None and rotational_feedbacks:
            loads_present = True

        if loads_present is False:
            # Return zero solution if no loads have been set.
            return self.zero_grid(), self.zero_grid(), self.zero_grid(), np.zeros(2)

        load = (
            direct_load
            + self.water_density * self.ocean_function * mean_sea_level_change
        )

        err = 1
        count = 0
        count_print = 0
        while err > rtol:
            (
                sea_level_change,
                displacement,
                gravity_potential_change,
                angular_velocity_change,
            ) = self._iterate_solver(
                load,
                mean_sea_level_change,
                displacement_load=displacement_load,
                gravitational_potential_load=gravitational_potential_load,
                angular_momentum_change=angular_momentum_change,
                rotational_feedbacks=rotational_feedbacks,
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
        self, gravity_potential_change, angular_velocity_change
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
        gravitaty_potential_change_lm.coeffs[:, 2, 1] += (
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
        # phi = longitude + 180.0
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

    def sea_level_point_measurement_adjoint_loads(self, latitude, longitude):
        """Returns the adjoint loads for a sea level measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields:
        """
        direct_load = self.point_load(latitude, longitude)
        return direct_load, None, None, None

    def displacement_load(self, latitude, longitude):
        """Returns the adjoint loads for a displacement measurement at a given location.

        Args:
            latitude (float): Latitude of the measurement.
            longitude (float): Longitude of the measurement.

        Returns:
            ResponseFields: Adjoint loads for the displacement measurement.
        """
        displacement_load = -1 * self.point_load(latitude, longitude)
        return None, displacement_load, None, None

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

    def gaussian_averaging_function_components(self, r, latitude, longitude, cut=False):

        pass
