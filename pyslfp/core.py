"""
Core physics definitions for the pyslfp library.

This module contains the fundamental Earth parameters, non-dimensionalization
schemes, and elastic Love numbers required for gravitationally consistent
sea-level fingerprinting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pyshtools as sh
from pyshtools import SHCoeffs, SHGrid
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


from pyslfp.data import DATADIR, ensure_data

# =====================================================================
# Default Earth Model Physical Parameters (PREM and standard values)
# =====================================================================

EQUATORIAL_RADIUS: float = 6378137.0
POLAR_RADIUS: float = 6356752.0
MEAN_RADIUS: float = 6371000.0
MEAN_SEA_FLOOR_RADIUS: float = 6368000.0
MASS: float = 5.974e24
GRAVITATIONAL_ACCELERATION: float = 9.825652323
EQUATORIAL_MOMENT_OF_INERTIA: float = 8.0096e37
POLAR_MOMENT_OF_INERTIA: float = 8.0359e37
ROTATION_FREQUENCY: float = 7.27220521664304e-05
WATER_DENSITY: float = 1000.0
ICE_DENSITY: float = 917.0

# =====================================================================


@dataclass(frozen=True, kw_only=True)
class EarthModelParameters:
    """
    Stores Earth model parameters and handles non-dimensionalization.

    This is a frozen dataclass; all properties are strictly immutable
    to ensure physical consistency across the solver lifecycle.

    Attributes:
        length_scale (float): The length scale used for non-dimensionalization. Default is 1.0.
        density_scale (float): The density scale used for non-dimensionalization. Default is 1.0.
        time_scale (float): The time scale used for non-dimensionalization. Default is 1.0.
        raw_equatorial_radius (float): Earth's equatorial radius in meters.
        raw_polar_radius (float): Earth's polar radius in meters.
        raw_mean_radius (float): Earth's mean radius in meters.
        raw_mean_sea_floor_radius (float): Mean radius of the solid Earth surface in meters.
        raw_mass (float): Total mass of the Earth in kilograms.
        raw_gravitational_acceleration (float): Surface gravity in m/s^2.
        raw_equatorial_moment_of_inertia (float): Equatorial moment of inertia in kg*m^2.
        raw_polar_moment_of_inertia (float): Polar moment of inertia in kg*m^2.
        raw_rotation_frequency (float): Earth's rotation frequency in rad/s.
        raw_water_density (float): Density of ocean water in kg/m^3.
        raw_ice_density (float): Density of ice in kg/m^3.
    """

    # Base scales
    length_scale: float = 1.0
    density_scale: float = 1.0
    time_scale: float = 1.0

    # Raw physical inputs (with defaults)
    raw_equatorial_radius: float = EQUATORIAL_RADIUS
    raw_polar_radius: float = POLAR_RADIUS
    raw_mean_radius: float = MEAN_RADIUS
    raw_mean_sea_floor_radius: float = MEAN_SEA_FLOOR_RADIUS
    raw_mass: float = MASS
    raw_gravitational_acceleration: float = GRAVITATIONAL_ACCELERATION
    raw_equatorial_moment_of_inertia: float = EQUATORIAL_MOMENT_OF_INERTIA
    raw_polar_moment_of_inertia: float = POLAR_MOMENT_OF_INERTIA
    raw_rotation_frequency: float = ROTATION_FREQUENCY
    raw_water_density: float = WATER_DENSITY
    raw_ice_density: float = ICE_DENSITY

    # Derived non-dimensional properties initialized automatically
    mass_scale: float = field(init=False)
    frequency_scale: float = field(init=False)
    load_scale: float = field(init=False)
    velocity_scale: float = field(init=False)
    acceleration_scale: float = field(init=False)
    gravitational_potential_scale: float = field(init=False)
    moment_of_inertia_scale: float = field(init=False)

    equatorial_radius: float = field(init=False)
    polar_radius: float = field(init=False)
    mean_radius: float = field(init=False)
    mean_sea_floor_radius: float = field(init=False)
    mass: float = field(init=False)
    gravitational_acceleration: float = field(init=False)
    gravitational_constant: float = field(init=False)
    equatorial_moment_of_inertia: float = field(init=False)
    polar_moment_of_inertia: float = field(init=False)
    rotation_frequency: float = field(init=False)
    water_density: float = field(init=False)
    ice_density: float = field(init=False)

    # Quantities useful in calculations
    rotation_factor: float = field(init=False)
    inertia_factor: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculates and locks in all derived scales and parameters."""

        # Base scales
        object.__setattr__(
            self, "mass_scale", self.density_scale * self.length_scale**3
        )
        object.__setattr__(self, "frequency_scale", 1.0 / self.time_scale)
        object.__setattr__(self, "load_scale", self.mass_scale / self.length_scale**2)
        object.__setattr__(self, "velocity_scale", self.length_scale / self.time_scale)
        object.__setattr__(
            self, "acceleration_scale", self.velocity_scale / self.time_scale
        )
        object.__setattr__(
            self,
            "gravitational_potential_scale",
            self.acceleration_scale * self.length_scale,
        )
        object.__setattr__(
            self, "moment_of_inertia_scale", self.mass_scale * self.length_scale**2
        )

        # Non-dimensional physical constants
        object.__setattr__(
            self, "equatorial_radius", self.raw_equatorial_radius / self.length_scale
        )
        object.__setattr__(
            self, "polar_radius", self.raw_polar_radius / self.length_scale
        )
        object.__setattr__(
            self, "mean_radius", self.raw_mean_radius / self.length_scale
        )
        object.__setattr__(
            self,
            "mean_sea_floor_radius",
            self.raw_mean_sea_floor_radius / self.length_scale,
        )
        object.__setattr__(self, "mass", self.raw_mass / self.mass_scale)
        object.__setattr__(
            self,
            "gravitational_acceleration",
            self.raw_gravitational_acceleration / self.acceleration_scale,
        )

        g_nd = 6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3
        object.__setattr__(self, "gravitational_constant", g_nd)

        object.__setattr__(
            self,
            "equatorial_moment_of_inertia",
            self.raw_equatorial_moment_of_inertia / self.moment_of_inertia_scale,
        )
        object.__setattr__(
            self,
            "polar_moment_of_inertia",
            self.raw_polar_moment_of_inertia / self.moment_of_inertia_scale,
        )
        object.__setattr__(
            self,
            "rotation_frequency",
            self.raw_rotation_frequency / self.frequency_scale,
        )
        object.__setattr__(
            self, "water_density", self.raw_water_density / self.density_scale
        )
        object.__setattr__(
            self, "ice_density", self.raw_ice_density / self.density_scale
        )

        object.__setattr__(
            self,
            "rotation_factor",
            np.sqrt((4 * np.pi) / 15.0)
            * self.rotation_frequency
            * self.mean_sea_floor_radius**2,
        )

        object.__setattr__(
            self,
            "inertia_factor",
            np.sqrt(5 / (12 * np.pi))
            * self.rotation_frequency
            * self.mean_sea_floor_radius**3
            / (
                self.gravitational_constant
                * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            ),
        )

    @staticmethod
    def from_standard_non_dimensionalisation() -> "EarthModelParameters":
        """
        Returns parameters using a standard non-dimensionalisation scheme.

        This scheme is based on the mean radius of the Earth, the density of water,
        and the length of an hour.

        Returns:
            EarthModelParameters: An instance populated with standard scales.
        """
        return EarthModelParameters(
            length_scale=6371000.0, density_scale=1000.0, time_scale=3600
        )


class LoveNumbers:
    """
    Loads, stores, and non-dimensionalizes elastic Love numbers.

    This class handles the ingestion of Love number data files and computes
    the appropriate non-dimensional forms required for the spherical harmonic
    solutions of the Sea Level Equation.
    """

    def __init__(
        self,
        lmax: int,
        params: EarthModelParameters,
        /,
        *,
        file: Optional[str] = None,
    ) -> None:
        """
        Initialize the Love numbers for a specific Earth model.

        Args:
            lmax (int): The maximum spherical harmonic degree.
            params (EarthModelParameters): The non-dimensionalized parameters of the Earth.
            file (Optional[str]): Path to a custom Love number `.dat` file. If None,
                it uses the default PREM_4096 dataset, downloading it if necessary.

        Raises:
            ValueError: If the requested lmax exceeds the maximum degree in the data file.
        """
        if file is None:
            ensure_data("LOVE_NUMBERS")
            file = str(DATADIR / "love_numbers" / "PREM_4096.dat")

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if lmax > data_degree:
            raise ValueError(
                f"lmax ({lmax}) exceeds Love number file max degree ({data_degree})."
            )

        self._lmax = lmax
        self._params = params

        # Non-dimensionalize Love numbers using the immutable parameters
        self._h_u = data[: lmax + 1, 1] * params.load_scale / params.length_scale
        self._k_u = (
            data[: lmax + 1, 2]
            * params.load_scale
            / params.gravitational_potential_scale
        )
        self._h_phi = data[: lmax + 1, 3] * params.load_scale / params.length_scale
        self._k_phi = (
            data[: lmax + 1, 4]
            * params.load_scale
            / params.gravitational_potential_scale
        )

        self._h = self._h_u + self._h_phi
        self._k = self._k_u + self._k_phi

        self._ht = (
            data[: lmax + 1, 5]
            * params.gravitational_potential_scale
            / params.length_scale
        )
        self._kt = data[: lmax + 1, 6]

    # ---------------------------------------------------------#
    #                     Properties                           #
    # ---------------------------------------------------------#

    @property
    def lmax(self) -> int:
        """The maximum spherical harmonic degree."""
        return self._lmax

    @property
    def h_u(self) -> np.ndarray:
        """Non-dimensional vertical displacement Love number for direct mass loading."""
        return self._h_u

    @property
    def k_u(self) -> np.ndarray:
        """Non-dimensional gravitational potential Love number for direct mass loading."""
        return self._k_u

    @property
    def h_phi(self) -> np.ndarray:
        """Non-dimensional vertical displacement Love number for potential loading."""
        return self._h_phi

    @property
    def k_phi(self) -> np.ndarray:
        """Non-dimensional gravitational potential Love number for potential loading."""
        return self._k_phi

    @property
    def h(self) -> np.ndarray:
        """Total displacement Love number (degree-dependent)."""
        return self._h

    @property
    def k(self) -> np.ndarray:
        """Total gravitational potential Love number (degree-dependent)."""
        return self._k

    @property
    def ht(self) -> np.ndarray:
        """Tidal (rotational) displacement Love number."""
        return self._ht

    @property
    def kt(self) -> np.ndarray:
        """Tidal (rotational) gravitational potential Love number."""
        return self._kt

    # ---------------------------------------------------------#
    #                 Green's Functions                        #
    # ---------------------------------------------------------#

    def displacement_greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None
    ) -> float:
        """
        Evaluates the displacement Green's function at a given angular separation.

        Args:
            angle (float): The angular separation in radians.
            lmax (Optional[int]): The maximum degree to include in the summation.
                If None, uses the instance's lmax.

        Returns:
            float: The non-dimensional displacement value.
        """
        return self._greens_function(angle, lmax=lmax, displacement=True)

    def potential_greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None
    ) -> float:
        """
        Evaluates the gravitational potential Green's function at a given angular separation.

        Args:
            angle (float): The angular separation in radians.
            lmax (Optional[int]): The maximum degree to include in the summation.
                If None, uses the instance's lmax.

        Returns:
            float: The non-dimensional gravitational potential value.
        """
        return self._greens_function(angle, lmax=lmax, displacement=False)

    def _greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None, displacement: bool = True
    ) -> float:
        """Internal helper for computing evaluating summed Legendre polynomials."""
        calc_lmax = lmax if lmax is not None else self.lmax

        x = np.cos(angle)
        ps = sh.legendre.PLegendre(calc_lmax, x)
        degrees = np.arange(calc_lmax + 1)

        love_numbers = (
            self.h[: calc_lmax + 1] if displacement else self.k[: calc_lmax + 1]
        )
        smoothing = np.exp(-10 * (degrees**2) / calc_lmax**2)

        terms = (
            (2 * degrees + 1)
            * love_numbers
            * smoothing
            * ps
            / (4 * np.pi * self._params.mean_sea_floor_radius**2)
        )
        return float(np.sum(terms))

    def plot_greens_functions(
        self, /, *, lmax: Optional[int] = None, n_points: int = 181
    ) -> Tuple[Figure, np.ndarray]:
        """
        Generates a quick visualization of the Green's functions.

        Args:
            lmax (Optional[int]): The maximum degree to evaluate.
            n_points (int): The number of points to sample between 0 and 180 degrees.

        Returns:
            Tuple[Figure, np.ndarray]: The matplotlib Figure and Axes objects.
        """
        calc_lmax = lmax if lmax is not None else self.lmax

        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)

        g_disp = [
            self.displacement_greens_function(angle, lmax=calc_lmax)
            for angle in angles_rad
        ]
        g_pot = [
            self.potential_greens_function(angle, lmax=calc_lmax)
            / self._params.gravitational_acceleration
            for angle in angles_rad
        ]

        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True, layout="tight")

        axes[0].plot(angles_deg, g_disp, "b-")
        axes[0].set_title("Displacement Green's function", fontsize=20)
        axes[0].set_ylabel("Non-dimensional length per unit mass", fontsize=20)
        axes[0].grid(True, linestyle=":", alpha=0.6)

        axes[1].plot(angles_deg, g_pot, "r-")
        axes[1].set_title("Potential Green's function", fontsize=20)
        axes[1].set_ylabel("Non-dimensional length per unit mass", fontsize=20)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].set_xlabel("Angular Separation (degrees)", fontsize=20)
        axes[1].set_xlim(0, 180)

        return fig, axes

    def plot_greens_functions_split(
        self,
        /,
        *,
        split_angle: float = 20.0,
        lmax: Optional[int] = None,
        n_points: int = 300,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Generates a broken axis plot to show detail for near and far fields.

        Args:
            split_angle (float): The angle in degrees at which to break the x-axis.
            lmax (Optional[int]): The maximum degree to evaluate.
            n_points (int): The number of points to sample.

        Returns:
            Tuple[Figure, np.ndarray]: The matplotlib Figure and Axes objects.
        """
        calc_lmax = lmax if lmax is not None else self.lmax

        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)

        g_disp = np.array(
            [self.displacement_greens_function(a, lmax=calc_lmax) for a in angles_rad]
        )
        g_geoid = np.array(
            [
                -self.potential_greens_function(a, lmax=calc_lmax)
                / self._params.gravitational_acceleration
                for a in angles_rad
            ]
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            gridspec_kw={"width_ratios": [1, 3], "wspace": 0.05},
            constrained_layout=True,
        )
        fig.supxlabel("Angular Separation (degrees)", fontsize=20)

        axes[0, 0].set_ylabel("Displacement", fontsize=20)
        axes[1, 0].set_ylabel("Geoid Anomaly", fontsize=20)

        near_mask = angles_deg < split_angle
        far_mask = angles_deg >= split_angle

        axes[0, 0].plot(angles_deg[near_mask], g_disp[near_mask], "b-")
        axes[0, 1].plot(angles_deg[far_mask], g_disp[far_mask], "b-")
        axes[0, 0].set_title("Near Field", fontsize=20)
        axes[0, 1].set_title("Far Field (Zoomed)", fontsize=20)

        axes[1, 0].plot(angles_deg[near_mask], g_geoid[near_mask], "r-")
        axes[1, 1].plot(angles_deg[far_mask], g_geoid[far_mask], "r-")

        for i in range(2):
            axes[i, 0].set_xlim(0, split_angle)
            axes[i, 1].set_xlim(split_angle, 180)
            for j in range(2):
                axes[i, j].grid(True, linestyle=":", alpha=0.6)

        for ax in axes[0, :]:
            ax.tick_params(axis="x", labelbottom=False)

        return fig, axes


class EarthModel:
    """
    The unified physics configuration object for the library.

    This object encapsulates the non-dimensionalized Earth parameters and
    the corresponding Love numbers, ensuring physical consistency across
    all calculations and solvers. Also sets values for SH transforms
    and provides a range of utility methods.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        parameters: Optional[EarthModelParameters] = None,
        love_number_file: Optional[str] = None,
        grid: str = "DH",
    ) -> None:
        """
        Initializes an EarthModel configuration.

        Args:
            lmax (int): The maximum spherical harmonic degree.
            parameters (Optional[EarthModelParameters]): The Earth's physical scales.
                If None, standard non-dimensionalized parameters are generated.
            love_number_file (Optional[str]): Path to a custom Love number file.
        """
        self._lmax = lmax
        self._parameters = (
            parameters or EarthModelParameters.from_standard_non_dimensionalisation()
        )
        self._love_numbers = LoveNumbers(lmax, self._parameters, file=love_number_file)

        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1

        # Internal parameters (do not change)
        self._extend: bool = True
        self._normalization: str = "ortho"
        self._csphase: int = 1

        self._normalization: str = "ortho"
        self._csphase: int = 1

        # Precompute the spherical integration factor
        self._integration_factor = (
            np.sqrt(4 * np.pi) * self.parameters.mean_sea_floor_radius**2
        )

    @property
    def lmax(self) -> int:
        """The maximum spherical harmonic degree."""
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
    def grid_name(self):
        """
        Returns the name of the grid corrected for sampling differences.
        """
        return self.grid if self._sampling == 1 else "DH2"

    @property
    def extend(self) -> bool:
        """True if grid extended to include 360 degree longitude."""
        return self._extend

    @property
    def parameters(self) -> EarthModelParameters:
        """The fundamental, non-dimensionalized Earth parameters."""
        return self._parameters

    @property
    def love_numbers(self) -> LoveNumbers:
        """The elastic Love numbers for this Earth model."""
        return self._love_numbers

    @classmethod
    def default(cls, lmax: int, /) -> "EarthModel":
        """
        Generates an EarthModel instance using standard PREM parameters
        and the default PREM Love numbers file.

        Args:
            lmax (int): The maximum spherical harmonic degree.

        Returns:
            EarthModel: A fully configured EarthModel ready for solvers.
        """
        return cls(lmax)

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
