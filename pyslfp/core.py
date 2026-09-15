"""
Core physics definitions for the pyslfp library.

This module contains the fundamental Earth parameters, non-dimensionalization
schemes, and elastic Love numbers required for gravitationally consistent
sea-level fingerprinting.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pyshtools as sh
from pyshtools import SHCoeffs, SHGrid
from pyshtools.utils import DHaj
from planetmodel import units
from planetmodel.units import Scales

from pyslfp.love_numbers.table import LoveNumbers

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
# The one definition of G: planetmodel's (CODATA 2018), so that Love numbers
# computed from a model and the rotational feedback use the same value.
GRAVITATIONAL_CONSTANT: float = units.G_SI

# =====================================================================


@dataclass(frozen=True, kw_only=True)
class EarthModelParameters:
    """
    Stores Earth model parameters and handles non-dimensionalization.

    This is a frozen dataclass; all properties are strictly immutable
    to ensure physical consistency across the solver lifecycle.

    The three base scales define a planetmodel `Scales` (length, mass and
    time), held as `scales`, and every non-dimensional value is its raw
    value divided by the scale factor of its dimensions.

    The radius of the solid surface, the surface gravity and G describe the
    body the Love numbers were computed for, and `EarthModel` takes them
    from its Love numbers through `with_body`; the sea level equation's
    adjoint is exact only when the two agree.

    Attributes:
        length_scale (float): The length scale used for non-dimensionalization. Default is 1.0.
        density_scale (float): The density scale used for non-dimensionalization. Default is 1.0.
        time_scale (float): The time scale used for non-dimensionalization. Default is 1.0.
        scales (Scales): The planetmodel scales the three define.
        raw_equatorial_radius (float): Earth's equatorial radius in meters.
        raw_polar_radius (float): Earth's polar radius in meters.
        raw_mean_radius (float): Earth's mean radius in meters.
        raw_mean_sea_floor_radius (float): Mean radius of the solid Earth surface in meters.
        raw_mass (float): Total mass of the Earth in kilograms.
        raw_gravitational_acceleration (float): Surface gravity in m/s^2.
        raw_gravitational_constant (float): G in m^3 kg^-1 s^-2. Defaults to
            planetmodel's CODATA 2018 value.
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
    raw_gravitational_constant: float = GRAVITATIONAL_CONSTANT
    raw_equatorial_moment_of_inertia: float = EQUATORIAL_MOMENT_OF_INERTIA
    raw_polar_moment_of_inertia: float = POLAR_MOMENT_OF_INERTIA
    raw_rotation_frequency: float = ROTATION_FREQUENCY
    raw_water_density: float = WATER_DENSITY
    raw_ice_density: float = ICE_DENSITY

    # The scales, and the non-dimensional values, set in __post_init__
    scales: Scales = field(init=False)

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
        """Builds the scales and locks in every non-dimensional value."""
        scales = Scales(
            length=self.length_scale,
            mass=self.density_scale * self.length_scale**3,
            time=self.time_scale,
        )

        def put(name: str, value: float) -> None:
            object.__setattr__(self, name, value)

        def nd(raw: float, dims: units.Dimensions) -> float:
            return raw / scales.factor(dims)

        moment_of_inertia = units.MASS * units.LENGTH**2

        put("scales", scales)
        put("equatorial_radius", nd(self.raw_equatorial_radius, units.LENGTH))
        put("polar_radius", nd(self.raw_polar_radius, units.LENGTH))
        put("mean_radius", nd(self.raw_mean_radius, units.LENGTH))
        put("mean_sea_floor_radius", nd(self.raw_mean_sea_floor_radius, units.LENGTH))
        put("mass", nd(self.raw_mass, units.MASS))
        put(
            "gravitational_acceleration",
            nd(self.raw_gravitational_acceleration, units.GRAVITY),
        )
        put(
            "gravitational_constant",
            nd(self.raw_gravitational_constant, units.GRAVITATIONAL_CONSTANT),
        )
        put(
            "equatorial_moment_of_inertia",
            nd(self.raw_equatorial_moment_of_inertia, moment_of_inertia),
        )
        put(
            "polar_moment_of_inertia",
            nd(self.raw_polar_moment_of_inertia, moment_of_inertia),
        )
        put("rotation_frequency", nd(self.raw_rotation_frequency, units.FREQUENCY))
        put("water_density", nd(self.raw_water_density, units.DENSITY))
        put("ice_density", nd(self.raw_ice_density, units.DENSITY))

        put(
            "rotation_factor",
            np.sqrt((4 * np.pi) / 15.0)
            * self.rotation_frequency
            * self.mean_sea_floor_radius**2,
        )
        put(
            "inertia_factor",
            np.sqrt(5 / (12 * np.pi))
            * self.rotation_frequency
            * self.mean_sea_floor_radius**3
            / (
                self.gravitational_constant
                * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
            ),
        )

    def with_body(self, love_numbers: LoveNumbers, /) -> EarthModelParameters:
        """
        The same parameters with the raw sea-floor radius, surface gravity
        and G replaced by those of a Love number table, where the table
        records them; a value the table does not know (NaN) is kept.
        """
        si = love_numbers.in_si()
        changes = {}
        for raw, value in (
            ("raw_mean_sea_floor_radius", si.radius),
            ("raw_gravitational_acceleration", si.surface_gravity),
            ("raw_gravitational_constant", si.G),
        ):
            if np.isfinite(value):
                changes[raw] = float(value)
        return replace(self, **changes)

    @staticmethod
    def from_defaults() -> EarthModelParameters:
        """
        Returns parameters using a standard non-dimensionalisation scheme.
        This is defined such that the Earth's radius, density, and
        surface gravitational acceleration are equal to one.
        """

        length_scale = MEAN_RADIUS
        density_scale = 3 * MASS / (4 * np.pi * length_scale**3)
        time_scale = np.sqrt(length_scale / GRAVITATIONAL_ACCELERATION)

        return EarthModelParameters(
            length_scale=length_scale,
            density_scale=density_scale,
            time_scale=time_scale,
        )


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
        love_numbers: Optional[Union[LoveNumbers, str, Path]] = None,
        love_number_file: Optional[Union[str, Path]] = None,
        grid: str = "DH",
        extend: bool = True,
    ) -> None:
        """
        Initializes an EarthModel configuration.

        Args:
            lmax (int): The maximum spherical harmonic degree.
            parameters (Optional[EarthModelParameters]): The Earth's physical scales.
                If None, standard non-dimensionalized parameters are generated.
                The sea-floor radius, surface gravity and G are then taken from
                the Love numbers, see `EarthModelParameters.with_body`.
            love_numbers (Optional[LoveNumbers | str | Path]): The Love numbers,
                as a `LoveNumbers` table (from `LoveNumbers.from_model`, say) or
                the path of a file written by `LoveNumbers.write`. If None, the
                shipped PREM table is used, downloaded on first use. The table
                must be real and reach `lmax`.
            love_number_file (Optional[str | Path]): The older name of
                `love_numbers` for a path; one of the two may be given.
            grid (str): The pyshtools grid format ("DH", "DH2" or "GLQ").
                Defaults to "DH".
            extend (bool): If True, grids include the redundant 360 degree longitude
                column (and, for DH grids, the 90 degree south latitude row). Set to
                False to work with native non-extended grids (e.g. high-resolution
                topography data). Defaults to True.
        """
        if love_numbers is not None and love_number_file is not None:
            raise ValueError("give love_numbers or love_number_file, not both")
        source = love_number_file if love_numbers is None else love_numbers
        if source is None:
            table = LoveNumbers.default()
        elif isinstance(source, LoveNumbers):
            table = source
        else:
            table = LoveNumbers.from_file(source)
        if table.is_complex:
            raise ValueError("the sea level equation takes real Love numbers")

        self._lmax = lmax
        base = parameters or EarthModelParameters.from_defaults()
        self._parameters = base.with_body(table)
        self._love_number_table = table
        self._love_numbers = table.converted(self._parameters.scales).truncated(lmax)

        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1

        self._extend: bool = extend

        # Internal parameters (do not change)
        self._normalization: str = "ortho"
        self._csphase: int = 1

        # Precompute the quadrature weights used for surface integration
        self._integration_weights = self._compute_integration_weights()

    @staticmethod
    def from_defaults(*, lmax: int = 256) -> EarthModel:
        """
        Returns the default Earth model, based on PREM with standard non-dimensionalisations.

        Args:
            lmax (int): Truncation degree for discretisation. Defaults to 256.
        """
        return EarthModel(lmax)

    @staticmethod
    def from_planet_model(
        model,
        lmax: int,
        /,
        *,
        parameters: Optional[EarthModelParameters] = None,
        mesh=None,
        ngll: int = 5,
        eps: float = 1e-8,
        grid: str = "DH",
        extend: bool = True,
    ) -> EarthModel:
        """
        Returns an Earth model whose Love numbers are computed from a planetmodel
        model, such as `planetmodel.PREM(ocean=False)`.

        The model's surface must be solid. Its radius, surface gravity and G
        become the body of the parameters. `mesh`, `ngll` and `eps` are passed
        to `LoveNumbers.from_model`; the mesh is sized for `lmax` by default.

        Args:
            model: A planetmodel model holding density and elastic moduli.
            lmax (int): Truncation degree for discretisation.
            parameters (Optional[EarthModelParameters]): The scales; see `EarthModel`.
            mesh: A planetmodel `RadialMesh` over the model, or None.
            ngll (int): Nodes per element of the default mesh.
            eps (float): The truncation level of the degree-l solutions.
            grid (str): The pyshtools grid format.
            extend (bool): Whether grids include the redundant longitude column.
        """
        love_numbers = LoveNumbers.from_model(
            model, lmax, mesh=mesh, ngll=ngll, eps=eps
        )
        return EarthModel(
            lmax,
            parameters=parameters,
            love_numbers=love_numbers,
            grid=grid,
            extend=extend,
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
    def sampling(self) -> int:
        """
        Returns the sampling value of the SH grid.
        """
        return self._sampling

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
        """The elastic Love numbers, in the model's non-dimensional units
        and truncated at its lmax."""
        return self._love_numbers

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
            f.lmax == self.lmax
            and f.grid == self.grid
            and f.extend == self.extend
            and getattr(f, "sampling", self._sampling) == self._sampling
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

        The integral is evaluated with the quadrature weights of the grid
        (Driscoll and Healy weights for DH grids, Gauss-Legendre weights
        for GLQ grids), which is exact for band-limited fields and
        identical to reading off the degree-zero coefficient.

        Args:
            f: The function to integrate, represented as an SHGrid object.

        Returns:
            The integral of the function over the surface.
        """
        self.check_field(f)
        return self._integrate_data(f.data)

    def _integrate_data(self, data: np.ndarray) -> float:
        """
        Integrates raw grid values laid out as for this model's grid.

        Internal fast path used by the solvers; no compatibility checks.
        """
        if self._extend:
            rows = data[:-1, :-1] if self._grid == "DH" else data[:, :-1]
        else:
            rows = data
        return float(self._integration_weights @ rows.sum(axis=1))

    def _compute_integration_weights(self) -> np.ndarray:
        """
        Returns the latitudinal quadrature weights w_j such that the surface
        integral of a field equals sum_j w_j sum_i f_ji over the unextended grid.
        """
        radius_squared = self.parameters.mean_sea_floor_radius**2
        if self._grid == "DH":
            nlat = 2 * self.lmax + 2
            nlon = nlat * self._sampling
            return 2.0 * np.sqrt(2.0) * np.pi * radius_squared * DHaj(nlat) / nlon
        elif self._grid == "GLQ":
            _, weights = sh.expand.SHGLQ(self.lmax)
            nlon = 2 * self.lmax + 1
            return 2.0 * np.pi * radius_squared * weights / nlon
        else:
            raise ValueError(f"Unsupported grid type: {self._grid}")

    def with_degree(self, lmax) -> EarthModel:
        """
        Returns a version of the EarthModel that is identical but for a change
        in the associated truncation degree.
        """
        return EarthModel(
            lmax,
            parameters=self.parameters,
            love_numbers=self._love_number_table,
            grid=self.grid_name,
            extend=self.extend,
        )
