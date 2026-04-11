"""
Core physics definitions for the pyslfp library.

This module contains the fundamental Earth parameters, non-dimensionalization
schemes, and elastic Love numbers required for sea-level fingerprinting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt

# Clean import from our new data layer
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
    Strictly immutable to ensure physical consistency.
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

    @staticmethod
    def from_standard_non_dimensionalisation() -> "EarthModelParameters":
        """
        Returns parameters using a standard non-dimensionalisation scheme based
        on the mean radius of the Earth, the density of water, and the length
        of an hour.
        """
        return EarthModelParameters(
            length_scale=6371000.0, density_scale=1000.0, time_scale=3600
        )


class LoveNumbers:
    """
    Loads and non-dimensionalizes elastic Love numbers.
    """

    def __init__(
        self,
        lmax: int,
        params: EarthModelParameters,
        /,
        *,
        file: Optional[str] = None,
    ):
        if file is None:
            ensure_data("LOVE_NUMBERS")
            file = str(DATADIR / "love_numbers" / "PREM_4096.dat")

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if lmax > data_degree:
            raise ValueError(
                f"lmax ({lmax}) exceeds Love number file max degree ({data_degree})."
            )

        self.lmax = lmax
        self._params = params

        # Non-dimensionalize Love numbers using the immutable parameters
        self.h_u = data[: lmax + 1, 1] * params.load_scale / params.length_scale
        self.k_u = (
            data[: lmax + 1, 2]
            * params.load_scale
            / params.gravitational_potential_scale
        )
        self.h_phi = data[: lmax + 1, 3] * params.load_scale / params.length_scale
        self.k_phi = (
            data[: lmax + 1, 4]
            * params.load_scale
            / params.gravitational_potential_scale
        )

        self.h = self.h_u + self.h_phi
        self.k = self.k_u + self.k_phi
        self.ht = (
            data[: lmax + 1, 5]
            * params.gravitational_potential_scale
            / params.length_scale
        )
        self.kt = data[: lmax + 1, 6]

    def displacement_greens_function(self, angle: float, lmax: int = None) -> float:
        return self._greens_function(angle, lmax=lmax, displacement=True)

    def potential_greens_function(self, angle: float, lmax: int = None) -> float:
        return self._greens_function(angle, lmax=lmax, displacement=False)

    def _greens_function(
        self, angle: float, lmax: int = None, displacement: bool = True
    ) -> float:
        if lmax is None:
            lmax = self.lmax

        x = np.cos(angle)
        ps = sh.legendre.PLegendre(lmax, x)
        degrees = np.arange(lmax + 1)
        love_numbers = self.h[: lmax + 1] if displacement else self.k[: lmax + 1]
        smoothing = np.exp(-10 * (degrees**2) / lmax**2)

        terms = (
            (2 * degrees + 1)
            * love_numbers
            * smoothing
            * ps
            / (4 * np.pi * self._params.mean_sea_floor_radius**2)
        )
        return float(np.sum(terms))

    def plot_greens_functions(
        self, lmax: Optional[int] = None, n_points: int = 181
    ) -> tuple:
        """Generates a quick visualization of the Green's functions."""
        if lmax is None:
            lmax = self.lmax

        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)
        g_disp = [
            self.displacement_greens_function(angle, lmax=lmax) for angle in angles_rad
        ]
        g_pot = [
            self.potential_greens_function(angle, lmax=lmax)
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
        self, split_angle: float = 20.0, lmax: Optional[int] = None, n_points: int = 300
    ) -> tuple:
        """Generates a broken axis plot to show detail for near and far fields."""
        if lmax is None:
            lmax = self.lmax

        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)
        g_disp = np.array(
            [self.displacement_greens_function(a, lmax=lmax) for a in angles_rad]
        )
        g_geoid = np.array(
            [
                -self.potential_greens_function(a, lmax=lmax)
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
    The unified physics configuration object for the pyslfp library.

    This replaces multiple inheritance by safely composing the non-dimensionalized
    Earth parameters and the corresponding Love numbers into a single object
    that can be passed to solvers and operators.
    """

    def __init__(
        self,
        lmax: int,
        parameters: Optional[EarthModelParameters] = None,
        love_number_file: Optional[str] = None,
    ):
        self.lmax = lmax
        self.parameters = (
            parameters or EarthModelParameters.from_standard_non_dimensionalisation()
        )
        self.love_numbers = LoveNumbers(lmax, self.parameters, file=love_number_file)
