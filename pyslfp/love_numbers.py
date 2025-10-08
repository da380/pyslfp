"""
Module for the LoveNumbers class, which handles loading and processing
of elastic Love numbers for glacial isostatic adjustment models.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt


import pyshtools as sh

from .config import DATADIR

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .physical_parameters import EarthModelParameters


class LoveNumbers:
    """
    A class to load, non-dimensionalize, and provide elastic Love numbers.
    """

    def __init__(
        self,
        lmax: int,
        params: EarthModelParameters,
        /,
        *,
        file: Optional[str] = None,
    ):
        """
        Initializes the LoveNumbers object.

        Args:
            lmax: The maximum spherical harmonic degree to load.
            params: An EarthModelParameters instance containing the
                non-dimensionalization scales.
            file: Path to the Love number data file. If None, a default
                file based on the PREM model is used.
        """

        if file is None:
            file = DATADIR + "/love_numbers/PREM_4096.dat"

        data = np.loadtxt(file)
        data_degree = len(data[:, 0]) - 1

        if lmax > data_degree:
            raise ValueError(
                f"lmax ({lmax}) is larger than the maximum degree "
                f"in the Love number file ({data_degree})."
            )

        # Non-dimensionalize the Love numbers using the provided parameters
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

        self._params = params
        self._lmax = lmax

    @property
    def h(self) -> np.ndarray:
        """The total displacement Love numbers, h."""
        return self._h

    @property
    def k(self) -> np.ndarray:
        """The total gravitational Love numbers, k."""
        return self._k

    @property
    def ht(self) -> np.ndarray:
        """The tidal displacement Love numbers, h_t."""
        return self._ht

    @property
    def kt(self) -> np.ndarray:
        """The tidal gravitational Love numbers, k_t."""
        return self._kt

    def _greens_function(
        self, angle: float, /, *, lmax: int = None, displacement: bool = True
    ) -> float:
        """
        Returns the value of the Green's function for displacement or potential for a given angular separation
        between the load and observation point.

        Args:
            angle: The angular separation between the load and observation point.
            lmax: The maximum spherical harmonic degree used in the expansion.The
                default is to use the maximum degree associated with the class.
            displacement: If True, the Greens' function for displacement is returned.
                If False, the Greens' function for potential is returned.

        Returns:
            The value of the Green's function.
        """

        if lmax is None:
            lmax = self._lmax

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
        return np.sum(terms)

    def displacement_greens_function(
        self, angle: float, /, *, lmax: int = None
    ) -> float:
        """
        Returns the value of the Green's function for displacement for a given angular separation
        between the load and observation point.

        Args:
            angle: The angular separation between the load and observation point.
            lmax: The maximum spherical harmonic degree used in the expansion.The
                default is to use the maximum degree associated with the class.

            Reuturns:
                The value of the Green's function.
        """
        return self._greens_function(angle, lmax=lmax, displacement=True)

    def potential_greens_function(self, angle: float, /, *, lmax: int = None) -> float:
        """
        Returns the value of the Green's function for potential for a given angular separation
        between the load and observation point.

        Args:
            angle: The angular separation between the load and observation point.
            lmax: The maximum spherical harmonic degree used in the expansion.The
                default is to use the maximum degree associated with the class.

            Reuturns:
                The value of the Green's function.
        """
        return self._greens_function(angle, lmax=lmax, displacement=False)

    def plot_greens_functions(
        self, lmax: Optional[int] = None, n_points: int = 181
    ) -> tuple:
        """
        Generates and returns the figure and axes for a plot of the
        displacement and potential Green's functions.

        This method is intended for quick visualization. The returned objects
        can be used for further customization.

        Args:
            lmax: The maximum spherical harmonic degree for the calculation.
                  Defaults to the lmax of the class instance.
            n_points: The number of points to calculate between 0 and 180
                      degrees for a smooth curve.

        Returns:
            A tuple containing the Matplotlib figure and axes objects
            (fig, axes).
        """
        if lmax is None:
            lmax = self._lmax

        # 1. Set up the angular separation axis
        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)

        # 2. Calculate the Green's function values
        g_disp = [
            self.displacement_greens_function(angle, lmax=lmax) for angle in angles_rad
        ]
        g_pot = [
            self.potential_greens_function(angle, lmax=lmax)
            / self._params.gravitational_acceleration
            for angle in angles_rad
        ]

        # 3. Create the plot with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True, layout="tight")

        # Top subplot: Displacement
        axes[0].plot(angles_deg, g_disp, "b-")
        axes[0].set_title("Displacement Green's function", fontsize=20)
        axes[0].set_ylabel("Non-dimensional length per unit mass", fontsize=20)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].tick_params(axis="x", labelsize=15)
        axes[0].tick_params(axis="y", labelsize=15)

        # Bottom subplot: Potential
        axes[1].plot(angles_deg, g_pot, "r-")
        axes[1].set_title("Potential Green's function", fontsize=20)
        axes[1].set_ylabel("Non-dimensional length per unit mass", fontsize=20)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].tick_params(axis="x", labelsize=15)
        axes[1].tick_params(axis="y", labelsize=15)

        # Shared x-axis label
        axes[1].set_xlabel("Angular Separation (degrees)", fontsize=20)
        axes[1].set_xlim(0, 180)

        # axes[0].set_yscale("symlog", linthresh=1)
        # axes[1].set_yscale("symlog", linthresh=1)

        # 4. Return the objects instead of showing the plot
        return fig, axes

    def _add_break_marks(self, ax1, ax2):
        """Helper function to draw break marks between two axes."""
        d = 0.015  # size of the marks in figure coordinates
        kwargs = dict(transform=ax1.get_figure().transFigure, color="k", clip_on=False)

        pos1 = ax1.get_position()
        pos2 = ax2.get_position()

        line_x = (pos1.x1 + pos2.x0) / 2

        # Draw top and bottom marks
        ax1.get_figure().add_artist(
            plt.Line2D([line_x - d, line_x + d], [pos1.y0 - d, pos1.y0 + d], **kwargs)
        )
        ax1.get_figure().add_artist(
            plt.Line2D([line_x - d, line_x + d], [pos1.y1 - d, pos1.y1 + d], **kwargs)
        )

    def plot_greens_functions_split(
        self,
        split_angle: float = 20.0,
        lmax: Optional[int] = None,
        n_points: int = 300,
    ) -> tuple:
        """
        Generates a 'broken axis' plot of the Green's functions to show
        detail for both near and far fields using different linear scales.
        """
        if lmax is None:
            lmax = self._lmax

        # 1. Generate the full range of data
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

        # 2. Create subplot grid using constrained_layout
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            gridspec_kw={
                "width_ratios": [1, 3],
                "wspace": 0.05,
            },  # Increased wspace slightly
            constrained_layout=True,
        )

        # 3. Use figure-level titles and labels for better layout management
        fig.supxlabel("Angular Separation (degrees)", fontsize=20)

        # Set a y-label for each row
        axes[0, 0].set_ylabel("Displacement", fontsize=20)
        axes[1, 0].set_ylabel("Geoid Anomaly", fontsize=20)

        # 4. Filter and plot data
        near_mask = angles_deg < split_angle
        far_mask = angles_deg >= split_angle

        # --- Top Row: Displacement ---
        axes[0, 0].plot(angles_deg[near_mask], g_disp[near_mask], "b-")
        axes[0, 1].plot(angles_deg[far_mask], g_disp[far_mask], "b-")
        axes[0, 0].set_title("Near Field", fontsize=20)
        axes[0, 1].set_title("Far Field (Zoomed)", fontsize=20)

        # --- Bottom Row: Geoid Anomaly ---
        axes[1, 0].plot(angles_deg[near_mask], g_geoid[near_mask], "r-")
        axes[1, 1].plot(angles_deg[far_mask], g_geoid[far_mask], "r-")

        # 5. Format axes
        for i in range(2):
            axes[i, 0].set_xlim(0, split_angle)
            axes[i, 1].set_xlim(split_angle, 180)

        for i in range(2):
            for j in range(2):
                axes[i, j].tick_params(axis="x", labelsize=15)
                axes[i, j].tick_params(axis="y", labelsize=15)

        for ax in axes[0, :]:
            ax.tick_params(axis="x", labelbottom=False)

        for ax in axes.flat:
            ax.grid(True, linestyle=":", alpha=0.6)

        return fig, axes
