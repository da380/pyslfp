"""
Abstract base classes for surface mass and ice models.

This module defines the required interface for any data provider
aiming to supply background states (ice thickness and topography)
to the pyslfp physical engine.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path

import numpy as np
from pyshtools import SHGrid, SHCoeffs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

from pyslfp.plot import plot


class BaseIceModel(ABC):
    """
    Abstract base class for all spatial ice and topography models.

    Any custom model (e.g., ICE-7G, BedMachine, AWIC, synthetic loads) must
    inherit from this class and implement the `get_ice_thickness_and_sea_level`
    method to be fully compatible with the EarthState container.
    """

    def __init__(self, /, *, length_scale: float = 1.0) -> None:
        """
        Initializes the base ice model.

        Args:
            length_scale (float): The scaling factor to non-dimensionalize
                the spatial outputs. Defaults to 1.0 (returns raw meters).
        """
        self._length_scale = length_scale

    @property
    def length_scale(self) -> float:
        """The scaling factor used to non-dimensionalize output grids."""
        return self._length_scale

    @abstractmethod
    def get_ice_thickness_and_sea_level(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> Tuple[SHGrid, SHGrid]:
        """
        Evaluates the model at a specific time and spatial resolution.

        Args:
            date (float): The time in kiloyears before present (ka).
            lmax (int): The maximum spherical harmonic degree.
            grid (str): The pyshtools grid format (e.g., "DH", "GLQ").
            sampling (int): The grid sampling factor (1 for standard, 2 for oversampled).
            extend (bool): Whether to extend the grid to include the poles.

        Returns:
            Tuple[SHGrid, SHGrid]: The (ice_thickness, sea_level) grids,
            scaled by the model's `length_scale`.
        """
        pass

    def get_ice_thickness(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the scaled ice thickness for a given date."""
        ice_thickness, _ = self.get_ice_thickness_and_sea_level(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return ice_thickness

    def get_sea_level(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the scaled sea level for a given date."""
        _, sea_level = self.get_ice_thickness_and_sea_level(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return sea_level

    def animate(
        self,
        output_file: str,
        /,
        *,
        field: str = "ice_thickness",
        start_date_ka: float = 26.0,
        end_date_ka: float = 0.0,
        num_frames: int = 261,
        fps: int = 15,
        lmax: int = 180,
        **plot_kwargs,
    ) -> None:
        """
        Generates and saves an animation of this ice model over time.

        Args:
            output_file (str): Path for the output video (e.g., 'anim.mp4').
            field (str): Data field to animate ('ice_thickness' or 'sea_level').
            start_date_ka (float): Start date in ka.
            end_date_ka (float): End date in ka.
            num_frames (int): Total number of frames in the animation.
            fps (int): Frames per second for the output video.
            lmax (int): Max spherical harmonic degree for the data grids.
            **plot_kwargs: Additional keyword arguments passed to the plot function.
        """
        print(f"Initializing animation for {self.__class__.__name__}...")
        valid_fields = ("ice_thickness", "sea_level")
        if field not in valid_fields:
            raise ValueError(f"Field must be one of {valid_fields}, not '{field}'.")

        dates = np.linspace(start_date_ka, end_date_ka, num_frames)

        def get_data_for_date(date: float):
            method_name = f"get_{field}"
            return getattr(self, method_name)(date, lmax)

        print("Generating initial frame...")
        initial_data = get_data_for_date(dates[0])

        if "symmetric" not in plot_kwargs and field == "sea_level":
            plot_kwargs["symmetric"] = True

        # Use the newly updated plot function from plotting.py
        ax, artist = plot(initial_data, **plot_kwargs)
        fig = ax.figure

        if getattr(artist, "colorbar", None):
            unit_label = (
                "Non-dimensional Units" if self.length_scale != 1.0 else "Meters"
            )
            artist.colorbar.set_label(unit_label)

        title = ax.set_title(f"Date: {dates[0]:.2f} ka")

        def update(frame_num: int):
            current_date = dates[frame_num]
            print(
                f"  -> Processing frame {frame_num + 1}/{num_frames} (Date: {current_date:.2f} ka)"
            )
            new_data = get_data_for_date(current_date)
            artist.set_array(new_data.data.ravel())
            title.set_text(f"Date: {current_date:.2f} ka")
            return [artist, title]

        print("Creating animation object...")
        ani = FuncAnimation(fig, func=update, frames=num_frames, blit=False)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving animation to '{output_file}'... (This may take a while)")
        ani.save(output_file, writer="ffmpeg", fps=fps, dpi=150)
        print("Done!")
        plt.close(fig)

    def animate_joint(
        self,
        output_file: str,
        /,
        *,
        start_date_ka: float = 26.0,
        end_date_ka: float = 0.0,
        num_frames: int = 261,
        fps: int = 15,
        lmax: int = 180,
        projection: ccrs.Projection = ccrs.Robinson(),
        ice_plot_kwargs: dict = None,
        sl_plot_kwargs: dict = None,
    ) -> None:
        """
        Generates a side-by-side animation of ice thickness and sea level.

        Args:
            output_file (str): Path for the output video (e.g., 'joint_anim.mp4').
            start_date_ka (float): Start date in ka.
            end_date_ka (float): End date in ka.
            num_frames (int): Total number of frames in the animation.
            fps (int): Frames per second for the output video.
            lmax (int): Max spherical harmonic degree for the grids.
            projection (ccrs.Projection): Cartopy projection for both maps.
            ice_plot_kwargs (dict, optional): Kwargs for the ice thickness plot.
            sl_plot_kwargs (dict, optional): Kwargs for the sea level plot.
        """
        print(f"Initializing joint animation for {self.__class__.__name__}...")

        dates = np.linspace(start_date_ka, end_date_ka, num_frames)
        initial_ice, initial_sl = self.get_ice_thickness_and_sea_level(dates[0], lmax)
        lons, lats = initial_ice.lons(), initial_ice.lats()

        ice_kwargs = {"cmap": "Blues", "vmin": 0}
        if ice_plot_kwargs:
            ice_kwargs.update(ice_plot_kwargs)

        sl_kwargs = {"cmap": "RdBu_r"}
        if sl_plot_kwargs:
            sl_kwargs.update(sl_plot_kwargs)

        if "vmin" in sl_kwargs and "vmax" not in sl_kwargs:
            sl_kwargs["vmax"] = -sl_kwargs["vmin"]
        elif "vmax" in sl_kwargs and "vmin" not in sl_kwargs:
            sl_kwargs["vmin"] = -sl_kwargs["vmax"]

        print("Generating initial frame...")
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": projection},
            layout="constrained",
        )

        def setup_ax(ax: GeoAxes, title: str):
            ax.set_title(title, fontsize=14)
            ax.coastlines()
            ax.gridlines(linestyle="--", alpha=0.5)
            ax.set_global()

        unit_label = "ND" if self.length_scale != 1.0 else "m"

        setup_ax(ax1, "Ice Thickness")
        artist_ice = ax1.pcolormesh(
            lons, lats, initial_ice.data, transform=ccrs.PlateCarree(), **ice_kwargs
        )
        cbar_ice = fig.colorbar(
            artist_ice, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cbar_ice.set_label(f"Ice Thickness ({unit_label})")

        setup_ax(ax2, "Sea Level")
        artist_sl = ax2.pcolormesh(
            lons, lats, initial_sl.data, transform=ccrs.PlateCarree(), **sl_kwargs
        )
        cbar_sl = fig.colorbar(
            artist_sl, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cbar_sl.set_label(f"Sea Level relative to present ({unit_label})")

        main_title = fig.suptitle(f"Date: {dates[0]:.2f} ka", fontsize=16)

        def update(frame_num: int):
            current_date = dates[frame_num]
            print(
                f"  -> Processing frame {frame_num + 1}/{num_frames} (Date: {current_date:.2f} ka)"
            )
            new_ice, new_sl = self.get_ice_thickness_and_sea_level(current_date, lmax)
            artist_ice.set_array(new_ice.data.ravel())
            artist_sl.set_array(new_sl.data.ravel())
            main_title.set_text(f"Date: {current_date:.2f} ka")
            return [artist_ice, artist_sl, main_title]

        print("Creating animation object...")
        ani = FuncAnimation(fig, func=update, frames=num_frames, blit=False)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving animation to '{output_file}'... (This may take a while)")
        ani.save(output_file, writer="ffmpeg", fps=fps, dpi=180)
        print("Done!")
        plt.close(fig)

def apply_cosine_taper(
    grid: SHGrid,
    target_lmax: int,
    l_start: int,
    /,
    *,
    grid_format: str = "DH",
    extend: bool = True,
) -> SHGrid:
    """
    Applies a cosine taper (Tukey window) in the spherical harmonic domain 
    to smoothly truncate a high-resolution grid down to a target lmax.
    This prevents the Gibbs phenomenon (spatial ringing) caused by hard truncation.

    Args:
        grid (SHGrid): The high-resolution input spatial grid.
        target_lmax (int): The final, lower maximum degree to truncate to.
        l_start (int): The spherical harmonic degree where the taper begins.
        grid_format (str): The output pyshtools grid format (e.g., "DH", "GLQ").
        extend (bool): Whether to extend the output grid to include the poles.

    Returns:
        SHGrid: A new, smoothed spatial grid evaluated at the target_lmax.
    """
    # 1. Transform the spatial grid to the spherical harmonic domain
    coeffs = grid.expand()
    current_lmax = coeffs.lmax

    # If the grid is already at or below the target, just pad/truncate and return
    if current_lmax <= target_lmax:
        return coeffs.expand(grid=grid_format, extend=extend, lmax=target_lmax)

    # 2. Extract the raw spherical harmonic coefficients (shape: [2, lmax+1, lmax+1])
    coeff_array = coeffs.to_array().copy()

    # 3. Apply the cosine taper weights
    if target_lmax > l_start:
        for l in range(l_start, target_lmax + 1):
            weight = 0.5 * (1.0 + np.cos(np.pi * (l - l_start) / (target_lmax - l_start)))
            coeff_array[:, l, :] *= weight

    # 4. Truncate the coefficient array exactly at target_lmax
    truncated_array = coeff_array[:, :target_lmax + 1, :target_lmax + 1]

    # 5. Reconstruct the spatial grid at the new, lower lmax
    new_coeffs = SHCoeffs.from_array(
        truncated_array, 
        normalization=coeffs.normalization, 
        csphase=coeffs.csphase
    )
    
    # Setting lmax explicitly during expand ensures the spatial grid matches target_lmax
    return new_coeffs.expand(grid=grid_format, extend=extend, lmax=target_lmax)