"""
Module for loading and interpolating the ICE-5G, ICE-6G, and ICE-7G
global ice history models.
"""

from __future__ import annotations
from typing import Tuple
import xarray as xr
from pyshtools import SHGrid
import numpy as np
import bisect
from scipy.interpolate import RegularGridInterpolator
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes


from .config import DATADIR
from .plotting import plot


class IceModel(Enum):
    """An enumeration for the different ICE-NG model versions."""

    ICE5G = 5
    ICE6G = 6
    ICE7G = 7


class IceNG:
    """
    A data loader for the ICE-5G, ICE-6G, and ICE-7G glacial isostatic
    adjustment models.

    This class provides methods to retrieve ice thickness, topography, and
    sea level for a given date, interpolating between the model's time
    slices as needed.
    """

    def __init__(self, /, *, version: IceModel = IceModel.ICE7G):
        """
        Args:
            version: The ice model version to use. Defaults to ICE-7G.
        """
        self._version = version
        # Set the densities of ice and water in kg/m^3.
        self.ice_density = 917.0
        self.water_density = 1028.0

    def _date_to_file(self, date: float) -> str:
        """Converts a date into the appropriate data file name."""
        if self._version in [IceModel.ICE6G, IceModel.ICE7G]:
            date_string = f"{int(date):d}" if date == int(date) else f"{date:3.1f}"
        else:
            date_string = f"{date:04.1f}"

        if self._version == IceModel.ICE7G:
            return DATADIR + "/ice7g/I7G_NA.VM7_1deg." + date_string + ".nc"
        elif self._version == IceModel.ICE6G:
            return DATADIR + "/ice6g/I6_C.VM5a_1deg." + date_string + ".nc"
        else:
            return DATADIR + "/ice5g/ice5g_v1.2_" + date_string + "k_1deg.nc"

    def _find_files(self, date: float) -> Tuple[str, str, float]:
        """
        Given a date, finds the data files that bound it and the fraction
        for linear interpolation.
        """
        if self._version in [IceModel.ICE6G, IceModel.ICE7G]:
            dates = np.append(np.linspace(0, 21, 43), np.linspace(22, 26, 5))
        else:
            dates = np.append(np.linspace(0, 17, 35), np.linspace(18, 21, 4))

        i = bisect.bisect_left(dates, date)
        if i == 0:
            date1 = date2 = dates[0]
        elif i == len(dates):
            date1 = date2 = dates[i - 1]
        else:
            date1 = dates[i - 1]
            date2 = dates[i]

        fraction = (date2 - date) / (date2 - date1) if date1 != date2 else 0.0

        return self._date_to_file(date1), self._date_to_file(date2), fraction

    def _get_time_slice(
        self, file: str, lmax: int, /, *, grid: str, sampling: int, extend: bool
    ) -> Tuple[SHGrid, SHGrid]:
        """Reads a data file and interpolates fields onto a pyshtools grid."""
        data = xr.open_dataset(file)
        ice_thickness = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )
        topography = SHGrid.from_zeros(
            lmax, grid=grid, sampling=sampling, extend=extend
        )

        if self._version == IceModel.ICE5G:
            ice_var, topo_var = "sftgit", "orog"
            lon_var = "long"
        else:
            ice_var, topo_var = "stgit", "Topo"
            lon_var = "lon"

        ice_thickness_function = RegularGridInterpolator(
            (data.lat.values, data[lon_var].values),
            data[ice_var].values,
            bounds_error=False,
            fill_value=None,
        )
        topography_function = RegularGridInterpolator(
            (data.lat.values, data[lon_var].values),
            data[topo_var].values,
            bounds_error=False,
            fill_value=None,
        )

        lats, lons = np.meshgrid(
            ice_thickness.lats(), ice_thickness.lons(), indexing="ij"
        )
        ice_thickness.data = ice_thickness_function((lats, lons))
        topography.data = topography_function((lats, lons))

        return ice_thickness, topography

    def get_ice_thickness_and_topography(
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
        Returns the ice thickness and topography (in meters) for a given date.

        If the date does not exist within the data set, linear interpolation is
        used. If the date is out of range, constant extrapolation is applied.

        Args:
            date: The date in thousands of years before present (ka).
            lmax: The maximum spherical harmonic degree for the output grids.
            grid: The `pyshtools` grid type. Defaults to 'DH'.
            sampling: The `pyshtools` grid sampling. Defaults to 1.
            extend: `pyshtools` grid extension option. Defaults to True.

        Returns:
            A tuple containing the ice thickness and topography as `SHGrid` objects.
        """
        file1, file2, fraction = self._find_files(date)
        if file1 == file2:
            ice_thickness, topography = self._get_time_slice(
                file1, lmax, grid=grid, sampling=sampling, extend=extend
            )
        else:
            ice_thickness1, topography1 = self._get_time_slice(
                file1, lmax, grid=grid, sampling=sampling, extend=extend
            )
            ice_thickness2, topography2 = self._get_time_slice(
                file2, lmax, grid=grid, sampling=sampling, extend=extend
            )
            ice_thickness = fraction * ice_thickness1 + (1 - fraction) * ice_thickness2
            topography = fraction * topography1 + (1 - fraction) * topography2
        return ice_thickness, topography

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
        Returns the ice thickness and sea level (in meters) for a given date.

        Sea level is computed from topography assuming isostatic balance for
        floating ice shelves.

        Args:
            date: The date in thousands of years before present (ka).
            lmax: The maximum spherical harmonic degree for the output grids.
            grid: The `pyshtools` grid type. Defaults to 'DH'.
            sampling: The `pyshtools` grid sampling. Defaults to 1.
            extend: `pyshtools` grid extension option. Defaults to True.

        Returns:
            A tuple containing the ice thickness and sea level as `SHGrid` objects.
        """
        ice_thickness, topography = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        # Compute the sea level using isostatic balance within ice shelves.
        ice_shelf_thickness = SHGrid.from_array(
            np.where(
                np.logical_and(topography.data < 0, ice_thickness.data > 0),
                ice_thickness.data,
                0,
            ),
            grid=grid,
        )
        sea_level = SHGrid.from_array(
            np.where(
                topography.data < 0,
                -topography.data,
                -topography.data + ice_thickness.data,
            ),
            grid=grid,
        )
        sea_level += self.ice_density * ice_shelf_thickness / self.water_density
        return ice_thickness, sea_level

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
        """Returns the ice thickness (in meters) for a given date."""
        ice_thickness, _ = self.get_ice_thickness_and_topography(
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
        """Returns the sea level (in meters) for a given date."""
        _, sea_level = self.get_ice_thickness_and_sea_level(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return sea_level

    def get_topography(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ) -> SHGrid:
        """Returns the topography (in meters) for a given date."""
        _, topography = self.get_ice_thickness_and_topography(
            date, lmax, grid=grid, sampling=sampling, extend=extend
        )
        return topography

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
    ):
        """
        Generates and saves an animation of this ice model over time.

        Args:
            output_file (str): Path for the output video (e.g., 'anim.mp4').
            field (str): Data field to animate. Must be one of
                'ice_thickness', 'sea_level', or 'topography'.
            start_date_ka (float): Start date in thousands of years before present.
            end_date_ka (float): End date in thousands of years before present.
            num_frames (int): Total number of frames in the animation.
            fps (int): Frames per second for the output video.
            lmax (int): Max spherical harmonic degree for the data grids.
            **plot_kwargs: Additional keyword arguments passed to the plot
                function (e.g., cmap, projection).
        """
        print(f"Initializing animation for {self._version.name}...")
        valid_fields = ("ice_thickness", "sea_level", "topography")
        if field not in valid_fields:
            raise ValueError(f"Field must be one of {valid_fields}, not '{field}'.")

        dates = np.linspace(start_date_ka, end_date_ka, num_frames)

        # Helper to get data by calling the appropriate method of this instance
        def get_data_for_date(date: float):
            method_name = f"get_{field}"
            return getattr(self, method_name)(date, lmax)

        # Create the initial plot (the first frame)
        print("Generating initial frame...")
        initial_data = get_data_for_date(dates[0])

        if "symmetric" not in plot_kwargs and field == "sea_level":
            plot_kwargs["symmetric"] = True

        fig, ax, artist = plot(initial_data, **plot_kwargs)

        cbar = fig.colorbar(
            artist, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7
        )
        cbar.set_label("Meters")
        title = ax.set_title(f"Date: {dates[0]:.2f} ka")

        # Define the update function for the animation
        def update(frame_num: int):
            current_date = dates[frame_num]
            print(
                f"  -> Processing frame {frame_num + 1}/{num_frames} (Date: {current_date:.2f} ka)"
            )

            new_data = get_data_for_date(current_date)
            artist.set_array(new_data.data.ravel())
            title.set_text(f"Date: {current_date:.2f} ka")
            return [artist, title]

        # Create and save the animation
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
    ):
        """
        Generates a side-by-side animation of ice thickness and sea level.

        This function creates a video with two synchronized maps, allowing for
        a direct comparison of how ice sheet changes drive sea level response.

        Args:
            output_file (str): Path for the output video (e.g., 'joint_anim.mp4').
            start_date_ka (float): Start date in thousands of years before present.
            end_date_ka (float): End date in thousands of years before present.
            num_frames (int): Total number of frames in the animation.
            fps (int): Frames per second for the output video.
            lmax (int): Max spherical harmonic degree for the data grids.
            projection (ccrs.Projection): The cartopy projection to use for both maps.
            ice_plot_kwargs (dict, optional): Kwargs for the ice thickness plot,
                passed to `ax.pcolormesh`. E.g., {'cmap': 'Blues', 'vmax': 4000}.
            sl_plot_kwargs (dict, optional): Kwargs for the sea level plot.
                E.g., {'cmap': 'RdBu_r', 'vmin': -150, 'vmax': 150}.
        """
        print(f"Initializing joint animation for {self._version.name}...")

        # 1. Setup dates and initial data
        dates = np.linspace(start_date_ka, end_date_ka, num_frames)
        initial_ice, initial_sl = self.get_ice_thickness_and_sea_level(dates[0], lmax)
        lons, lats = initial_ice.lons(), initial_ice.lats()

        # 2. Setup plotting kwargs with sensible defaults
        ice_kwargs = {"cmap": "Blues", "vmin": 0}
        if ice_plot_kwargs:
            ice_kwargs.update(ice_plot_kwargs)

        sl_kwargs = {"cmap": "RdBu_r"}
        if sl_plot_kwargs:
            sl_kwargs.update(sl_plot_kwargs)
        # Ensure sea level colormap is symmetric if not specified
        if "vmin" in sl_kwargs and "vmax" not in sl_kwargs:
            sl_kwargs["vmax"] = -sl_kwargs["vmin"]
        elif "vmax" in sl_kwargs and "vmin" not in sl_kwargs:
            sl_kwargs["vmin"] = -sl_kwargs["vmax"]

        # 3. Create the figure with two subplots
        print("Generating initial frame...")
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": projection},
            layout="constrained",
        )

        def setup_ax(ax: GeoAxes, title: str):
            """Helper to format each map axis."""
            ax.set_title(title, fontsize=14)
            ax.coastlines()
            ax.gridlines(linestyle="--", alpha=0.5)
            ax.set_global()

        # 4. Plot initial data on each subplot
        setup_ax(ax1, "Ice Thickness")
        artist_ice = ax1.pcolormesh(
            lons, lats, initial_ice.data, transform=ccrs.PlateCarree(), **ice_kwargs
        )
        cbar_ice = fig.colorbar(
            artist_ice, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cbar_ice.set_label("Ice Thickness (m)")

        setup_ax(ax2, "Sea Level")
        artist_sl = ax2.pcolormesh(
            lons, lats, initial_sl.data, transform=ccrs.PlateCarree(), **sl_kwargs
        )
        cbar_sl = fig.colorbar(
            artist_sl, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8
        )
        cbar_sl.set_label("Sea Level relative to present (m)")

        main_title = fig.suptitle(f"Date: {dates[0]:.2f} ka", fontsize=16)

        # 5. Define the animation update function
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

        # 6. Create and save the animation
        print("Creating animation object...")
        ani = FuncAnimation(fig, func=update, frames=num_frames, blit=False)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving animation to '{output_file}'... (This may take a while)")
        ani.save(output_file, writer="ffmpeg", fps=fps, dpi=180)
        print("Done!")
        plt.close(fig)
