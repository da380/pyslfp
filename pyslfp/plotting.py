"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Union
import numpy as np
from pyshtools import SHGrid

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes


def plot(
    f: SHGrid,
    /,
    *,
    projection: ccrs.Projection = ccrs.Robinson(),
    contour: bool = False,
    cmap: str = "RdBu",
    coasts: bool = True,
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: bool = False,
    **kwargs,
) -> Tuple[Figure, GeoAxes, Union[QuadMesh, QuadContourSet]]:
    """
    Plots a pyshtools SHGrid object on a map. 🗺️

    This function provides a flexible interface to visualize spherical harmonic
    grid data using cartopy for projections and matplotlib for plotting.

    Args:
        f (SHGrid): The scalar field to be plotted.
        projection (ccrs.Projection): The cartopy projection to be used.
            Defaults to ccrs.Robinson().
        contour (bool): If True, a filled contour plot is created. If False,
            a pcolormesh plot is created. Defaults to False.
        cmap (str): The colormap for the plot. Defaults to 'RdBu'.
        coasts (bool): If True, coastlines are drawn. Defaults to True.
        rivers (bool): If True, major rivers are drawn. Defaults to False.
        borders (bool): If True, country borders are drawn. Defaults to False.
        map_extent (Optional[List[float]]): Sets the longitude and latitude
            range for the plot, given as [lon_min, lon_max, lat_min, lat_max].
            Defaults to None (global extent).
        gridlines (bool): If True, latitude and longitude gridlines are
            included. Defaults to True.
        symmetric (bool): If True, the color scale is set symmetrically
            around zero based on the field's maximum absolute value. This is
            overridden if 'vmin' or 'vmax' are provided in kwargs.
            Defaults to False.
        **kwargs: Additional keyword arguments are forwarded to the underlying
            matplotlib plotting function (ax.pcolormesh or ax.contourf).

    Returns:
        Tuple[Figure, GeoAxes, Union[QuadMesh, QuadContourSet]]:
            A tuple containing the matplotlib Figure, the cartopy GeoAxes,
            and the plot artist object (e.g., QuadMesh or QuadContourSet).
    """

    if not isinstance(f, SHGrid):
        raise ValueError("Scalar field must be of SHGrid type.")

    lons = f.lons()
    lats = f.lats()

    figsize = kwargs.pop("figsize", (10, 8))
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

    if map_extent is not None:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())

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
        data_max = 1.2 * np.nanmax(np.abs(f.data))
        kwargs.setdefault("vmin", -data_max)
        kwargs.setdefault("vmax", data_max)

    levels = kwargs.pop("levels", 10)
    im: Union[QuadMesh, QuadContourSet]

    if contour:
        im = ax.contourf(
            lons,
            lats,
            f.data,
            transform=ccrs.PlateCarree(),
            levels=levels,
            **kwargs,
        )
    else:
        im = ax.pcolormesh(
            lons,
            lats,
            f.data,
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
