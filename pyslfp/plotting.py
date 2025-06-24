"""
Module for plotting functions
"""

import numpy as np
from pyshtools import SHGrid
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot(
    f,
    /,
    *,
    projection=ccrs.Robinson(),
    contour=False,
    cmap="RdBu",
    coasts=True,
    rivers=False,
    borders=False,
    map_extent=None,
    gridlines=True,
    symmetric=False,
    **kwargs,
):
    """
    Return a plot of a SHGrid object.

    Args:
        f (SHGrid): Scalar field to be plotted.
        projection: cartopy projection to be used. Default is Robinson.
        contour (bool): If True, a contour plot is made, otherwise a pcolor plot.
        cmap (string): colormap. Default is RdBu.
        coasts (bool): If True, coast lines plotted. Default is True.
        rivers (bool): If True, major rivers plotted. Default is False.
        borders (bool): If True, country borders are plotted. Default is False.
        map_extent ([float]): Sets the (lon, lat) range for plotting.
            Tuple of [lon_min, lon_max, lat_min, lat_max]. Default is None.
        gridlines (bool): If True, gridlines are included. Default is True.
        symmetric (bool): If True, clim values set symmetrically based on the fields maximum absolute value.
            Option overridden if vmin or vmax are set.
        kwargs: Keyword arguments for forwarding to the plotting functions.
    """

    if not isinstance(f, SHGrid):
        raise ValueError("Scalar field is not of SHGrid type.")

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
