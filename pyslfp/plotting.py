"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Any

import numpy as np
from pyshtools import SHGrid

# Directly import the fully-featured corner plot from pygeoinf
from pygeoinf.symmetric_space import sphere

import cartopy.crs as ccrs
from cartopy.crs import Projection
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure


def create_map_figure(
    figsize: Optional[Tuple[float, float]] = None,
    projection: Optional[Projection] = None,
    **kwargs,
) -> Tuple[Figure, GeoAxes]:
    """
    Convenience helper to create a Matplotlib Figure and Cartopy GeoAxes.
    Delegates to pygeoinf's modern constrained_layout engine while keeping
    the pyslfp default of a Robinson projection.
    """
    if projection is None:
        projection = ccrs.Robinson()

    return sphere.create_map_figure(figsize=figsize, projection=projection, **kwargs)


def plot(
    f: SHGrid,
    /,
    *,
    ax: Optional[GeoAxes] = None,
    projection: Optional[Projection] = None,
    contour: bool = False,
    cmap: str = "RdBu",
    coasts: bool = True,  # pyslfp default (pygeoinf is False)
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: bool = False,
    contour_lines: bool = False,
    contour_lines_kwargs: Optional[dict] = None,
    num_levels: int = 10,
    colorbar: bool = True,  # pyslfp default (pygeoinf is False)
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Plots a pyshtools SHGrid object on a map.

    This acts as a transparent wrapper around pygeoinf.sphere.plot. It shares
    the exact same API signature, but injects pyslfp's preferred defaults
    (Robinson projection, enabled coastlines, and an active colorbar).
    """
    if projection is None and ax is None:
        projection = ccrs.Robinson()

    if colorbar and colorbar_kwargs is None:
        colorbar_kwargs = {"orientation": "horizontal", "shrink": 0.7, "pad": 0.05}

    return sphere.plot(
        f,
        ax=ax,
        projection=projection,
        contour=contour,
        cmap=cmap,
        coasts=coasts,
        rivers=rivers,
        borders=borders,
        map_extent=map_extent,
        gridlines=gridlines,
        symmetric=symmetric,
        contour_lines=contour_lines,
        contour_lines_kwargs=contour_lines_kwargs,
        num_levels=num_levels,
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        **kwargs,
    )




def plot_points(
    points: List[Tuple[float, float]],
    /,
    *,
    data: Optional[List[float] | np.ndarray] = None,
    ax: Optional[GeoAxes] = None,
    projection: Optional[Projection] = None,
    cmap: str = "RdBu",
    color: str = "red",
    s: float = 20,
    marker: str = "o",
    edgecolors: str = "none",
    zorder: int = 5,
    colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Plots discrete observation points (e.g., tide gauges or altimetry tracks) on a map.

    Args:
        points: A list of (latitude, longitude) tuples in degrees.
        data: Optional array of values to color the points by.
        ax: An existing Cartopy GeoAxes object. If None, creates a new one.
        projection: A cartopy projection instance if creating a new axes.
        cmap: The colormap to use if `data` is provided.
        color: The fixed color to use if `data` is NOT provided.
        s: The marker size.
        marker: The marker style (e.g., 'o', '*', '^').
        edgecolors: The color of the marker edges.
        zorder: The drawing order (higher means drawn on top of other elements).
        colorbar: If True and `data` is provided, attaches a colorbar.
        colorbar_kwargs: Formatting options for the colorbar.
        **kwargs: Additional keyword arguments passed to `ax.scatter`
            (e.g., vmin, vmax, alpha).

    Returns:
        A tuple `(ax, sc)` containing the GeoAxes and the PathCollection artist.
    """
    if ax is None:
        fig, ax = create_map_figure(projection=projection)
        # If we created a new map, it's nice to add coasts by default
        ax.set_global()
        ax.coastlines(linewidth=0.5, alpha=0.5)
    else:
        fig = ax.get_figure()

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    # If data is provided, map it to the colormap. Otherwise, use the fixed color.
    if data is not None:
        c = data
        kwargs.setdefault("cmap", cmap)
    else:
        c = color

    sc = ax.scatter(
        lons,
        lats,
        c=c,
        s=s,
        marker=marker,
        edgecolors=edgecolors,
        zorder=zorder,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )

    if colorbar and data is not None and fig:
        cb_opts = colorbar_kwargs or {}
        cb_opts.setdefault("orientation", "horizontal")
        cb_opts.setdefault("shrink", 0.7)
        cb_opts.setdefault("pad", 0.05)
        fig.colorbar(sc, ax=ax, **cb_opts)

    return ax, sc
