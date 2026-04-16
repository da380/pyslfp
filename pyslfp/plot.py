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
    coasts: bool = True,
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
    coasts: bool = True,  # pyslfp default (pygeoinf is False)
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: bool = False,
    colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Plots discrete observation points (e.g., tide gauges or altimetry tracks) on a map.

    This acts as a transparent wrapper around pygeoinf.sphere.plot_points. It shares
    the exact same API signature, but injects pyslfp's preferred defaults
    (Robinson projection, enabled coastlines).
    """
    if projection is None and ax is None:
        projection = ccrs.Robinson()

    return sphere.plot_points(
        points,
        data=data,
        ax=ax,
        projection=projection,
        cmap=cmap,
        color=color,
        s=s,
        marker=marker,
        edgecolors=edgecolors,
        zorder=zorder,
        coasts=coasts,
        rivers=rivers,
        borders=borders,
        map_extent=map_extent,
        gridlines=gridlines,
        symmetric=symmetric,
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        **kwargs,
    )


def plot_coastline(
    f: SHGrid,
    ax: GeoAxes,
    /,
    *,
    color: str = "black",
    linewidth: float = 1.0,
    zorder: int = 10,
    **kwargs,
) -> Any:
    """
    Plots a specific isoline (coastline) from an SHGrid onto an existing map.

    Args:
        f: The SHGrid object whose zeros define the coast line.
        ax: The Cartopy GeoAxes to plot onto.
        color: Line color.
        linewidth: Thickness of the line.
        zorder: Drawing order (default 10 to ensure it's above the data field).
        **kwargs: Additional arguments passed to ax.contour.

    Returns:
        The QuadContourSet artist created by the contour call.
    """
    # 1. Extract coordinates from the grid itself to ensure alignment
    lons = f.lons()
    lats = f.lats()

    # 2. Add standard contour styling defaults
    kwargs.setdefault("colors", color)
    kwargs.setdefault("linewidths", linewidth)
    kwargs.setdefault("zorder", zorder)

    # 3. Plot with the mandatory PlateCarree transform for degree-based data
    contour_set = ax.contour(
        lons, lats, f.data, levels=[0], transform=ccrs.PlateCarree(), **kwargs
    )

    return contour_set
