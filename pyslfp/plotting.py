"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Any

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

    # If a colorbar is requested but no kwargs are provided, inject the
    # traditional pyslfp aesthetics so the user doesn't have to type them.
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
