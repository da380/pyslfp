"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Union
import numpy as np
from pyshtools import SHGrid

from pygeoinf.symmetric_space.sphere import SphereHelper

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
    Plots a pyshtools SHGrid object on a map.

    This function provides a flexible interface to visualize spherical harmonic
    grid data by acting as a wrapper around the plotting facilities provided
    by the pygeoinf library.

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
            around zero. This is overridden if 'vmin' or 'vmax' are provided
            in kwargs. Defaults to False.
        **kwargs: Additional keyword arguments are forwarded to the underlying
            matplotlib plotting function (ax.pcolormesh or ax.contourf).

    Returns:
        Tuple[Figure, GeoAxes, Union[QuadMesh, QuadContourSet]]:
            A tuple containing the matplotlib Figure, the cartopy GeoAxes,
            and the plot artist object (e.g., QuadMesh or QuadContourSet).
    """

    if not isinstance(f, SHGrid):
        raise ValueError("must be of SHGrid type.")

    # Instantiate the helper class from pygeoinf.
    sphere_helper = SphereHelper(f.lmax, 1, f.grid, f.extend)

    # --- Create a dictionary to hold all keyword arguments ---
    plot_options = {
        "projection": projection,
        "contour": contour,
        "cmap": cmap,
        "coasts": coasts,
        "rivers": rivers,
        "borders": borders,
        "map_extent": map_extent,
        "gridlines": gridlines,
        "symmetric": symmetric,
    }

    plot_options.update(kwargs)

    # Call the underlying plot method, unpacking the collected options.
    return sphere_helper.plot(f, **plot_options)
