"""Map plotting: thin wrappers over pygeoinf's spherical plotting
with pyslfp's defaults (Robinson projection, coastlines, an active
colour bar), so figures look the same across both libraries."""

from typing import Tuple, Optional, List, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from pyshtools import SHGrid

import pygeoinf
from pygeoinf.symmetric_space.sphere import Lebesgue

import cartopy.crs as ccrs
from cartopy.crs import Projection
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure

from pyslfp.core.response import SurfaceDisplacement


def subplots(
    nrows: int = 1, ncols: int = 1, projection: Optional[Projection] = None, /, **kwargs
) -> Tuple[plt.Figure, GeoAxes]:
    """
    A wrapper for matplotlib.pyplot subplots that returns cartopy compatible axes.

    Args:
        nrows: Number of rows in the axes grid
        ncols: Number of columns in the axes grid
        projection: A `cartopy.crs` projection instance. Defaults to Robinson.
        **kwargs: Additional keyword arguments passed to `plt.subplots()`.

    Returns:
        A tuple `(fig, ax)` containing the Matplotlib Figure and Cartopy GeoAxes.
    """
    if projection is None:
        projection = ccrs.Robinson()

    kwargs.setdefault("layout", "constrained")

    fig, axes = plt.subplots(
        nrows, ncols, subplot_kw={"projection": projection}, **kwargs
    )

    return fig, axes


def create_map_figure(
    figsize: Optional[Tuple[float, float]] = None,
    projection: Optional[Projection] = None,
    **kwargs,
) -> Tuple[Figure, GeoAxes]:
    """
    Creates a Matplotlib Figure and a single Cartopy GeoAxes, with
    constrained layout and pyslfp's default Robinson projection.
    """
    if figsize is not None:
        kwargs["figsize"] = figsize
    return subplots(1, 1, projection, **kwargs)


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
    symmetric: Union[bool, float] = False,
    contour_lines: bool = False,
    contour_lines_kwargs: Optional[dict] = None,
    num_levels: int = 10,
    colorbar: bool = True,  # pyslfp default (pygeoinf is False)
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Plots a pyshtools SHGrid object on a map.

    A wrapper around `pygeoinf.plot` with pyslfp's defaults: a Robinson
    projection, coastlines, and an active colour bar. Keywords not listed
    here are passed to it; see there for their meaning.

    Args:
        f: The field to plot.
        ax: Axes to draw on. A new Robinson map is made if omitted.
        projection: The projection of the new map, when `ax` is omitted.
        symmetric: True centres the colour scale on zero, with limits at
            the largest absolute value. A number s sets the limits to
            s times that value instead, saturating the scale for s < 1.
        contour_lines: Overlay contour lines.
        contour_lines_kwargs: Passed to `ax.contour` for those lines,
            over the defaults of thin black lines.
        num_levels: The number of contour levels, for `contour` and
            `contour_lines`. A `levels` keyword takes precedence.

    Returns:
        The `(axes, mappable)` pair.
    """
    if ax is None:
        _, ax = create_map_figure(projection=projection)

    symmetric = _apply_symmetric_limits(f.data, symmetric, kwargs)
    levels = kwargs.pop("levels", num_levels)

    ax, mappable = pygeoinf.plot(
        _sphere_of(f),
        f,
        ax=ax,
        contour=contour,
        cmap=cmap,
        coasts=coasts,
        rivers=rivers,
        borders=borders,
        map_extent=map_extent,
        gridlines=gridlines,
        symmetric=symmetric,
        levels=levels,
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        **kwargs,
    )

    if contour_lines:
        line_kwargs = dict(colors="black", linewidths=0.5)
        line_kwargs.update(contour_lines_kwargs or {})
        lons, lats, values = _closed_grid(f)
        ax.contour_set = ax.contour(
            lons,
            lats,
            values,
            levels=levels,
            transform=ccrs.PlateCarree(),
            **line_kwargs,
        )

    return ax, mappable


def plot_displacement(
    displacement: SurfaceDisplacement,
    /,
    *,
    ax: Optional[GeoAxes] = None,
    projection: Optional[Projection] = None,
    scale: float = 1.0,
    arrows: int = 30,
    key: Optional[float] = None,
    units: str = "",
    arrow_kwargs: Optional[dict] = None,
    symmetric: Union[bool, float] = True,
    **kwargs,
) -> Tuple[GeoAxes, Any, Any]:
    """
    Plots a surface displacement: the vertical part in colour, the horizontal
    part as arrows over it.

    The arrows are regridded to a regular grid in the map's own projection,
    so they are evenly spaced whatever the projection, and turned into its
    directions. Their lengths are to one scale, shown by a key arrow beneath
    the map. Without a horizontal part only the colours are drawn.

    Args:
        displacement: The displacement, as a `GRDResponse` holds it.
        ax: Axes to draw on. A new Robinson map is made if omitted.
        projection: The projection of the new map, when `ax` is omitted.
        scale: A factor applied to both parts, to give them units; for a
            non-dimensional displacement in millimetres, the length scale
            in metres times 1000.
        arrows: The number of arrows across the map, and down it.
        key: The length of the key arrow, in the scaled units. A round
            number near the largest horizontal displacement if omitted.
        units: The units, for the key's label.
        arrow_kwargs: Passed to `ax.quiver`, over the defaults of black
            arrows pivoting at their middles and drawn to a scale at which
            the longest is one arrow spacing, so that none overruns its
            neighbours. A `scale` there sets the scale instead, in scaled
            units per width of the axes.
        symmetric: As for `plot`; the colour scale is centred on zero by
            default, uplift and subsidence being of either sign.
        **kwargs: Passed to `plot` for the vertical part.

    Returns:
        The axes, the colour mappable, and the quiver (None without a
        horizontal part).
    """
    vertical = displacement.vertical_field() * scale
    ax, mappable = plot(
        vertical, ax=ax, projection=projection, symmetric=symmetric, **kwargs
    )
    if displacement.horizontal is None:
        return ax, mappable, None

    east, north = displacement.horizontal_fields()
    lons, lats = np.meshgrid(east.lons(), east.lats())
    # East and north are undefined at the poles; the rows there are dropped.
    rows = np.abs(east.lats()) < 90.0
    u = east.data[rows] * scale
    v = north.data[rows] * scale
    largest = float(np.max(np.hypot(u, v)))
    quiver_kwargs = dict(color="black", pivot="middle", scale_units="width")
    if largest > 0.0:
        quiver_kwargs["scale"] = largest * arrows
    quiver_kwargs.update(arrow_kwargs or {})
    quiver = ax.quiver(
        lons[rows],
        lats[rows],
        u,
        v,
        transform=ccrs.PlateCarree(),
        regrid_shape=arrows,
        **quiver_kwargs,
    )
    if key is None:
        key = _round_down(largest)
    if key > 0.0:
        label = f"{key:g} {units}".strip()
        ax.quiverkey(quiver, 0.9, -0.06, key, label, labelpos="E", coordinates="axes")
    return ax, mappable, quiver


def _round_down(value: float) -> float:
    """The value rounded down to one significant figure."""
    if value <= 0.0 or not np.isfinite(value):
        return 0.0
    power = 10.0 ** np.floor(np.log10(value))
    return float(np.floor(value / power) * power)


def plot_points(
    points: List[Tuple[float, float]],
    /,
    *,
    data: Optional[List[float] | np.ndarray] = None,
    ax: Optional[GeoAxes] = None,
    projection: Optional[Projection] = None,
    cmap: str = "RdBu",
    color: str = "k",
    s: float = 20,
    marker: str = "o",
    edgecolors: str = "None",
    zorder: int = 5,
    coasts: bool = True,  # pyslfp default (pygeoinf is False)
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: Union[bool, float] = False,
    colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Plots discrete observation points (tide gauges, altimetry tracks)
    on a map.

    A wrapper around `pygeoinf.plot_points` with pyslfp's defaults: a
    Robinson projection and coastlines. Keywords not listed here are
    passed to `ax.scatter`.

    Args:
        points: (latitude, longitude) pairs in degrees.
        data: One value per point to colour the markers by.
        ax: Axes to draw on. A new global Robinson map is made if omitted.
        projection: The projection of the new map, when `ax` is omitted.
        s: Marker area.
        map_extent: (west, east, south, north) in degrees. Without it a
            new map is global and existing axes keep their extent.
        gridlines: Add a graticule, unless the axes already carry one.
        symmetric: As for `plot`, applied to `data`.

    Returns:
        The `(axes, collection)` pair.
    """
    new_axes = ax is None
    if new_axes:
        _, ax = create_map_figure(projection=projection)

    if map_extent is not None:
        ax.set_extent(list(map_extent), crs=ccrs.PlateCarree())
    elif new_axes:
        ax.set_global()

    if coasts:
        ax.coastlines(linewidth=0.5)
    if borders or rivers:
        import cartopy.feature as cfeature

        if borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        if rivers:
            ax.add_feature(cfeature.RIVERS, linewidth=0.3)
    if gridlines and getattr(ax, "gridliner", None) is None:
        ax.gridliner = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    if data is not None:
        symmetric = _apply_symmetric_limits(np.asarray(data), symmetric, kwargs)

    return pygeoinf.plot_points(
        _MAP_SPACE,
        points,
        data=data,
        ax=ax,
        marker=marker,
        size=s,
        color=color,
        cmap=cmap,
        symmetric=bool(symmetric),
        colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs,
        edgecolors=edgecolors,
        zorder=zorder,
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


# ================================================================ #
#                        Private helpers                           #
# ================================================================ #

# pygeoinf's renderers dispatch on the space. A scatter of points uses
# nothing of it beyond its type, so one small sphere serves every call.
_MAP_SPACE = Lebesgue(1)


def _sphere_of(f: SHGrid) -> Lebesgue:
    """The pygeoinf space whose vectors have this grid's layout.

    A renderer reads the grid through the space, so the truncation, grid
    type, sampling and wrap column must match the field's. The radius plays
    no part in a map and is left at its default.
    """
    return Lebesgue(
        f.lmax, grid=f.grid, sampling=getattr(f, "sampling", 1), extend=f.extend
    )


def _apply_symmetric_limits(
    values: np.ndarray, symmetric: Union[bool, float], kwargs: dict
) -> bool:
    """Translates pyslfp's `symmetric` into pygeoinf's.

    A number is a saturation factor: the limits become plus and minus that
    multiple of the largest absolute value, written into `kwargs` as `vmin`
    and `vmax` unless the caller gave those. Returns the flag pygeoinf takes.
    """
    if isinstance(symmetric, (bool, np.bool_)):
        return bool(symmetric)
    extent = float(symmetric) * float(np.nanmax(np.abs(values)))
    kwargs.setdefault("vmin", -extent)
    kwargs.setdefault("vmax", extent)
    return True


def _closed_grid(f: SHGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longitudes, latitudes and values with the seam at 360 degrees closed,
    so that contour lines run across it."""
    lons, lats, values = f.lons(), f.lats(), f.data
    if not f.extend:
        lons = np.append(lons, lons[0] + 360.0)
        values = np.concatenate([values, values[:, :1]], axis=1)
    return lons, lats, values
