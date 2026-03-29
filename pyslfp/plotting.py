"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Union, Any

import numpy as np
import matplotlib.pyplot as plt

from pyshtools import SHGrid


from pygeoinf import plot_corner_distributions as pygeoinf_corner_plot
from pygeoinf.symmetric_space import sphere

from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

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

    This eliminates the need to manually import cartopy.crs or remember
    the exact subplot_kw syntax just to set up a basic map canvas.

    Args:
        figsize: A tuple of (width, height) in inches.
        projection: A `cartopy.crs` projection instance. Defaults to PlateCarree.
        **kwargs: Additional keyword arguments passed to `plt.subplots()`.

    Returns:
        A tuple `(fig, ax)` containing the Matplotlib Figure and Cartopy GeoAxes.
    """
    if projection is None:
        projection = ccrs.Robinson()

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": projection}, **kwargs
    )

    return fig, ax


def plot(
    f: SHGrid,
    /,
    *,
    ax: Optional[GeoAxes] = None,
    projection: Optional[ccrs.Projection] = None,
    contour: bool = False,
    cmap: str = "RdBu",
    coasts: bool = True,
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: bool = False,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "horizontal",
    colorbar_pad: float = 0.05,
    colorbar_shrink: float = 0.7,
    **kwargs,
) -> Tuple[GeoAxes, Union[QuadMesh, QuadContourSet]]:
    """
    Plots a pyshtools SHGrid object on a map.

    This acts as a wrapper around pygeoinf.sphere.plot, aligning with standard
    matplotlib APIs by optionally accepting an 'ax' and returning only the Axes
    and the mappable artist.
    """
    if not isinstance(f, SHGrid):
        raise ValueError("f must be of SHGrid type.")

    if projection is None and ax is None:
        projection = ccrs.Robinson()

    # Delegate to pygeoinf.sphere.plot, passing down the optional ax
    ax, im = sphere.plot(
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
        **kwargs,
    )

    # Add colorbar if requested by grabbing the figure dynamically
    if colorbar:
        fig = ax.get_figure()
        if fig:
            fig.colorbar(
                im,
                ax=ax,
                orientation=colorbar_orientation,
                pad=colorbar_pad,
                shrink=colorbar_shrink,
                label=colorbar_label,
            )

    return ax, im


def plot_corner_distributions(
    posterior_measure: Any,
    /,
    *,
    prior_measure: Optional[Any] = None,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
    width_scaling: float = 3.75,
    legend_position: tuple = (0.9, 0.95),
) -> np.ndarray:
    """
    Create a corner plot for multi-dimensional posterior distributions.
    Delegates base plotting to pygeoinf, but injects a custom secondary x-axis
    showing the width relative to the prior standard deviation if prior_measure
    is provided.
    """

    # 1. Let pygeoinf handle the heavy lifting of the corner plot grid
    axes = pygeoinf_corner_plot(
        posterior_measure,
        true_values=true_values,
        labels=labels,
        title=title,
        figsize=figsize,
        include_sigma_contours=include_sigma_contours,
        colormap=colormap,
        parallel=parallel,
        n_jobs=n_jobs,
        width_scaling=width_scaling,
        legend_position=legend_position,
    )

    # 2. Inject pyslfp's custom prior secondary axes on the diagonals
    if prior_measure is not None:
        mean_prior = prior_measure.expectation
        cov_prior = prior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )

        n_dims = len(mean_prior)

        for i in range(n_dims):
            ax = axes[i, i]  # Target the 1D marginal diagonals

            prior_mu = mean_prior[i]
            prior_sigma = np.sqrt(cov_prior[i, i])

            # Closures for the secondary axis transformations
            def make_forward(p_mu, p_sig):
                return lambda val: (val - p_mu) / p_sig

            def make_inverse(p_mu, p_sig):
                return lambda stds: stds * p_sig + p_mu

            sec_ax = ax.secondary_xaxis(
                "top",
                functions=(
                    make_forward(prior_mu, prior_sigma),
                    make_inverse(prior_mu, prior_sigma),
                ),
            )
            sec_ax.set_xlabel(
                r"Distance from Prior Mean)",
                fontsize=10,
                color="darkgreen",
            )
            sec_ax.tick_params(axis="x", colors="darkgreen")

    return axes
