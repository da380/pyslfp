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

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy.stats as stats
from typing import Union

from pygeoinf import GaussianMeasure


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


def plot_corner_distributions(
    posterior_measure: GaussianMeasure,
    /,
    *,
    prior_measure: Optional[GaussianMeasure] = None,
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
    Create a professional corner plot for multi-dimensional posterior distributions.

    Args:
        posterior_measure: Multi-dimensional posterior measure (pygeoinf GaussianMeasure)
        prior_measure: Optional prior measure to plot secondary axes showing prior standard deviations.
        true_values: True values for each dimension (optional)
        labels: Labels for each dimension (optional)
        title: Title for the plot
        figsize: Figure size tuple (if None, calculated based on dimensions)
        include_sigma_contours: Whether to include 1-sigma contour lines
        colormap: Colormap for 2D plots
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.
        width_scaling: Width scaling factor in standard deviations (default: 3.75)
        legend_position: Position of legend as (x, y) tuple (default: (0.9, 0.95))

    Returns:
        axes: An N x N NumPy array of Matplotlib Axes objects.
    """
    # Strict type validation ensuring it's an authentic GaussianMeasure
    if not isinstance(posterior_measure, GaussianMeasure):
        raise TypeError(
            f"posterior_measure must be an instance of GaussianMeasure, "
            f"but got {type(posterior_measure).__name__}."
        )

    mean_posterior = posterior_measure.expectation
    cov_posterior = posterior_measure.covariance.matrix(
        dense=True, parallel=parallel, n_jobs=n_jobs
    )

    # Pre-compute prior matrices if provided
    if prior_measure is not None:
        if not isinstance(prior_measure, GaussianMeasure):
            raise TypeError("prior_measure must be a GaussianMeasure.")
        mean_prior = prior_measure.expectation
        cov_prior = prior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )

    n_dims = len(mean_posterior)

    if labels is None:
        labels = [f"Dimension {i+1}" for i in range(n_dims)]

    if figsize is None:
        figsize = (3 * n_dims, 3 * n_dims)

    # Tight grid spacing using the modern layout engine
    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=figsize,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        layout="constrained",
    )
    fig.suptitle(title, fontsize=16)

    # Ensure axes is always 2D array
    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = axes.reshape(2, 2)

    pcm = None

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            # --- DIAGONALS (1D PDFs) ---
            if i == j:
                mu = mean_posterior[i]
                sigma = np.sqrt(cov_posterior[i, i])
                x = np.linspace(
                    mu - width_scaling * sigma, mu + width_scaling * sigma, 200
                )
                pdf = stats.norm.pdf(x, mu, sigma)

                ax.plot(x, pdf, "darkblue", label="Posterior PDF")
                ax.fill_between(x, pdf, color="lightblue", alpha=0.6)

                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                # Inject prior secondary axis if requested
                if prior_measure is not None:
                    prior_mu = mean_prior[i]
                    prior_sigma = np.sqrt(cov_prior[i, i])

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
                        r"Distance from Prior Mean ($\sigma_{prior}$)",
                        fontsize=10,
                        color="darkgreen",
                    )
                    sec_ax.tick_params(axis="x", colors="darkgreen")

                # X-axis logic
                if i == n_dims - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.tick_params(labelbottom=False)

                # Y-axis logic: Hide all ticks, only label the very first plot
                ax.set_yticks([])
                if i == 0:
                    ax.set_ylabel("Density")
                else:
                    ax.set_ylabel("")

            # --- OFF-DIAGONALS (2D Contours) ---
            elif i > j:
                mean_2d = np.array([mean_posterior[j], mean_posterior[i]])
                cov_2d = np.array(
                    [
                        [cov_posterior[j, j], cov_posterior[j, i]],
                        [cov_posterior[i, j], cov_posterior[i, i]],
                    ]
                )

                sigma_j = np.sqrt(cov_posterior[j, j])
                sigma_i = np.sqrt(cov_posterior[i, i])

                x_range = np.linspace(
                    mean_2d[0] - width_scaling * sigma_j,
                    mean_2d[0] + width_scaling * sigma_j,
                    100,
                )
                y_range = np.linspace(
                    mean_2d[1] - width_scaling * sigma_i,
                    mean_2d[1] + width_scaling * sigma_i,
                    100,
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)

                # Failsafe: If the Gaussian spike falls between grid points
                # and the entire grid underflows to exactly 0.0
                if z_max <= 0.0:
                    z_max = 1.0
                    z_min = 1e-10
                else:
                    z_min = Z[Z > 0].min() if np.any(Z > 0) else z_max * 1e-10

                # Ensure minimum is strictly less than maximum
                if z_min >= z_max:
                    z_min = z_max * 1e-3

                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    shading="auto",
                    cmap=colormap,
                    norm=colors.LogNorm(vmin=z_min, vmax=z_max),
                )

                ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)

                if include_sigma_contours:
                    sigma_level = rv.pdf(mean_2d) * np.exp(-0.5)
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=[sigma_level],
                        colors="red",
                        linewidths=1,
                        linestyles="--",
                        alpha=0.8,
                    )

                ax.plot(
                    mean_posterior[j],
                    mean_posterior[i],
                    "r+",
                    markersize=10,
                    mew=2,
                    label="Posterior Mean",
                )

                if true_values is not None:
                    ax.plot(
                        true_values[j],
                        true_values[i],
                        "kx",
                        markersize=10,
                        mew=2,
                        label="True Value",
                    )

                # X-axis logic
                if i == n_dims - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.tick_params(labelbottom=False)

                # Y-axis logic
                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.tick_params(labelleft=False)

            # --- EMPTY UPPER TRIANGLE ---
            else:
                ax.set_visible(False)

    # Force Matplotlib to align the outer axis labels so they don't stagger
    fig.align_labels()

    # Extract legend handles safely from the first available plots
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    cleaned_labels = [label.split(":")[0] for label in labels_leg]

    # Avoid duplicate legends by dict conversion trick
    unique_legend = dict(zip(cleaned_labels, handles))

    # Let constrained_layout automatically make room for the legend on the right
    fig.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper right",
        bbox_to_anchor=legend_position,
    )

    # Let constrained_layout handle the colorbar natively
    if n_dims > 1 and pcm is not None:
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.7, aspect=30, pad=0.02)
        cbar.set_label("Probability Density", size=12)

    return axes
