"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Union

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from pyshtools import SHGrid
import matplotlib.colors as colors


from pygeoinf.symmetric_space import sphere

from matplotlib.figure import Figure
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

import cartopy.crs as ccrs
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
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "horizontal",
    colorbar_pad: float = 0.05,
    colorbar_shrink: float = 0.7,
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
        colorbar (bool): If True, a colorbar is added to the plot.
            Defaults to True.
        colorbar_label (Optional[str]): Label for the colorbar.
            Defaults to None (no label).
        colorbar_orientation (str): Orientation of the colorbar
            ('horizontal' or 'vertical'). Defaults to 'horizontal'.
        colorbar_pad (float): Padding between the axes and the colorbar.
            Defaults to 0.05.
        colorbar_shrink (float): Fraction by which to multiply the size
            of the colorbar. Defaults to 0.7.
        **kwargs: Additional keyword arguments are forwarded to the underlying
            matplotlib plotting function (ax.pcolormesh or ax.contourf).

    Returns:
        Tuple[Figure, GeoAxes, Union[QuadMesh, QuadContourSet]]:
            A tuple containing the matplotlib Figure, the cartopy GeoAxes,
            and the plot artist object (e.g., QuadMesh or QuadContourSet).
    """

    if not isinstance(f, SHGrid):
        raise ValueError("must be of SHGrid type.")

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
    fig, ax, im = sphere.plot(f, **plot_options)

    # Add colorbar if requested
    if colorbar:
        fig.colorbar(
            im,
            ax=ax,
            orientation=colorbar_orientation,
            pad=colorbar_pad,
            shrink=colorbar_shrink,
            label=colorbar_label,
        )

    return fig, ax, im


def plot_corner_distributions(
    posterior_measure: object,
    /,
    *,
    prior_measure: Optional[object] = None,
    true_values: Optional[Union[List[float], np.ndarray]] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    show_plot: bool = True,
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
    width_scaling: float = 3.75,
    legend_position: tuple = (0.9, 0.95),
):
    """
    Create a corner plot for multi-dimensional posterior distributions.
    Optionally accepts a prior_measure to plot a secondary x-axis showing the
    width relative to the prior standard deviation.
    """

    # Extract statistics from the posterior measure
    if hasattr(posterior_measure, "expectation") and hasattr(
        posterior_measure, "covariance"
    ):
        mean_posterior = posterior_measure.expectation
        cov_posterior = posterior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )
    else:
        raise ValueError(
            "posterior_measure must have 'expectation' and 'covariance' attributes"
        )

    # Extract statistics from the prior measure if provided
    if prior_measure is not None:
        mean_prior = prior_measure.expectation
        cov_prior = prior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )

    n_dims = len(mean_posterior)

    if labels is None:
        labels = [f"Dimension {i+1}" for i in range(n_dims)]

    if figsize is None:
        figsize = (3 * n_dims, 3 * n_dims)

    fig, axes = plt.subplots(n_dims, n_dims, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = axes.reshape(2, 2)

    pcm = None

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:  # Diagonal plots (1D marginal distributions)
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

                ax.set_xlabel(labels[i])
                ax.set_ylabel("Density" if i == 0 else "")
                ax.set_yticklabels([])

                # NEW: Add secondary axis for the prior scale
                if prior_measure is not None:
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
                        r"Distance from Prior Mean ($\sigma_{prior}$)",
                        fontsize=10,
                        color="darkgreen",
                    )
                    sec_ax.tick_params(axis="x", colors="darkgreen")

            elif i > j:  # Lower triangle: 2D joint distributions
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

                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    shading="auto",
                    cmap=colormap,
                    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
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

                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])

            else:  # Upper triangle
                ax.axis("off")

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    cleaned_labels = [label.split(":")[0] for label in labels_leg]

    fig.legend(
        handles, cleaned_labels, loc="upper right", bbox_to_anchor=legend_position
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])

    if n_dims > 1 and pcm is not None:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label("Probability Density", size=12)

    if show_plot:
        plt.show()

    return fig, axes
