"""
Module for plotting functions using matplotlib and cartopy.
"""

from typing import Tuple, Optional, List, Any, Union

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
    colormap: str = "Blues",
    contour_color: str = "darkblue",
    parallel: bool = False,
    n_jobs: int = -1,
    width_scaling: float = 3.75,
    legend_position: tuple = (0.9, 0.95),
    fill_density: bool = False,
    num_sigmas: int = 3,
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
        colormap: Colormap for 2D plots (used when fill_density=True)
        contour_color: Uniform color for the 2D contour lines (used when fill_density=False)
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.
        width_scaling: Width scaling factor in standard deviations for default boundaries (default: 3.75)
        legend_position: Position of legend as (x, y) tuple (default: (0.9, 0.95))
        fill_density: Whether to fill the 2D contour background with color. False is recommended for sparse truth values.
        num_sigmas: Minimum number of standard deviation contours to draw (dynamically scales up to enclose true values).

    Returns:
        axes: An N x N NumPy array of Matplotlib Axes objects.
    """
    if not isinstance(posterior_measure, GaussianMeasure):
        raise TypeError(
            f"posterior_measure must be an instance of GaussianMeasure, "
            f"but got {type(posterior_measure).__name__}."
        )

    mean_posterior = posterior_measure.expectation
    cov_posterior = posterior_measure.covariance.matrix(
        dense=True, parallel=parallel, n_jobs=n_jobs
    )
    std_posterior = np.sqrt(np.diag(cov_posterior))

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

    # --- Smart Contour Level Calculation (Mahalanobis Distance) ---
    effective_num_sigmas = num_sigmas
    if true_values is not None:
        max_dist = 0.0
        if n_dims > 1:
            # Check the mathematical 2D distance for every plot pair
            for i in range(n_dims):
                for j in range(i):
                    diff = np.array(
                        [
                            true_values[j] - mean_posterior[j],
                            true_values[i] - mean_posterior[i],
                        ]
                    )
                    cov_2d = np.array(
                        [
                            [cov_posterior[j, j], cov_posterior[j, i]],
                            [cov_posterior[i, j], cov_posterior[i, i]],
                        ]
                    )
                    # Add tiny epsilon to prevent singular matrix errors in perfectly correlated edge cases
                    cov_2d += np.eye(2) * 1e-12
                    inv_cov = np.linalg.inv(cov_2d)
                    dist = np.sqrt(diff.T @ inv_cov @ diff)
                    max_dist = max(max_dist, dist)
        else:
            # Fallback for 1D edge cases
            max_dist = np.abs(true_values[0] - mean_posterior[0]) / std_posterior[0]

        # Ensure we draw enough contours to swallow the furthest point, capped at 15 to prevent memory crashes
        effective_num_sigmas = min(15, max(num_sigmas, int(np.ceil(max_dist))))

    # --- Smart Span Calculation ---
    display_spans = np.zeros(n_dims)
    eval_spans = np.zeros(n_dims)

    for idx in range(n_dims):
        z_score = 0.0
        if true_values is not None:
            z_score = (
                np.abs(true_values[idx] - mean_posterior[idx]) / std_posterior[idx]
            )

        # Display window must contain the default width OR the true value with a 5% visual buffer
        display_spans[idx] = max(width_scaling, z_score * 1.05)
        # Math evaluation grid must be at least as wide as the display OR the dynamically calculated contours
        eval_spans[idx] = max(display_spans[idx], effective_num_sigmas + 1.0)

    fig, axes = plt.subplots(
        n_dims,
        n_dims,
        figsize=figsize,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        layout="constrained",
    )
    fig.suptitle(title, fontsize=16)

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
                sigma = std_posterior[i]

                i_eval = eval_spans[i]
                i_disp = display_spans[i]

                # Scale grid resolution, but cap it to prevent memory issues for extreme true values
                n_pts_1d = min(5000, max(200, int(50 * i_eval)))

                x = np.linspace(mu - i_eval * sigma, mu + i_eval * sigma, n_pts_1d)
                pdf = stats.norm.pdf(x, mu, sigma)

                ax.plot(x, pdf, "darkblue", label="Posterior PDF")

                if fill_density:
                    ax.fill_between(x, pdf, color="lightblue", alpha=0.6)

                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                ax.set_xlim(mu - i_disp * sigma, mu + i_disp * sigma)

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

                if i == n_dims - 1:
                    ax.set_xlabel(labels[i])
                else:
                    ax.tick_params(labelbottom=False)

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

                sigma_j = std_posterior[j]
                sigma_i = std_posterior[i]

                j_eval = eval_spans[j]
                i_eval = eval_spans[i]

                # Scale grid resolution, cap to max 500x500 to prevent severe slowdowns
                n_pts_j = min(500, max(100, int(25 * j_eval)))
                n_pts_i = min(500, max(100, int(25 * i_eval)))

                x_range = np.linspace(
                    mean_2d[0] - j_eval * sigma_j,
                    mean_2d[0] + j_eval * sigma_j,
                    n_pts_j,
                )
                y_range = np.linspace(
                    mean_2d[1] - i_eval * sigma_i,
                    mean_2d[1] + i_eval * sigma_i,
                    n_pts_i,
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                rv = stats.multivariate_normal(mean_2d, cov_2d)
                Z = rv.pdf(pos)

                peak_density = rv.pdf(mean_2d)

                # Values are sorted ascending (lowest density/outermost ring first, to highest density/innermost last)
                sigma_levels = sorted(
                    [
                        peak_density * np.exp(-0.5 * s**2)
                        for s in range(1, effective_num_sigmas + 1)
                    ]
                )

                if fill_density:
                    pcm = ax.pcolormesh(X, Y, Z, shading="auto", cmap=colormap)
                    ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)
                    if effective_num_sigmas >= 1:
                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=[peak_density * np.exp(-0.5)],
                            colors="red",
                            linewidths=1,
                            linestyles="--",
                            alpha=0.8,
                        )
                else:
                    if sigma_levels:
                        # Extract the base RGB components of our chosen contour color
                        base_rgba = colors.to_rgba(contour_color)

                        # Build an array of opacities from faint (outer) to solid (inner)
                        min_alpha = 0.2
                        max_alpha = 0.9
                        if effective_num_sigmas == 1:
                            level_colors = [
                                (base_rgba[0], base_rgba[1], base_rgba[2], max_alpha)
                            ]
                        else:
                            # np.linspace aligns perfectly with the sorted sigma_levels:
                            # index 0 is outermost ring (gets min_alpha), last index is innermost ring (gets max_alpha)
                            alpha_array = np.linspace(
                                min_alpha, max_alpha, effective_num_sigmas
                            )
                            level_colors = [
                                (base_rgba[0], base_rgba[1], base_rgba[2], a)
                                for a in alpha_array
                            ]

                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=sigma_levels,
                            colors=level_colors,
                            linewidths=1.5,
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

                ax.set_xlim(
                    mean_2d[0] - display_spans[j] * sigma_j,
                    mean_2d[0] + display_spans[j] * sigma_j,
                )
                ax.set_ylim(
                    mean_2d[1] - display_spans[i] * sigma_i,
                    mean_2d[1] + display_spans[i] * sigma_i,
                )

                if i == n_dims - 1:
                    ax.set_xlabel(labels[j])
                else:
                    ax.tick_params(labelbottom=False)

                if j == 0:
                    ax.set_ylabel(labels[i])
                else:
                    ax.tick_params(labelleft=False)

            else:
                ax.set_visible(False)

    fig.align_labels()

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[1, 0].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    cleaned_labels = [label.split(":")[0] for label in labels_leg]
    unique_legend = dict(zip(cleaned_labels, handles))

    fig.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper right",
        bbox_to_anchor=legend_position,
    )

    if n_dims > 1 and pcm is not None and fill_density:
        cbar = fig.colorbar(pcm, ax=axes, shrink=0.7, aspect=30, pad=0.02)
        cbar.set_label("Probability Density", size=12)

    return axes
