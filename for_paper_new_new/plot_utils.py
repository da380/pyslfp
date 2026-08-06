import numpy as np
import scipy.stats as stats


def plot_normalized_mc_errors(
    ax,
    raw_errs_x,
    raw_errs_y,
    raw_mean_2d,
    raw_cov_2d,
    std_x,
    std_y,
    /,
    *,
    title="",
    xlabel=None,
    ylabel=None,
    label_x="Standard",
    label_y="Bayesian",
    show_legend=False,
    show_samples=True,
):
    """
    Plots a 2D Monte Carlo validation scatter plot with analytical PDF contours
    and 1-sigma / 2-sigma background shading.
    """
    # Normalize the analytical mean and covariance
    mu_2d = np.array([raw_mean_2d[0] / std_x, raw_mean_2d[1] / std_y])

    var_x = raw_cov_2d[0, 0] / (std_x**2)
    var_y = raw_cov_2d[1, 1] / (std_y**2)
    cov_xy = raw_cov_2d[0, 1] / (std_x * std_y)
    cov_2d = np.array([[var_x, cov_xy], [cov_xy, var_y]])

    # Dynamically calculate symmetrical plot limits based on the analytical distribution.
    # We want to capture the mean shift plus 4 standard deviations of the true distribution.
    std_ana_x = np.sqrt(var_x)
    std_ana_y = np.sqrt(var_y)

    limit_from_cov = max(
        np.abs(mu_2d[0]) + 4 * std_ana_x, np.abs(mu_2d[1]) + 4 * std_ana_y
    )
    plot_limit = np.ceil(limit_from_cov)

    # If samples are shown, ensure they also fit within the limits
    if show_samples and raw_errs_x is not None:
        errs_x = raw_errs_x / std_x
        errs_y = raw_errs_y / std_y
        max_err = max(np.max(np.abs(errs_x)), np.max(np.abs(errs_y)))
        plot_limit = max(plot_limit, np.ceil(max_err) + 0.5)

    # Scatter Plot
    if show_samples and raw_errs_x is not None:
        ax.scatter(
            errs_x, errs_y, alpha=0.6, color="purple", edgecolor="white", s=20, zorder=3
        )

    # Analytical 2D PDF Contours
    x_grid, y_grid = np.mgrid[-plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j]
    pos = np.dstack((x_grid, y_grid))

    rv = stats.multivariate_normal(mu_2d, cov_2d)
    max_density = rv.pdf(mu_2d)
    levels = [max_density * np.exp(-0.5 * k**2) for k in [4, 3, 2, 1]]

    ax.contour(
        x_grid,
        y_grid,
        rv.pdf(pos),
        levels=levels,
        colors="indigo",
        linewidths=[0.5, 1.0, 1.5],
        alpha=0.8,
        zorder=4,
    )

    # Background Shading
    ax.axhline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
    ax.axvline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
    ax.axhspan(
        -1,
        1,
        color="blue",
        alpha=0.15,
        zorder=0,
        label=rf"{label_y} 1$\sigma$ Expected",
    )
    ax.axhspan(-2, 2, color="blue", alpha=0.05, zorder=0)
    ax.axvspan(
        -1, 1, color="red", alpha=0.15, zorder=0, label=rf"{label_x} 1$\sigma$ Expected"
    )
    ax.axvspan(-2, 2, color="red", alpha=0.05, zorder=0)

    # Formatting
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)

    if show_legend:
        # Dummy line to add the contour to the legend cleanly
        ax.plot([], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF")
        ax.legend(loc="upper left", fontsize=9)

    return ax
