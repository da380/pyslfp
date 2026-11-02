"""
Figures shared by the experiments of Heathcote et al. (2026): the style and
helpers of the maps, single maps with a panel letter, the densities of an
estimator's error, the grids of model fields and of covariances of the ocean
inversions, region boundaries, corner plots, the kernel-error summary of the
GRACE inversion, and the GMSL split ellipses and component variance dots of
the joint inversion.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import pygeoinf as inf
import pyslfp as sl

from common import ETA, ICE, RHO

# The fonts of the inversion scripts' figures; see `set_inversion_style`.
INVERSION_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
}

GRIDLINES = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}
COLORBAR = {"orientation": "horizontal", "shrink": 0.8, "pad": 0.05}

# The fields of the ocean experiments, in the rows of their map grids: the
# row title and the display unit (see `common.OceanPhysics.display`).
FIELDS = (ICE, ETA, RHO)
ROW_TITLES = {
    ICE: "Ice Thickness Change",
    ETA: "Sterodynamic SL Change",
    RHO: "Averaged Density Change",
}
UNIT_LABELS = {ICE: "mm", ETA: "mm", RHO: r"g m$^{-3}$"}


def set_inversion_style() -> None:
    """Sets the larger fonts of the inversion scripts' figures."""
    plt.rcParams.update(INVERSION_STYLE)


def field_label(name: str) -> str:
    """The colour bar label of an ocean field, with its unit."""
    return f"{ROW_TITLES[name]} ({UNIT_LABELS[name]})"


def covariance_label(row: str, perturbed: str) -> str:
    """The colour bar label of the covariance of field `row` with field
    `perturbed`, in the product of their units."""
    heights = {ICE, ETA}
    if row in heights and perturbed in heights:
        return "Covariance (mm²)"
    if row not in heights and perturbed not in heights:
        return r"Covariance ((g m$^{-3}$)$^{2}$)"
    return r"Covariance (mm $\cdot$ g m$^{-3}$)"


def peak(field) -> float:
    """The largest absolute value of a field on its grid."""
    return float(np.max(np.abs(field.data)))


# ================================================================ #
#                       Labels and boundaries                      #
# ================================================================ #


def panel_label(ax, text: str) -> None:
    """A panel letter in the top left corner of an axis."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
        zorder=10,
    )


def style_map(ax, mappable, label: str) -> None:
    """The colour bar label and the tick and gridline label sizes of a
    single map."""
    mappable.colorbar.set_label(label, fontsize=16)
    mappable.colorbar.ax.tick_params(labelsize=14)
    ax.gridliner.xlabel_style = {"size": 12, "color": "black"}
    ax.gridliner.ylabel_style = {"size": 12, "color": "black"}


def label_rows(axes, titles) -> None:
    """Row titles, written vertically to the left of a grid of axes."""
    for i, title in enumerate(titles):
        axes[i, 0].annotate(
            title,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            fontsize=16,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )


def label_columns(axes, titles) -> None:
    """Column titles above a grid of axes."""
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=16, fontweight="bold", pad=20)


def draw_boundaries(state, axes, names) -> None:
    """The boundaries of the regions `names` on every axis; none if `names`
    is empty."""
    if not names:
        return
    for ax in np.ravel(axes):
        state.plot_boundaries(ax, names)


def draw_region_boundaries(state, ax, regions_dict, **kwargs):
    """The boundaries of every region of a regions dictionary, whose values
    are a region name or a list of names that make up one region."""
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("linewidth", 2.0)

    raw_regions = []
    for val in regions_dict.values():
        if isinstance(val, str):
            raw_regions.append(val)
        else:
            raw_regions.extend(val)

    state.plot_boundaries(ax, raw_regions, **kwargs)


# ================================================================ #
#                     Single maps and densities                    #
# ================================================================ #


def field_map(field, label: str, panel: str, /, *, state=None, regions=None):
    """One field on a map, on a colour scale symmetric about zero, with a
    panel letter and, given `regions` (a regions dictionary), their
    boundaries."""
    fig, ax = sl.create_map_figure(figsize=(12, 6))
    _, image = sl.plot(field, ax=ax, cmap="seismic", symmetric=True)
    style_map(ax, image, label)
    panel_label(ax, panel)
    if regions is not None:
        draw_region_boundaries(state, ax, regions)
    return fig


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))


def error_density(
    ax,
    mean: float,
    std: float,
    nominal_std: float,
    true_std: float,
    /,
    *,
    samples=None,
    linewidth: float = 2.0,
    top_fontsize: float = 10,
) -> None:
    """
    The density of an estimator's actual error N(mean, std) against the
    nominal error N(0, nominal std), over four standard deviations of the
    actual error, with a histogram of Monte Carlo errors if `samples` is
    given. The top axis divides the error by the true quantity's standard
    deviation `true_std`. The caller sets the labels of the bottom axis and
    the legend.
    """
    if samples is not None:
        ax.hist(
            samples, bins=50, alpha=0.3, color="red", density=True, label="MC Samples"
        )
    x = np.linspace(mean - 4 * std, mean + 4 * std, 300)
    ax.plot(
        x, normal_pdf(x, mean, std), "r-", linewidth=linewidth, label="Actual error"
    )
    ax.plot(
        x,
        normal_pdf(x, 0.0, nominal_std),
        "b-",
        linewidth=2,
        label="Nominal error (noise only)",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    top = ax.secondary_xaxis(
        "top", functions=(lambda v: v / true_std, lambda v: v * true_std)
    )
    top.set_xlabel(
        r"Error Standardised by True Signal $\sigma$",
        fontsize=top_fontsize,
        color="darkgreen",
    )
    top.tick_params(axis="x", colors="darkgreen")


def corner_figure(measure, /, **kwargs):
    """A corner plot of a low-dimensional Gaussian measure, without a title;
    `kwargs` go to `pygeoinf.plot_corner`. Returns the figure."""
    axes = inf.plot_corner(measure, title="", **kwargs)
    return np.asarray(axes).flat[0].figure


# ================================================================ #
#                   Map grids of the ocean inversions              #
# ================================================================ #


def state_maps(
    true,
    mean,
    std=None,
    /,
    *,
    limits: dict,
    figsize,
    std_limits: dict = None,
    mean_title: str = "Posterior Expectation",
    row_titles: bool = False,
    state=None,
    boundaries=None,
):
    """
    The true state and a posterior expectation, field by field, and the
    posterior's pointwise standard deviation if `std` is given: three rows
    and two or three columns. Each argument maps the field names to fields
    on their supports in display units (`common.OceanPhysics.display`).
    `limits` holds each row's symmetric colour limit and `std_limits` that of
    the standard deviations, autoscaled if omitted. With `row_titles` the
    rows are titled; with `boundaries` (region names) their boundaries are
    drawn on every map.
    """
    ncols = 2 if std is None else 3
    fig, axes = sl.subplots(3, ncols, figsize=figsize, gridspec_kw={"hspace": 0.15})
    for row, name in enumerate(FIELDS):
        for col, field in enumerate((true[name], mean[name])):
            sl.plot(
                field,
                ax=axes[row, col],
                colorbar=True,
                vmin=-limits[name],
                vmax=limits[name],
                cmap="seismic",
                colorbar_kwargs={**COLORBAR, "label": field_label(name)},
                gridlines_kwargs=GRIDLINES,
            )
        if std is not None:
            bounds = {} if std_limits is None else {"vmin": 0, "vmax": std_limits[name]}
            sl.plot(
                std[name],
                ax=axes[row, 2],
                colorbar=True,
                cmap="Blues",
                **bounds,
                colorbar_kwargs={
                    **COLORBAR,
                    "label": f"{ROW_TITLES[name]} STD ({UNIT_LABELS[name]})",
                },
                gridlines_kwargs=GRIDLINES,
            )
    draw_boundaries(state, axes, boundaries)
    label_columns(axes, ["True State", mean_title, "Pointwise Std. Deviation"][:ncols])
    if row_titles:
        label_rows(axes, [ROW_TITLES[name] for name in FIELDS])
    return fig


def covariance_maps(
    columns,
    perturbed: str,
    point,
    /,
    *,
    marker: str = "star",
    wspace=None,
    state=None,
    boundaries=None,
):
    """
    Maps of the covariance of every field with the field `perturbed` at
    `point`, one column per measure and one row per field, each map on its
    own symmetric colour scale; a map that vanishes takes the largest scale
    of its row. `columns` is a list of (title, fields), the fields by name
    on their supports in display units (`common.OceanPhysics.point_covariance`).
    The point is starred in the first column (`marker="star"`) or dotted in
    every one (`marker="dot"`). `wspace` sets the gap between the columns;
    with `boundaries` (region names) their boundaries are drawn on every map.
    """
    ncols = len(columns)
    gridspec = {"hspace": 0.15}
    if wspace is not None:
        gridspec["wspace"] = wspace
    fig, axes = sl.subplots(3, ncols, figsize=(8 * ncols, 16), gridspec_kw=gridspec)
    for row, name in enumerate(FIELDS):
        fields = [fields[name] for _, fields in columns]
        limits = [peak(field) for field in fields]
        fallback = max(limits) if max(limits) > 0 else 1.0
        limits = [limit if limit != 0 else fallback for limit in limits]
        for col, (field, limit) in enumerate(zip(fields, limits)):
            ax = axes[row, col]
            sl.plot(
                field,
                ax=ax,
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-limit,
                vmax=limit,
                colorbar_kwargs={
                    **COLORBAR,
                    "label": covariance_label(name, perturbed),
                },
                gridlines_kwargs=GRIDLINES,
            )
            if marker == "dot":
                sl.plot_points(
                    [point], ax=ax, color="black", zorder=10, gridlines=False
                )
            elif col == 0:
                sl.plot_points(
                    [point],
                    ax=ax,
                    marker="*",
                    color="gold",
                    edgecolors="black",
                    s=150,
                    zorder=10,
                    gridlines=False,
                )
    label_columns(axes, [title for title, _ in columns])
    label_rows(axes, [ROW_TITLES[name] for name in FIELDS])
    draw_boundaries(state, axes, boundaries)
    return fig


# ================================================================ #
#                       Comparison summaries                       #
# ================================================================ #

# Styled through rc_context, so that the scripts' own rcParams do not leak
# in, and sized for a single column.
_SUMMARY_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "font.family": "sans-serif",
}
_C_BAYES = "#0072B2"  # Okabe-Ito blue
_C_WMB = "#D55E00"  # Okabe-Ito vermillion


def plot_kernel_error_ratios(
    region_rows,
    /,
    *,
    degree_rows=None,
    degree_label=None,
):
    """
    Dumbbell chart of relative estimator-kernel errors ||k - t|| / ||t||
    in the load-space norm: Bayesian (filled) vs WMB (open) per region,
    with an optional Bayesian-only cluster for single-degree
    coefficients below a separator. Smaller is better. region_rows is a
    sequence of (name, bayes_ratio, wmb_ratio); degree_rows of
    (name, bayes_ratio). Returns the figure.
    """
    region_rows = list(region_rows)
    degree_rows = list(degree_rows) if degree_rows else []
    n_deg = len(degree_rows)
    n_total = len(region_rows) + n_deg
    ys_reg = np.arange(len(region_rows))[::-1] + (n_deg + 1.0 if n_deg else 0.0)
    ys_deg = np.arange(n_deg)[::-1]

    ratios = [b for _, b, _ in region_rows]
    ratios += [w for _, _, w in region_rows]
    ratios += [b for _, b in degree_rows]
    xmax = max(0.5, 1.12 * max(ratios))

    with mpl.rc_context(_SUMMARY_RC):
        fig, ax = plt.subplots(
            figsize=(4.8, 1.0 + 0.30 * (n_total + (1 if n_deg else 0))),
            layout="constrained",
        )
        for y, (_, bayes, wmb) in zip(ys_reg, region_rows):
            ax.plot([bayes, wmb], [y, y], color="0.6", lw=1.1, zorder=1)
            ax.plot(bayes, y, "o", ms=5, color=_C_BAYES, zorder=2)
            ax.plot(
                wmb,
                y,
                "o",
                ms=5,
                mec=_C_WMB,
                mfc="white",
                mew=1.4,
                zorder=2,
            )
        for y, (_, bayes) in zip(ys_deg, degree_rows):
            ax.plot(bayes, y, "o", ms=5, color=_C_BAYES, zorder=2)
        if n_deg:
            ax.axhline(n_deg, color="0.8", lw=0.7, ls=":")
            if degree_label:
                ax.text(
                    0.98 * xmax,
                    ys_deg[0] + 0.55,
                    degree_label,
                    fontsize=7.5,
                    color="0.4",
                    ha="right",
                )
        ax.set_yticks(np.concatenate([ys_reg, ys_deg]))
        ax.set_yticklabels([r[0] for r in region_rows] + [d[0] for d in degree_rows])
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(-0.6, ys_reg[0] + 1.05)
        ax.set_xlabel("Relative estimator-kernel error  $\\|k - t\\|\\, /\\, \\|t\\|$")
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.annotate(
            "$\\leftarrow$ better",
            xy=(0.02 * xmax, ys_reg[0] + 0.62),
            fontsize=7.5,
            color="0.4",
        )
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                color=_C_BAYES,
                label="Bayesian",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                mec=_C_WMB,
                mfc="white",
                mew=1.4,
                label="WMB",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)
    return fig


_C_PRIOR = "#888888"
_C_ALT = "#D55E00"
_C_GRACE = "#0072B2"
_C_JOINT = "#009E73"


def plot_gmsl_split_ellipses(
    prior_split,
    alt_split,
    grace_split,
    joint_split,
    /,
    *,
    true_values=None,
    n_sigma=1.0,
):
    """Single-panel covariance-ellipse overlay for the (barystatic, steric) GMSLR split."""
    cases = [
        ("Prior", prior_split, dict(edgecolor=_C_PRIOR, lw=1.4, ls="-")),
        ("GRACE only", grace_split, dict(edgecolor=_C_GRACE, lw=1.8, ls="-")),
        ("Altimetry only", alt_split, dict(edgecolor=_C_ALT, lw=1.8, ls="--")),
        ("Joint", joint_split, dict(edgecolor=_C_JOINT, lw=1.8, ls="-")),
    ]
    with mpl.rc_context(_SUMMARY_RC):
        fig, ax = plt.subplots(figsize=(4.4, 4.4), layout="constrained")
        xlo = ylo = np.inf
        xhi = yhi = -np.inf
        handles = []
        for name, measure, style in cases:
            mean = np.asarray(measure.expectation, dtype=float).ravel()
            cov = np.asarray(measure.covariance.matrix(form="components"), dtype=float)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ax.add_patch(
                Ellipse(
                    mean,
                    2 * n_sigma * np.sqrt(vals[0]),
                    2 * n_sigma * np.sqrt(vals[1]),
                    angle=angle,
                    fill=False,
                    **style,
                )
            )
            ax.plot(*mean, ".", color=style["edgecolor"], ms=4)
            sx = n_sigma * np.sqrt(cov[0, 0])
            sy = n_sigma * np.sqrt(cov[1, 1])
            xlo, xhi = min(xlo, mean[0] - sx), max(xhi, mean[0] + sx)
            ylo, yhi = min(ylo, mean[1] - sy), max(yhi, mean[1] + sy)
            handles.append(
                Line2D(
                    [],
                    [],
                    color=style["edgecolor"],
                    lw=style["lw"],
                    ls=style["ls"],
                    label=name,
                )
            )

        if true_values is not None:
            tb, ts = np.asarray(true_values, dtype=float).ravel()[:2]
            ax.plot(tb, ts, "x", color="black", ms=7, mew=1.6, zorder=5)
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="x",
                    color="black",
                    ms=7,
                    mew=1.6,
                    label="True Value",
                )
            )
            xlo, xhi = min(xlo, tb), max(xhi, tb)
            ylo, yhi = min(ylo, ts), max(yhi, ts)
            sum_anchor = tb + ts
        else:
            sum_anchor = float(np.sum(np.asarray(joint_split.expectation, dtype=float)))

        pad = 0.10 * max(xhi - xlo, yhi - ylo)
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.set_ylim(ylo - pad, yhi + pad)

        gx = np.array(ax.get_xlim())
        span = max(xhi - xlo, yhi - ylo) + 2 * pad
        for k in (-1, 0, 1):
            ax.plot(
                gx,
                (sum_anchor + 0.55 * k * span) - gx,
                color="0.82",
                lw=0.7,
                ls="--",
                zorder=0,
            )
        handles.append(
            Line2D([], [], color="0.82", lw=0.9, ls="--", label="constant $b+s$")
        )

        ax.set_aspect("equal")
        ax.set_xlabel("Barystatic GMSLR contribution, $b$ (mm)")
        ax.set_ylabel("Steric GMSLR contribution, $s$ (mm)")
        ax.legend(handles=handles, loc="best", frameon=False)
        ax.text(
            0.02,
            0.02,
            f"{n_sigma:g}$\\sigma$ contours",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            va="bottom",
        )
    return fig


def plot_component_variance_dots(
    component_names,
    prior_vars,
    alt_vars,
    grace_vars,
    joint_vars,
    /,
    *,
    xlabel="Marginal variance (mm$^2$)",
):
    """Log-scale dot chart of marginal variances per component."""
    series = [
        ("Prior", prior_vars, dict(marker="o", color=_C_PRIOR, mfc="white"), 0.24),
        ("Altimetry only", alt_vars, dict(marker="^", color=_C_ALT, mfc=_C_ALT), 0.08),
        (
            "GRACE only",
            grace_vars,
            dict(marker="s", color=_C_GRACE, mfc=_C_GRACE),
            -0.08,
        ),
        ("Joint", joint_vars, dict(marker="D", color=_C_JOINT, mfc=_C_JOINT), -0.24),
    ]
    n = len(component_names)
    with mpl.rc_context(_SUMMARY_RC):
        fig, ax = plt.subplots(figsize=(5.2, 1.1 + 0.42 * n), layout="constrained")
        ys = np.arange(n)[::-1]
        for y in ys:
            ax.axhline(y, color="0.92", lw=4, zorder=0)
        handles = []
        for name, vals, style, dodge in series:
            ax.plot(
                np.asarray(vals, dtype=float),
                ys + dodge,
                ls="none",
                ms=4.5,
                mew=1.1,
                marker=style["marker"],
                mec=style["color"],
                mfc=style["mfc"],
            )
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    ms=5,
                    mew=1.1,
                    marker=style["marker"],
                    mec=style["color"],
                    mfc=style["mfc"],
                    label=name,
                )
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(list(component_names))
        ax.set_ylim(-0.55, n - 0.45)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.legend(
            handles=handles,
            ncols=4,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
        )
    return fig
