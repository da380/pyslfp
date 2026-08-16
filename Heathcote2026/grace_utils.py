"""
grace_utils.py
==============
Shared utilities, physics initializations, and plotting for Bayesian GRACE inversions.

The direct load prior and the spatial noise measure use Sobolev (Matern-type)
covariances, giving rougher, power-law-tailed fields. The noise spectrum
(its amplitude and Sobolev exponent) is solved in closed form from a specified
low-degree signal-to-noise ratio and the degree at which the spectral SNR crosses one.
"""

import pygeoinf as inf

from pyslfp import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    sea_level_change_to_load_operator,
    averaging_operator,
)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def build_physics_components(lmax, load_order, load_scale_km):
    """Initialises the Earth state, Sobolev spaces, and fingerprint operator."""
    state = EarthState.from_defaults(lmax=lmax)

    length_scale = state.model.parameters.length_scale
    water_density = state.model.parameters.water_density

    load_to_water_thickness_mm = 1000 * length_scale / water_density
    load_space_scale = load_scale_km * 1000 / length_scale

    finger_print_operator = FingerPrintOperator(
        state, load_parameters=(load_order, load_space_scale)
    )

    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    return (
        state,
        load_space,
        response_space,
        finger_print_operator,
        load_to_water_thickness_mm,
    )


def invariant_coefficient_profile(
    order, scale, space_order, space_scale, degrees, /, *, radius=1.0
):
    """Unnormalised per-coefficient L2 variances c_l at the given degrees (Sobolev kernel)."""
    degrees = np.asarray(degrees, dtype=float)
    lam = degrees * (degrees + 1.0) / radius**2
    space_weight = (1.0 + space_scale**2 * lam) ** (-space_order)
    return (1.0 + scale**2 * lam) ** (-order) * space_weight


def resolve_noise_amplitude(
    load_space,
    prior_order,
    signal_scale,
    signal_std,
    noise_scale,
    snr_crossover_degree,
    snr_low_degree,
    /,
    *,
    snr_reference_degree=2.0,
    obs_degree=None,
    std_to_mm=None,
    label="spatial noise",
):
    """
    Resolves the noise spectrum (exponent and amplitude) and reports the
    spectral signal-to-noise structure of a (signal, noise) pair of
    invariant Sobolev kernel measures on the same space.

    The exponent and amplitude are solved in closed form using two
    log-linear conditions: an amplitude SNR = snr_low_degree at degree
    snr_reference_degree, and SNR = 1 at the crossover.

    The noise must be a legitimate random field (q_n > 1) for its
    covariance operator to be trace-class. This mathematically caps
    the low-degree contrast; infeasible requests raise with the cap reported.
    """
    if snr_crossover_degree is None or snr_low_degree is None:
        raise ValueError(
            "Both snr_crossover_degree and snr_low_degree must be provided to resolve the noise spectrum."
        )

    space_order = load_space.order
    space_scale = load_space.scale
    lmax = load_space.lmax
    radius = load_space.radius
    q_s = prior_order

    degrees = np.arange(lmax + 1, dtype=float)
    weights = 2.0 * degrees + 1.0

    def lam(ell):
        ell = np.asarray(ell, dtype=float)
        return ell * (ell + 1.0) / radius**2

    def log_kernel(scale, ell):
        return float(np.log1p(scale**2 * lam(ell)))

    # Signal spectrum and its kernel amplitude
    prof_s = invariant_coefficient_profile(
        q_s, signal_scale, space_order, space_scale, degrees, radius=radius
    )
    z_s = float(np.sum(weights * prof_s))
    kernel_amplitude_signal = 4.0 * np.pi * signal_std**2 / z_s

    band_degree = int(min(obs_degree, lmax)) if obs_degree is not None else lmax
    ref_degree = float(snr_reference_degree)
    lstar = float(snr_crossover_degree)

    # if not 0.0 < lstar <= band_degree:
    #    raise ValueError(f"snr_crossover_degree must lie in (0, {band_degree}].")
    if ref_degree < 1.0:
        raise ValueError("snr_reference_degree must be >= 1.")
    if not ref_degree < lstar:
        raise ValueError("snr_reference_degree must lie below snr_crossover_degree.")
    if not snr_low_degree > 1.0:
        raise ValueError("snr_low_degree must exceed 1.")

    # Two-condition solve for (q_n, B)
    ln_rho_var = 2.0 * np.log(snr_low_degree)
    dl_s = log_kernel(signal_scale, lstar) - log_kernel(signal_scale, ref_degree)
    dl_n = log_kernel(noise_scale, lstar) - log_kernel(noise_scale, ref_degree)
    q_n = float((q_s * dl_s - ln_rho_var) / dl_n)

    # Legitimate Field Trace-Class Check (q_n > 1)
    if q_n <= 1.0 + 1.0e-12:
        rho_cap = np.exp((q_s * dl_s - dl_n) / 2.0)
        raise ValueError(
            f"The solved noise exponent q_n = {q_n:.3f} gives an "
            "illegitimate field (needs q_n > 1). For this crossover "
            "degree and these scales the low-degree amplitude SNR must be "
            f"below {rho_cap:.2f}."
        )

    lin_0 = q_s * signal_scale**2 - q_n * noise_scale**2
    lin_1 = (q_s - q_n) * signal_scale**2 * noise_scale**2
    monotone = lin_0 >= 0.0 and lin_1 >= 0.0 and (lin_0 + lin_1) > 0.0

    if not monotone:
        raise ValueError(
            "The SNR crossover requires a monotonically decreasing "
            "signal-to-noise ratio: q_n <= q_s and q_n * s_n^2 <= "
            "q_s * s_s^2 (reduce noise_scale relative to signal_scale)."
        )

    log_b = (
        np.log(kernel_amplitude_signal)
        - q_s * log_kernel(signal_scale, lstar)
        + q_n * log_kernel(noise_scale, lstar)
    )
    kernel_amplitude = float(np.exp(log_b))

    prof_n = invariant_coefficient_profile(
        q_n, noise_scale, space_order, space_scale, degrees, radius=radius
    )
    z_n = float(np.sum(weights * prof_n))
    noise_std = float(np.sqrt(kernel_amplitude * z_n / (4.0 * np.pi)))

    kern_s = invariant_coefficient_profile(
        q_s, signal_scale, 0.0, space_scale, degrees, radius=radius
    )
    kern_n = invariant_coefficient_profile(
        q_n, noise_scale, 0.0, space_scale, degrees, radius=radius
    )
    nsr = (kernel_amplitude * kern_n) / (kernel_amplitude_signal * kern_s)
    snr_amp_ref = float(1.0 / np.sqrt(nsr[int(round(ref_degree))]))

    band_fraction = float(
        np.sum(weights[: band_degree + 1] * prof_n[: band_degree + 1]) / z_n
    )

    info = {
        "kernel_amplitude": float(kernel_amplitude),
        "noise_std": float(noise_std),
        "std_factor": float(noise_std / signal_std),
        "noise_order": q_n,
        "crossover_degree": lstar,
        "snr_low_degree": snr_amp_ref,
        "snr_reference_degree": ref_degree,
        "band_degree": band_degree,
        "band_variance_fraction": band_fraction,
        "band_std": float(noise_std * np.sqrt(band_fraction)),
        "nsr_band_edge": float(nsr[band_degree]),
        "monotone": monotone,
        "legitimate_field": True,  # ValueError acts as guardrail above
        "plot_smoothing_scale": (
            radius * float(np.sqrt(np.log(2.0) / (lstar * (lstar + 1.0))))
        ),
    }

    if std_to_mm:
        in_mm = lambda x: f"{x * std_to_mm:.3f} mm"  # noqa: E731
    else:
        in_mm = lambda x: f"{x:.4e}"  # noqa: E731

    print(
        f"{label}: Sobolev, scale ratio s_n/s_s = " f"{noise_scale / signal_scale:.2f}"
    )
    print(
        f"  solved from amplitude SNR = {snr_amp_ref:.2f} at degree "
        f"{ref_degree:g} and crossover at degree {lstar:g}:"
    )
    print(f"  noise order q_n = {q_n:+.3f} " f"(q_n > 1: legitimate field)")
    print(
        f"  pointwise std = {in_mm(noise_std)} "
        f"({info['std_factor']:.4f} x signal std); "
        f"{100 * band_fraction:.1f}% of the noise variance lies in the "
        f"band l <= {band_degree} at lmax = {lmax} "
        f"(band std = {in_mm(info['band_std'])})"
    )
    print(
        f"  noise/signal amplitude = {np.sqrt(info['nsr_band_edge']):.1f} "
        f"at l = {band_degree}"
    )
    return info


def build_measures(
    state,
    load_space,
    direct_scale_km,
    direct_std_m,
    noise_scale_factor,
    /,
    *,
    remove_degree_1=False,
    prior_shift=0.0,
    prior_order=1.0,
    snr_crossover_degree=None,
    snr_low_degree=None,
    snr_reference_degree=2.0,
    obs_degree=None,
):
    """
    Constructs the prior and noise Gaussian measures using Sobolev
    (Matern-type) covariances.

    The noise spectrum is solved in closed form from the amplitude SNR
    at a low reference degree and the degree where the per-coefficient
    variances are equal. The noise is required to be a legitimate random
    field (q_n > 1), which caps the achievable low-degree SNR.
    """
    if prior_order < 1.0:
        raise ValueError("prior_order must be greater than one.")

    length_scale = state.model.parameters.length_scale
    water_density = state.model.parameters.water_density

    direct_load_measure_scale = direct_scale_km * 1000 / length_scale
    direct_load_measure_std = water_density * direct_std_m / length_scale

    initial_direct_load_prior = (
        load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            prior_order, direct_load_measure_scale, std=direct_load_measure_std
        )
    )

    if remove_degree_1:
        constraint_lmax = 1
        constraint_operator = load_space.to_coefficient_operator(constraint_lmax)
        constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
        direct_load_prior = constraint_subspace.condition_gaussian_measure(
            initial_direct_load_prior
        )
    else:
        direct_load_prior = initial_direct_load_prior

    if prior_shift != 0.0:
        offset_shape = direct_load_prior.sample()
        direct_load_prior = direct_load_prior.affine_mapping(
            translation=offset_shape * prior_shift
        )

    noise_load_measure_scale = noise_scale_factor * direct_load_measure_scale

    noise_info = resolve_noise_amplitude(
        load_space,
        prior_order,
        direct_load_measure_scale,
        direct_load_measure_std,
        noise_load_measure_scale,
        snr_crossover_degree,
        snr_low_degree,
        snr_reference_degree=snr_reference_degree,
        obs_degree=obs_degree,
        std_to_mm=1000.0 * length_scale / water_density,
        label="GRACE spatial noise",
    )

    # The spectrum is fixed directly by its kernel amplitude, with no
    # pointwise normalisation of the rough field.
    _b = noise_info["kernel_amplitude"]
    _q = noise_info["noise_order"]
    noise_load_measure = load_space.invariant_gaussian_measure(
        lambda k, b=_b, s=noise_load_measure_scale, q=_q: (
            b * (1.0 + s * s * k) ** (-q)
        )
    )

    return (
        initial_direct_load_prior,
        direct_load_prior,
        noise_load_measure,
        noise_load_measure_scale,
        noise_info,
    )


def build_total_load_operator(state, response_space, load_space, finger_print_operator):
    """Builds the operator linking direct load to total physical load (including SLE)."""
    sea_level_projection = response_space.subspace_projection(0)
    sea_level_to_load = sea_level_change_to_load_operator(
        state, sea_level_projection.codomain, load_space
    )
    induced_load_operator = (
        sea_level_to_load @ sea_level_projection @ finger_print_operator
    )
    return load_space.identity_operator() + induced_load_operator


def get_regional_averaging(
    state, load_space, /, *, regions_dict=None, smoothing_scale_km=None
):
    """Sets up the averaging operator using specialised geophysical basins."""

    if regions_dict is None:
        regions_dict = {
            "GRL (S Basin)": ["SE", "SW"],
            "WAIS": ["F-G", "Ep-F", "J-Jpp", "G-H", "H-Hp"],
            "Gulf of Mexico": "Gulf of Mexico",
            "GBM basin": "4030025450",
        }

    region_names = list(regions_dict.keys())

    weighting_functions = [
        state.get_projection(raw_names, value=0.0)
        for raw_names in regions_dict.values()
    ]

    if smoothing_scale_km is not None and smoothing_scale_km > 0:
        smoothing_scale = (
            smoothing_scale_km * 1000 / state.model.parameters.length_scale
        )
        smoothing_measure = load_space.heat_kernel_gaussian_measure(smoothing_scale)
        smoothing_operator = smoothing_measure.covariance
        weighting_functions = [smoothing_operator(wf) for wf in weighting_functions]

    avg_operator = averaging_operator(state, load_space, weighting_functions)

    return (
        region_names,
        avg_operator,
        weighting_functions,
        regions_dict,
    )


def draw_region_boundaries(state, ax, regions_dict, **kwargs):
    """
    Helper to plot all boundaries defined in a regions dictionary.
    Handles nested lists automatically for composite regions.
    """
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("linewidth", 2.0)

    raw_regions = []
    for val in regions_dict.values():
        if isinstance(val, str):
            raw_regions.append(val)
        else:
            raw_regions.extend(val)

    state.plot_boundaries(ax, raw_regions, **kwargs)


# ---------------------------------------------------------------------------
# Comparison-summary figures.
# Styled via rc_context so the calling script's global rcParams do not
# leak in; sized for single-column use. Style block duplicated in
# joint_utils to keep the two pipelines independent.
# ---------------------------------------------------------------------------

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
