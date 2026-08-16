"""
joint_utils.py
=======================
Shared utilities, physics initializations, and Bayesian measures for
extended Joint (Altimetry + GRACE) inversions using a 3-component model.

Model space (three fields on the sphere):

    m = [ Dh, eta, drho ]

      Dh   : ice thickness change (grounded ice),
      eta  : sterodynamic sea level change (the TOTAL dynamic
             perturbation to the sea surface), applying the load rho_w*C*eta,
      drho : vertically averaged ocean density change, applying the load
             D*C*drho (the column mass change at fixed column thickness).

The prior enforces <rho_w*C*eta + D*C*drho>_O = 0, so the combined ocean
load is a pure redistribution of ocean mass. GRACE observes the response to
the total load; altimetry observes the GRD response plus the direct dynamic
term eta.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    ocean_density_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    grace_observation_operator,
    WMBMethod,
)

from altimetry_utils import (
    build_conditioned_prior as build_conditioned_prior,
    print_calibration_report as print_calibration_report,
    barystatic_gmsl_weighting as barystatic_gmsl_weighting,
    effective_steric_scale as effective_steric_scale,
    gmsl_split_operators as gmsl_split_operators,
    regional_decomposition_operators as regional_decomposition_operators,
    steric_sea_level_operator as steric_sea_level_operator,
    true_gmsl_operator as true_gmsl_operator,
)

from grace_utils import resolve_noise_amplitude


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
    """Constructs the 3-component physical operators for Joint Inversion."""
    state = EarthState.from_defaults(lmax=lmax)
    scale_mm = 1000.0 * state.model.parameters.length_scale
    load_scale = load_scale_km * 1000.0 / state.model.parameters.length_scale

    max_iters = 1 if is_surrogate else None

    finger_print_operator = FingerPrintOperator(
        state,
        load_parameters=(load_order, load_scale),
        response_parameters=(load_order, load_scale),
        max_iterations=max_iters,
    )
    load_space = finger_print_operator.domain

    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)

    if is_surrogate:
        ocean_dyn_to_load = load_space.zero_operator(load_space)
        ocean_rho_to_load = load_space.zero_operator(load_space)
    else:
        ocean_dyn_to_load = sea_level_change_to_load_operator(
            state, load_space, load_space
        )
        ocean_rho_to_load = ocean_density_change_to_load_operator(
            state, load_space, load_space
        )

    joint_to_load = inf.RowLinearOperator(
        [ice_to_load, ocean_dyn_to_load, ocean_rho_to_load]
    )
    joint_space = joint_to_load.domain

    # --- A. ALTIMETRY TRACK (SSH) ---
    op1_alt = inf.ColumnLinearOperator(
        [joint_to_load, joint_space.subspace_projection(1)]
    )
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    barystatic_ssh_op = static_ssh_op @ finger_print_operator

    op2_ssh = inf.BlockDiagonalLinearOperator(
        [barystatic_ssh_op, load_space.identity_operator()]
    )
    op3_alt = inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.identity_operator()]
    )

    continuous_ssh_operator = op3_alt @ op2_ssh @ op1_alt
    point_eval = load_space.point_evaluation_operator(points)
    alt_track = point_eval @ continuous_ssh_operator

    # --- B. GRACE TRACK ---
    grace_obs_op = grace_observation_operator(
        finger_print_operator.codomain, obs_degree
    )
    grace_track = grace_obs_op @ finger_print_operator @ joint_to_load

    # --- C. TRUE SEA LEVEL TRACK (For GMSLR) ---
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )
    op2_sl = inf.BlockDiagonalLinearOperator(
        [barystatic_sl_op, load_space.identity_operator()]
    )
    continuous_sl_operator = op3_alt @ op2_sl @ op1_alt

    # --- D. FULL JOINT FORWARD OPERATOR ---
    joint_forward_operator = inf.ColumnLinearOperator([alt_track, grace_track])

    return {
        "state": state,
        "load_space": load_space,
        "fp_op": finger_print_operator,
        "alt_track": alt_track,
        "grace_track": grace_track,
        "joint_forward": joint_forward_operator,
        "continuous_ssh": continuous_ssh_operator,
        "continuous_sl": continuous_sl_operator,
        "point_eval": point_eval,
        "scale_mm": scale_mm,
    }


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    gmsl_bary_steric_ratio,
    ocean_dyn_scale_factor,
    ocean_dyn_std_mm,
    ocean_rho_scale_factor,
    steric_dyn_std_ratio,
    alt_noise_corr_scale_factor,
    alt_noise_std_factor,
    grace_noise_scale_km,
    obs_degree,
    points,
    scale_mm,
    grace_snr_crossover_degree,
    grace_snr_low_degree,
    grace_snr_reference_degree,
    /,
    *,
    prior_shift=0.0,
    is_surrogate=False,
    ocean_corr=0.0,
    ocean_corr_scale_factor=0.4,
    prior_order=3.0,
    alt_noise_corr_std_factor=0.0,
    point_evaluation_operator=None,
    derived_stds=None,
):
    """Constructs the 3-component joint prior and dual-sensor noise measures."""

    if not 0.0 <= ocean_corr < 1.0:
        raise ValueError(
            "ocean_corr must lie in [0, 1); the anti-correlation sign is "
            "applied internally."
        )

    if alt_noise_corr_std_factor < 0.0:
        raise ValueError("alt_noise_corr_std_factor must be non-negative.")

    def load_measure(scale, std):
        return load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            prior_order, scale, std=std
        )

    # --- 1. PRIORS: amplitude derivation, assembly, masking and conditioning ---
    ice_scale = load_space.scale * ice_scale_factor
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor

    model_prior, unmasked_prior, calib = build_conditioned_prior(
        state,
        load_space,
        load_measure,
        ice_scale,
        ocean_dyn_scale,
        ocean_rho_scale,
        gmsl_bary_steric_ratio,
        ocean_dyn_std_mm / scale_mm,
        steric_dyn_std_ratio,
        is_surrogate=is_surrogate,
        ocean_corr=ocean_corr,
        corr_scale=load_space.scale * ocean_corr_scale_factor,
        derived_stds=derived_stds,
    )
    ice_std = calib["derived_stds"]["ice_std"]

    if prior_shift != 0.0:
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, model_prior.sample())
        )

    if is_surrogate:
        return {
            "model_prior": model_prior,
            "unmasked_prior": unmasked_prior,
            "calib": calib,
        }

    # --- 2. NOISE MEASURES ---
    alt_noise_std = alt_noise_std_factor * calib["derived_stds"]["dyn_std"]
    alt_precond_noise_meas = inf.GaussianMeasure.from_standard_deviation(
        inf.EuclideanSpace(len(points)), alt_noise_std
    )
    alt_noise_meas = alt_precond_noise_meas
    if alt_noise_corr_std_factor > 0.0:
        if point_evaluation_operator is None:
            point_evaluation_operator = load_space.point_evaluation_operator(points)
        alt_corr_field_meas = load_measure(
            load_space.scale * alt_noise_corr_scale_factor,
            alt_noise_corr_std_factor * calib["derived_stds"]["dyn_std"],
        )
        gmsl_fn = load_space.l2_products_operator(
            [state.ocean_projection(value=0.0) / state.ocean_area]
        )
        calib.setdefault("noise", {})["alt_corr_gmsl_floor"] = float(
            np.sqrt(
                alt_corr_field_meas.directional_variance(gmsl_fn.adjoint(np.ones(1)))
            )
        )
        alt_corr_noise_meas = alt_corr_field_meas.affine_mapping(
            operator=point_evaluation_operator
        )
        alt_noise_meas = alt_noise_meas + alt_corr_noise_meas

    # GRACE Noise
    grace_spatial_scale = (
        grace_noise_scale_km * 1000.0 / state.model.parameters.length_scale
    )
    ice_load_std = (
        ice_std
        * state.model.parameters.ice_density
        / state.model.parameters.water_density
    )
    grace_noise_info = resolve_noise_amplitude(
        load_space,
        prior_order,
        ice_scale,
        ice_load_std,
        grace_spatial_scale,
        grace_snr_crossover_degree,
        grace_snr_low_degree,
        snr_reference_degree=grace_snr_reference_degree,
        obs_degree=obs_degree,
        std_to_mm=scale_mm,
        label="GRACE spatial noise",
    )

    _b = grace_noise_info["kernel_amplitude"]
    _q = grace_noise_info["noise_order"]
    grace_spatial_noise = load_space.invariant_gaussian_measure(
        lambda k, b=_b, s=grace_spatial_scale, q=_q: (b * (1.0 + s * s * k) ** (-q))
    )

    wmb = WMBMethod(state.model, obs_degree)
    grace_noise_meas = wmb.load_measure_to_observation_measure(grace_spatial_noise)

    joint_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_noise_meas, grace_noise_meas]
    )
    joint_precond_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_precond_noise_meas, grace_noise_meas]
    )

    return {
        "model_prior": model_prior,
        "unmasked_prior": unmasked_prior,
        "alt_noise": alt_noise_meas,
        "grace_noise": grace_noise_meas,
        "joint_noise": joint_noise_meas,
        "alt_precond_noise": alt_precond_noise_meas,
        "joint_precond_noise": joint_precond_noise_meas,
        "wmb": wmb,
        "gmsl_std": calib.get("gmsl", {}).get("barystatic_std"),
        "grace_noise_info": grace_noise_info,
        "ice_std": ice_std,
        "ice_scale": ice_scale,
        "calib": calib,
    }


# ---------------------------------------------------------------------------
# Comparison-summary figures
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
            cov = np.asarray(measure.covariance.matrix(dense=True), dtype=float)
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
