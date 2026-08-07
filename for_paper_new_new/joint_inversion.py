"""
Joint Bayesian Inversion (GRACE + Satellite Altimetry)
===============================================================

This script performs a joint Bayesian inversion of synthetic GRACE gravimetry
and satellite altimetry data to estimate ice sheet mass loss, the total ocean
dynamic sea surface height change, and vertically averaged ocean density
changes. Quantities of interest are post-processed into the standard
barystatic / steric / ocean-dynamic split (see joint_utils).

It runs head-to-head Altimetry-only, GRACE-only, and Joint (Alt + GRACE)
inversions on the exact same physical scenario to demonstrate the added value
of each sensor, including a push-forward onto the 2D split of GMSL change
into barystatic and steric contributions, and an optionally anti-correlated
(Dyn, Rho) prior.
"""

import argparse
import os
import numpy as np

import matplotlib


import matplotlib.pyplot as plt
import pygeoinf as inf

import joint_utils as utils

import pyslfp as sl
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points

matplotlib.use("Agg")


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
    }
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian Inversion (Ice, Dyn, Rho) vs single-sensor."
    )
    # --- Output Options ---
    parser.add_argument(
        "--all", action="store_true", help="Enable all plotting options."
    )
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot 1D analytical PDFs of the GMSL estimate.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps (True vs Joint / Alt-only / GRACE-only) & Covariances.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot regional 3-way signal decomposition.",
    )
    parser.add_argument(
        "--plot-gmsl-split",
        action="store_true",
        help="Push forward onto the 2D GMSL split (barystatic vs steric).",
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Number of samples for pointwise std estimates (applied to each posterior).",
    )

    # --- Resolution Settings ---
    parser.add_argument(
        "--lmax", type=int, default=128, help="Exact model max SH degree."
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Preconditioner max SH degree."
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Max SH degree of GRACE observations.",
    )
    parser.add_argument(
        "--load-order", type=float, default=2.0, help="Sobolev space order."
    )
    parser.add_argument(
        "--load-scale-km", type=float, default=500.0, help="Sobolev length scale."
    )
    parser.add_argument(
        "--spacing", type=float, default=4.0, help="Altimetry observation spacing."
    )

    # --- Prior Settings ---
    parser.add_argument(
        "--ice-scale-factor", type=float, default=1.0, help="Ice correlation scale."
    )
    parser.add_argument(
        "--ice-std-mm", type=float, default=10.0, help="Ice std dev (mm)."
    )

    parser.add_argument(
        "--ocean-dyn-scale-factor",
        type=float,
        default=0.2,
        help="Ocean dynamic correlation scale.",
    )
    parser.add_argument(
        "--ocean-dyn-std-factor",
        type=float,
        default=2.0,
        help="Ocean dynamic std as factor of GMSL std.",
    )

    parser.add_argument(
        "--ocean-rho-scale-factor",
        type=float,
        default=1.0,
        help="Ocean density correlation scale.",
    )
    parser.add_argument(
        "--ocean-rho-std-factor",
        type=float,
        default=0.25,
        help="Effective steric SL std as a fraction of the dynamic SSH std.",
    )

    parser.add_argument(
        "--alt-noise-scale-factor",
        type=float,
        default=0.0,
        help="Alt instrument noise correlation scale.",
    )
    parser.add_argument(
        "--alt-noise-std-factor",
        type=float,
        default=1.0,
        help="Alt instrument noise std as factor of GMSL std.",
    )

    parser.add_argument(
        "--grace-noise-scale-km",
        type=float,
        default=50.0,
        help="GRACE spatial noise correlation scale (km).",
    )
    parser.add_argument(
        "--grace-noise-std-factor",
        type=float,
        default=0.1,
        help="GRACE spatial noise std as factor of ice std.",
    )

    parser.add_argument(
        "--ocean-corr",
        type=float,
        default=0.9,
        help=(
            "Magnitude of the long-wavelength anti-correlation between the "
            "ocean dynamic and density fields (0 disables; must lie in "
            "[0, 1); the sign is applied internally."
        ),
    )
    parser.add_argument(
        "--ocean-corr-scale-factor",
        type=float,
        default=0.4,
        help=(
            "Decay scale of the (Dyn, Rho) spectral correlation with "
            "wavenumber, as a factor of the load scale."
        ),
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )

    return parser.parse_args()


def compute_pointwise_std(load_space, post_meas, expectation, n_samples, n_jobs=10):
    """
    Sample-based pointwise standard deviation of a 3-component posterior
    measure about its expectation. Returns (std_ice, std_dyn, std_rho).
    """
    post_ice, post_dyn, post_rho = expectation
    samples = post_meas.samples(n_samples, parallel=True, n_jobs=n_jobs)

    v_ice = load_space.zero
    v_dyn = load_space.zero
    v_rho = load_space.zero

    for s_ice, s_dyn, s_rho in samples:
        diff_ice = load_space.subtract(s_ice, post_ice)
        load_space.axpy(
            1.0 / n_samples,
            load_space.vector_multiply(diff_ice, diff_ice),
            v_ice,
        )

        diff_dyn = load_space.subtract(s_dyn, post_dyn)
        load_space.axpy(
            1.0 / n_samples,
            load_space.vector_multiply(diff_dyn, diff_dyn),
            v_dyn,
        )

        diff_rho = load_space.subtract(s_rho, post_rho)
        load_space.axpy(
            1.0 / n_samples,
            load_space.vector_multiply(diff_rho, diff_rho),
            v_rho,
        )

    return (
        load_space.vector_sqrt(v_ice),
        load_space.vector_sqrt(v_dyn),
        load_space.vector_sqrt(v_rho),
    )


def plot_state_maps(
    state,
    true_fields,
    post_fields,
    std_fields,
    vmaxes,
    std_vmaxes,
    ocean_mask_mm,
    ice_mask_mm,
    ocean_mask_gm3,
    scale_mm,
    post_label,
    regions=None,
):
    """
    3-row map figure comparing the true state with a posterior expectation,
    with an optional third column of pointwise standard deviations.
    Rows: ice thickness (mm), dynamic SSH (mm), vertically averaged density
    (g/m^3). The supplied vmaxes are shared across figures so that the
    different posteriors can be compared on identical colour scales.
    """
    cmap = "seismic"
    cmap_std = "Blues"
    gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}
    cb_kwargs = {"orientation": "horizontal", "shrink": 0.8, "pad": 0.05}

    plot_std = std_fields is not None
    ncols = 3 if plot_std else 2
    fig_width = 22 if plot_std else 14

    true_ice, true_dyn, true_rho = true_fields
    post_ice, post_dyn, post_rho = post_fields
    vmax_ice, vmax_dyn, vmax_rho = vmaxes

    fig, axes = sl.subplots(
        3, ncols, figsize=(fig_width, 15), gridspec_kw={"hspace": 0.15}
    )

    row_specs = [
        (
            true_ice * scale_mm,
            post_ice * scale_mm,
            "Ice Thickness (mm)",
            vmax_ice,
        ),
        (
            true_dyn * ocean_mask_mm,
            post_dyn * ocean_mask_mm,
            "Dynamic SSH (mm)",
            vmax_dyn,
        ),
        (
            true_rho * ocean_mask_gm3,
            post_rho * ocean_mask_gm3,
            r"Density Change (g m$^{-3}$)",
            vmax_rho,
        ),
    ]

    for i, (t_field, p_field, label, vmax) in enumerate(row_specs):
        for j, field in enumerate([t_field, p_field]):
            sl.plot(
                field,
                ax=axes[i, j],
                colorbar=True,
                vmin=-vmax,
                vmax=vmax,
                cmap=cmap,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )

    if plot_std:
        std_ice, std_dyn, std_rho = std_fields
        vmax_std_ice, vmax_std_dyn, vmax_std_rho = std_vmaxes
        std_specs = [
            (std_ice * ice_mask_mm, "Ice STD (mm)", vmax_std_ice),
            (std_dyn * ocean_mask_mm, "Dynamic SSH STD (mm)", vmax_std_dyn),
            (
                std_rho * ocean_mask_gm3,
                r"Density STD (g m$^{-3}$)",
                vmax_std_rho,
            ),
        ]
        for i, (field, label, vmax) in enumerate(std_specs):
            sl.plot(
                field,
                ax=axes[i, 2],
                colorbar=True,
                cmap=cmap_std,
                vmin=0,
                vmax=vmax,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )

    if regions:
        for ax in axes.flatten():
            state.plot_boundaries(ax, regions)

    # Labels
    col_labels = ["True State", post_label]
    if plot_std:
        col_labels.append("Pointwise Std. Deviation")

    for j in range(ncols):
        axes[0, j].set_title(col_labels[j], fontsize=16, fontweight="bold", pad=20)

    row_titles = ["Ice thickness", "Dynamic SSH", "Density"]
    for i, row_title in enumerate(row_titles):
        axes[i, 0].annotate(
            row_title,
            xy=(-0.1, 0.5),
            xycoords="axes fraction",
            fontsize=16,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )

    return fig


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        args.plot_gmsl_split = True

    output_dir = "output_plots_joint_inversion"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    metrics_file = os.path.join(output_dir, "joint_metrics.txt")
    with open(metrics_file, "w") as f_metrics:
        f_metrics.write("Joint Inversion (Alt + GRACE) Metrics\n")
        f_metrics.write("======================================\n\n")

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    inf.configure_threading(n_threads=1)
    regions_to_analyze = ["Tasman Sea"]

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(
        f"\nBuilding EXACT 3-Component joint physical operators (lmax={args.lmax})..."
    )
    exact_phys = utils.build_physics_components(
        args.lmax,
        args.load_order,
        args.load_scale_km,
        points,
        args.obs_degree,
        is_surrogate=False,
    )

    exact_meas = utils.build_measures(
        exact_phys["state"],
        exact_phys["load_space"],
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.alt_noise_scale_factor,
        args.alt_noise_std_factor,
        args.grace_noise_scale_km,
        args.grace_noise_std_factor,
        args.obs_degree,
        points,
        exact_phys["scale_mm"],
        prior_shift=args.prior_shift,
        is_surrogate=False,
        ocean_corr=args.ocean_corr,
        ocean_corr_scale_factor=args.ocean_corr_scale_factor,
    )

    scale_mm = exact_phys["scale_mm"]
    print(
        f"Implied GMSL prior standard deviation: {exact_meas['gmsl_std'] * scale_mm:.3f} mm"
    )
    if args.ocean_corr > 0.0:
        print(
            f"Correlated (Dyn, Rho) prior enabled: r0 = {args.ocean_corr:.2f}, "
            f"correlation scale factor = {args.ocean_corr_scale_factor:.2f}"
        )

    ocean_mask_mm = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
    ice_mask_mm = scale_mm * exact_phys["state"].ice_projection(value=0.0)

    # Unitless masks for covariance plotting
    ocean_mask = exact_phys["state"].ocean_projection(value=0.0)
    ice_mask = exact_phys["state"].ice_projection(value=0.0)

    print("\nDrawing MASTER synthetic model and dataset...")
    master_forward = inf.LinearForwardProblem(
        exact_phys["joint_forward"], data_error_measure=exact_meas["joint_noise"]
    )
    true_model, joint_data = master_forward.synthetic_model_and_data(
        exact_meas["model_prior"]
    )
    alt_data = joint_data[0]
    grace_data = joint_data[1]

    # ------------------ 2. SETUP INVERSE PROBLEMS ------------------
    joint_inv = inf.LinearBayesianInversion(master_forward, exact_meas["model_prior"])

    alt_fwd = inf.LinearForwardProblem(
        exact_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
    )
    alt_inv = inf.LinearBayesianInversion(alt_fwd, exact_meas["model_prior"])

    grace_fwd = inf.LinearForwardProblem(
        exact_phys["grace_track"], data_error_measure=exact_meas["grace_noise"]
    )
    grace_inv = inf.LinearBayesianInversion(grace_fwd, exact_meas["model_prior"])

    # ------------------ 3. PRECONDITIONER SETUP ------------------
    print(
        f"\nBuilding SURROGATE operators (lmax={args.surrogate_degree}) for preconditioning..."
    )
    surr_phys = utils.build_physics_components(
        args.surrogate_degree,
        args.load_order,
        args.load_scale_km,
        points,
        args.obs_degree,
        is_surrogate=True,
    )

    surr_meas = utils.build_measures(
        surr_phys["state"],
        surr_phys["load_space"],
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.alt_noise_scale_factor,
        args.alt_noise_std_factor,
        args.grace_noise_scale_km,
        args.grace_noise_std_factor,
        args.obs_degree,
        points,
        surr_phys["scale_mm"],
        prior_shift=args.prior_shift,
        is_surrogate=True,
        ocean_corr=args.ocean_corr,
        ocean_corr_scale_factor=args.ocean_corr_scale_factor,
    )

    woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=8)
    alpha = 0.1

    P_joint = (1 - alpha) * joint_inv.surrogate_woodbury_data_preconditioner(
        woodbury_solver,
        alternate_forward_operator=surr_phys["joint_forward"],
        alternate_prior_measure=surr_meas["unmasked_prior"],
        alternate_data_error_measure=exact_meas["joint_noise"],
    ) + alpha * exact_meas["joint_noise"].inverse_covariance

    surr_alt_fwd = inf.LinearForwardProblem(
        surr_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
    )
    surr_alt_inv = inf.LinearBayesianInversion(
        surr_alt_fwd, surr_meas["unmasked_prior"]
    )
    P_alt = (1 - alpha) * surr_alt_inv.woodbury_data_preconditioner(
        woodbury_solver
    ) + alpha * exact_meas["alt_noise"].inverse_covariance

    surr_grace_fwd = inf.LinearForwardProblem(
        surr_phys["grace_track"], data_error_measure=exact_meas["grace_noise"]
    )
    surr_grace_inv = inf.LinearBayesianInversion(
        surr_grace_fwd, surr_meas["unmasked_prior"]
    )
    P_grace = (1 - alpha) * surr_grace_inv.woodbury_data_preconditioner(
        woodbury_solver
    ) + alpha * exact_meas["grace_noise"].inverse_covariance

    # ------------------ 4. SOLVING POSTERIORS ------------------
    print("\nSolving for Posteriors...")
    callback = inf.ProgressCallback()
    tolerance = 0.01 * min(args.alt_noise_std_factor, args.grace_noise_std_factor)
    solver = inf.CGSolver(callback=callback, rtol=tolerance)

    print(" -> Solving Altimetry-only...")
    post_alt = alt_inv.model_posterior_measure(alt_data, solver, preconditioner=P_alt)

    print(" -> Solving GRACE-only...")
    post_grace = grace_inv.model_posterior_measure(
        grace_data, solver, preconditioner=P_grace
    )

    print(" -> Solving Joint Inversion...")
    post_joint = joint_inv.model_posterior_measure(
        joint_data, solver, preconditioner=P_joint
    )

    # ------------------ 5. GMSL ------------------
    if args.plot_pdfs:
        print("\nPlotting GMSL PDFs...")
        true_gmsl_op = (
            utils.true_gmsl_operator(
                exact_phys["state"],
                exact_phys["load_space"],
                exact_phys["continuous_sl"],
            )
            * scale_mm
        )

        alt_avg_op = sl.linear_operators.altimetry_averaging_operator(points) * scale_mm

        prior_gmsl_measure = (
            exact_meas["model_prior"]
            .affine_mapping(operator=true_gmsl_op)
            .with_dense_covariance()
        )

        alt_data_measure = exact_meas["alt_noise"].affine_mapping(translation=alt_data)
        alt_gmsl_measure = alt_data_measure.affine_mapping(
            operator=alt_avg_op
        ).with_dense_covariance()

        post_gmsl_alt = post_alt.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        post_gmsl_grace = post_grace.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        post_gmsl_joint = post_joint.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        true_gmsl_val_mm = true_gmsl_op(true_model)[0]

        # Log GMSL metrics
        prior_var = prior_gmsl_measure.covariance.matrix(dense=True)[0, 0]
        alt_var = post_gmsl_alt.covariance.matrix(dense=True)[0, 0]
        grace_var = post_gmsl_grace.covariance.matrix(dense=True)[0, 0]
        joint_var = post_gmsl_joint.covariance.matrix(dense=True)[0, 0]

        alt_red = 100.0 * (1.0 - (alt_var / prior_var))
        grace_red = 100.0 * (1.0 - (grace_var / prior_var))
        joint_red = 100.0 * (1.0 - (joint_var / prior_var))

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(
                f"{'Target':<12} | {'Estimator':<12} | {'KL Div':<10} | {'Prior Var':<12} | {'Post Var':<12} | {'Reduction'}\n"
            )
            f_metrics.write("-" * 80 + "\n")
            f_metrics.write(
                f"{'GMSL':<12} | {'Alt-Only':<12} | {post_gmsl_alt.kl_divergence(prior_gmsl_measure):<10.4f} | {prior_var:<12.4f} | {alt_var:<12.4f} | {alt_red:>6.2f}%\n"
            )
            f_metrics.write(
                f"{'GMSL':<12} | {'GRACE-Only':<12} | {post_gmsl_grace.kl_divergence(prior_gmsl_measure):<10.4f} | {prior_var:<12.4f} | {grace_var:<12.4f} | {grace_red:>6.2f}%\n"
            )
            f_metrics.write(
                f"{'GMSL':<12} | {'Joint':<12} | {post_gmsl_joint.kl_divergence(prior_gmsl_measure):<10.4f} | {prior_var:<12.4f} | {joint_var:<12.4f} | {joint_red:>6.2f}%\n"
            )

        measures = [alt_gmsl_measure, post_gmsl_alt, post_gmsl_grace, post_gmsl_joint]
        labels = ["Simple averaging", "Altimetry-Only", "GRACE-Only", "Joint"]

        fig_pdf, ax_pdf = plt.subplots(figsize=(10, 6), layout="constrained")
        inf.plot_1d_distributions(
            measures,
            true_value=true_gmsl_val_mm,
            ax=ax_pdf,
            title="",
            xlabel="Global Mean Sea Level Change (mm)",
            posterior_labels=labels,
        )
        figures_to_save["gmsl_pdf"] = fig_pdf

    # ---------- 5b. GMSL SPLIT: BARYSTATIC vs STERIC ----------
    if args.plot_gmsl_split:
        print("\nPushing forward onto the 2D GMSL split (barystatic, steric)...")
        bary_op, steric_gmsl_op, steric_direct_op, dyn_direct_op = (
            utils.gmsl_split_operators(
                exact_phys["state"],
                exact_phys["load_space"],
                exact_phys["continuous_sl"],
            )
        )
        split_op = inf.ColumnLinearOperator([bary_op, steric_gmsl_op]) * scale_mm
        final_split_op = split_op.codomain.coordinate_projection @ split_op

        true_split = final_split_op(true_model)

        prior_split = (
            exact_meas["model_prior"]
            .affine_mapping(operator=final_split_op)
            .with_dense_covariance(parallel=True, n_jobs=2)
        )
        alt_split = post_alt.affine_mapping(
            operator=final_split_op
        ).with_dense_covariance(parallel=True, n_jobs=2)
        grace_split = post_grace.affine_mapping(
            operator=final_split_op
        ).with_dense_covariance(parallel=True, n_jobs=2)
        joint_split = post_joint.affine_mapping(
            operator=final_split_op
        ).with_dense_covariance(parallel=True, n_jobs=2)

        prior_cov_mat = prior_split.covariance.matrix(dense=True)
        alt_cov_mat = alt_split.covariance.matrix(dense=True)
        grace_cov_mat = grace_split.covariance.matrix(dense=True)
        joint_cov_mat = joint_split.covariance.matrix(dense=True)

        def correlation(cov):
            return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        # Consistency checks. The residual steric GMSL (total minus
        # barystatic) should equal the direct ocean average of the steric
        # sea level: their difference diagnoses the SLE mass balance
        # (solver convergence), independently of the prior constraint.
        # Separately, under the prior mass constraint the ocean average of
        # the dynamic SSH equals the direct steric average, so their
        # difference (the ocean mean of the manometric part, <zeta>_O)
        # diagnoses the constraint.
        steric_resid_mm = steric_gmsl_op(true_model)[0] * scale_mm
        steric_direct_mm = steric_direct_op(true_model)[0] * scale_mm
        dyn_direct_mm = dyn_direct_op(true_model)[0] * scale_mm

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write("\n\nGMSL Split: Barystatic vs Steric\n")
            f_metrics.write("=" * 125 + "\n")
            f_metrics.write(
                f"{'Metric':<22} | {'Prior':<12} | {'Alt-Only':<12} | "
                f"{'GRACE-Only':<12} | {'Joint':<12}\n"
            )
            f_metrics.write("-" * 125 + "\n")
            f_metrics.write(
                f"{'KL Div vs prior':<22} | {'-':<12} | "
                f"{alt_split.kl_divergence(prior_split):<12.4f} | "
                f"{grace_split.kl_divergence(prior_split):<12.4f} | "
                f"{joint_split.kl_divergence(prior_split):<12.4f}\n"
            )
            f_metrics.write(
                f"{'Bary-Steric corr':<22} | {correlation(prior_cov_mat):<+12.4f} | "
                f"{correlation(alt_cov_mat):<+12.4f} | "
                f"{correlation(grace_cov_mat):<+12.4f} | "
                f"{correlation(joint_cov_mat):<+12.4f}\n"
            )
            f_metrics.write(
                f"SLE mass-balance check:  steric residual = {steric_resid_mm:.6f} mm | "
                f"steric direct = {steric_direct_mm:.6f} mm | "
                f"diff = {steric_resid_mm - steric_direct_mm:.3e} mm\n"
            )
            f_metrics.write(
                f"Constraint check:        <dyn SSH>_O = {dyn_direct_mm:.6f} mm | "
                f"<zeta>_O = {dyn_direct_mm - steric_direct_mm:.3e} mm "
                f"(should be ~0)\n"
            )
            f_metrics.write("-" * 125 + "\n")
            f_metrics.write(
                f"{'Component':<22} | {'Prior Var':<12} | {'Alt Var':<12} | "
                f"{'Alt Red%':<10} | {'GRACE Var':<12} | {'GRACE Red%':<10} | "
                f"{'Joint Var':<12} | {'Joint Red%'}\n"
            )
            f_metrics.write("-" * 125 + "\n")
            for i, name in enumerate(["Barystatic", "Steric"]):
                pr_v = prior_cov_mat[i, i]
                a_v = alt_cov_mat[i, i]
                g_v = grace_cov_mat[i, i]
                j_v = joint_cov_mat[i, i]
                a_red = 100.0 * (1.0 - (a_v / pr_v)) if pr_v > 0 else 0.0
                g_red = 100.0 * (1.0 - (g_v / pr_v)) if pr_v > 0 else 0.0
                j_red = 100.0 * (1.0 - (j_v / pr_v)) if pr_v > 0 else 0.0
                f_metrics.write(
                    f"{name:<22} | {pr_v:<12.4f} | {a_v:<12.4f} | {a_red:>9.2f}% | "
                    f"{g_v:<12.4f} | {g_red:>9.2f}% | "
                    f"{j_v:<12.4f} | {j_red:>9.2f}%\n"
                )
            f_metrics.write("-" * 125 + "\n")

        split_labels = ["Barystatic GMSL (mm)", "Steric GMSL (mm)"]

        inf.plot_corner_distributions(
            alt_split,
            prior_measure=prior_split,
            true_values=true_split,
            labels=split_labels,
            title="",
            fill_density=False,
        )
        figures_to_save["gmsl_split_corner_altimetry"] = plt.gcf()

        inf.plot_corner_distributions(
            grace_split,
            prior_measure=prior_split,
            true_values=true_split,
            labels=split_labels,
            title="",
            fill_density=False,
        )
        figures_to_save["gmsl_split_corner_grace"] = plt.gcf()

        inf.plot_corner_distributions(
            joint_split,
            prior_measure=prior_split,
            true_values=true_split,
            labels=split_labels,
            title="",
            fill_density=False,
        )
        figures_to_save["gmsl_split_corner_joint"] = plt.gcf()

    # ------------------ 6. MAPPING & COVARIANCE ------------------
    if args.plot_maps:
        print("\nGenerating 3-component spatial maps...")
        gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}
        cb_kwargs = {"orientation": "horizontal", "shrink": 0.8, "pad": 0.05}

        state = exact_phys["state"]
        load_space = exact_phys["load_space"]
        scale_mm = exact_phys["scale_mm"]

        # Physical density units for the raw density maps (nondimensional
        # density x density_scale = kg/m^3; x1000 -> g/m^3).
        rho_scale_gm3 = 1.0e3 * state.model.parameters.density_scale
        ocean_mask_gm3 = rho_scale_gm3 * ocean_mask

        plot_std = args.std_samples > 0

        posterior_cases = [
            ("joint", "Joint Posterior", post_joint),
            ("altimetry", "Altimetry-Only Posterior", post_alt),
            ("grace", "GRACE-Only Posterior", post_grace),
        ]
        expectations = {name: post.expectation for name, _, post in posterior_cases}

        # Shared per-row colour scales so the three posterior figures are
        # directly comparable.
        true_ice, true_dyn, true_rho = true_model
        vmax_ice = max(
            np.max(np.abs(true_ice.data * scale_mm)),
            max(np.max(np.abs(e[0].data * scale_mm)) for e in expectations.values()),
        )
        vmax_dyn = max(
            np.max(np.abs(true_dyn.data * scale_mm)),
            max(np.max(np.abs(e[1].data * scale_mm)) for e in expectations.values()),
        )
        vmax_rho = max(
            np.max(np.abs(true_rho.data * rho_scale_gm3)),
            max(
                np.max(np.abs(e[2].data * rho_scale_gm3)) for e in expectations.values()
            ),
        )
        vmaxes = (vmax_ice, vmax_dyn, vmax_rho)

        stds = {name: None for name, _, _ in posterior_cases}
        std_vmaxes = None
        if plot_std:
            for name, label, post in posterior_cases:
                print(
                    f"  Computing pointwise standard deviation from "
                    f"{args.std_samples} {label} samples..."
                )
                stds[name] = compute_pointwise_std(
                    load_space, post, expectations[name], args.std_samples, n_jobs=10
                )
            std_vmaxes = (
                max(np.max(sd[0].data * scale_mm) for sd in stds.values()),
                max(np.max(sd[1].data * scale_mm) for sd in stds.values()),
                max(np.max(sd[2].data * rho_scale_gm3) for sd in stds.values()),
            )

        for name, label, _ in posterior_cases:
            fig_maps = plot_state_maps(
                state,
                true_model,
                expectations[name],
                stds[name],
                vmaxes,
                std_vmaxes,
                ocean_mask_mm,
                ice_mask_mm,
                ocean_mask_gm3,
                scale_mm,
                label,
                regions=regions_to_analyze if args.plot_regions else None,
            )
            figures_to_save[f"posterior_maps_{name}"] = fig_maps

        # =================================================================
        # Point-wise Covariance Maps (Prior vs Alt-Only vs Joint)
        # =================================================================
        print("\nGenerating Point-wise Covariance Maps (Prior vs Alt-Only vs Joint)...")

        # Covariance maps show the raw model parameters, matching the
        # posterior maps: ice and dynamic SSH in mm, density in g/m^3.
        # Rows mixing a height component with the density component
        # therefore carry mixed units.
        comp_scales = [scale_mm, scale_mm, rho_scale_gm3]

        def cov_label(row_idx, pt_idx):
            mm_components = {0, 1}
            if row_idx in mm_components and pt_idx in mm_components:
                return "Covariance (mm\u00b2)"
            if row_idx not in mm_components and pt_idx not in mm_components:
                return r"Covariance ((g m$^{-3}$)$^{2}$)"
            return r"Covariance (mm $\cdot$ g m$^{-3}$)"

        row_titles = ["Ice thickness", "Dynamic SSH", "Density"]

        scenarios = [
            ("Ice", 0, (-78.0, -110.0), "WAIS"),
            ("Ocean Dyn", 1, (30.0, -45.0), "North_Atlantic"),
        ]

        def plot_cov_row(axes, fields, pt, label):
            """Helper to plot 3-way prior/alt-only/joint covariance side-by-side."""
            vmaxs = [np.max(np.abs(f.data)) for f in fields]

            # Fallback if one or more fields vanish completely
            for i in range(len(vmaxs)):
                if vmaxs[i] == 0:
                    vmaxs[i] = max(vmaxs) if max(vmaxs) > 0 else 1.0

            for ax, field, vmax in zip(axes, fields, vmaxs):
                sl.plot(
                    field,
                    ax=ax,
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-vmax,
                    vmax=vmax,
                    colorbar_kwargs={**cb_kwargs, "label": label},
                    gridlines_kwargs=gl_kwargs,
                )
                sl.plot_points(
                    [pt],
                    ax=ax,
                    color="black",
                    zorder=10,
                    gridlines=False,
                )

        for comp_name, comp_idx, pt, pt_name in scenarios:
            print(f"  Evaluating perturbation in {comp_name} at {pt}...")

            dirac_rep = load_space.dirac_representation(pt)
            test_vec = [load_space.zero, load_space.zero, load_space.zero]
            test_vec[comp_idx] = dirac_rep

            # Extract covariances for Prior, Alt-Only, and Joint
            prior_cov = exact_meas["model_prior"].covariance(test_vec)
            post_alt_cov = post_alt.covariance(test_vec)
            post_joint_cov = post_joint.covariance(test_vec)

            pr_ice, pr_dyn, pr_rho = prior_cov
            po_alt_ice, po_alt_dyn, po_alt_rho = post_alt_cov
            po_joint_ice, po_joint_dyn, po_joint_rho = post_joint_cov

            perturb_scale = comp_scales[comp_idx]

            # Apply scaling and spatial masks
            pr_ice_plot = pr_ice * (perturb_scale * scale_mm) * ice_mask
            po_alt_ice_plot = po_alt_ice * (perturb_scale * scale_mm) * ice_mask
            po_joint_ice_plot = po_joint_ice * (perturb_scale * scale_mm) * ice_mask

            pr_dyn_plot = pr_dyn * (perturb_scale * scale_mm) * ocean_mask
            po_alt_dyn_plot = po_alt_dyn * (perturb_scale * scale_mm) * ocean_mask
            po_joint_dyn_plot = po_joint_dyn * (perturb_scale * scale_mm) * ocean_mask

            pr_rho_plot = pr_rho * (perturb_scale * rho_scale_gm3) * ocean_mask
            po_alt_rho_plot = po_alt_rho * (perturb_scale * rho_scale_gm3) * ocean_mask
            po_joint_rho_plot = (
                po_joint_rho * (perturb_scale * rho_scale_gm3) * ocean_mask
            )

            # Setup 3x3 grid
            fig_cov, axes_cov = sl.subplots(
                3, 3, figsize=(24, 16), gridspec_kw={"hspace": 0.15, "wspace": 0.1}
            )

            # Plot each row
            plot_cov_row(
                axes_cov[0],
                [pr_ice_plot, po_alt_ice_plot, po_joint_ice_plot],
                pt,
                cov_label(0, comp_idx),
            )
            plot_cov_row(
                axes_cov[1],
                [pr_dyn_plot, po_alt_dyn_plot, po_joint_dyn_plot],
                pt,
                cov_label(1, comp_idx),
            )
            plot_cov_row(
                axes_cov[2],
                [pr_rho_plot, po_alt_rho_plot, po_joint_rho_plot],
                pt,
                cov_label(2, comp_idx),
            )

            # Apply Custom Titles to Grid Layout
            axes_cov[0, 0].set_title("Prior", fontsize=16, fontweight="bold", pad=20)
            axes_cov[0, 1].set_title(
                "Altimetry-Only Posterior", fontsize=16, fontweight="bold", pad=20
            )
            axes_cov[0, 2].set_title(
                "Joint Posterior", fontsize=16, fontweight="bold", pad=20
            )

            for i, row_title in enumerate(row_titles):
                axes_cov[i, 0].annotate(
                    row_title,
                    xy=(-0.1, 0.5),
                    xycoords="axes fraction",
                    fontsize=16,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                )

            if args.plot_regions:
                for ax in axes_cov.flatten():
                    state.plot_boundaries(ax, regions_to_analyze)

            figures_to_save[
                f"covariance_comparison_{comp_name.replace(' ', '_')}_{pt_name}"
            ] = fig_cov

    # ------------------ 7. REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")
        # Ocean dynamic SL includes the regional GRD response to the ocean
        # load; steric SL loads nothing; the three coordinates sum exactly
        # to the regional mean sea level change (see joint_utils).
        op_dyn, op_steric, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        combined_op = (
            inf.ColumnLinearOperator([op_dyn, op_steric, op_ice_fp]) * scale_mm
        )
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(true_model)
        prior_meas = (
            exact_meas["model_prior"]
            .affine_mapping(operator=final_op)
            .with_dense_covariance(parallel=True, n_jobs=3)
        )
        post_alt_meas = post_alt.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)
        post_grace_meas = post_grace.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)
        post_joint_meas = post_joint.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)

        labels = [
            "Ocean dynamic SL (mm)",
            "Steric SL (mm)",
            "Barystatic GRD SL (mm)",
        ]

        # Log Regional metrics
        prior_cov_mat = prior_meas.covariance.matrix(dense=True)
        alt_cov_mat = post_alt_meas.covariance.matrix(dense=True)
        grace_cov_mat = post_grace_meas.covariance.matrix(dense=True)
        joint_cov_mat = post_joint_meas.covariance.matrix(dense=True)

        pr_trace = np.trace(prior_cov_mat)
        alt_trace = np.trace(alt_cov_mat)
        grace_trace = np.trace(grace_cov_mat)
        joint_trace = np.trace(joint_cov_mat)

        alt_trace_red = 100.0 * (1.0 - (alt_trace / pr_trace)) if pr_trace > 0 else 0.0
        grace_trace_red = (
            100.0 * (1.0 - (grace_trace / pr_trace)) if pr_trace > 0 else 0.0
        )
        joint_trace_red = (
            100.0 * (1.0 - (joint_trace / pr_trace)) if pr_trace > 0 else 0.0
        )

        pr_det = np.linalg.det(prior_cov_mat)
        alt_det = np.linalg.det(alt_cov_mat)
        grace_det = np.linalg.det(grace_cov_mat)
        joint_det = np.linalg.det(joint_cov_mat)

        alt_det_red = 100.0 * (1.0 - (alt_det / pr_det)) if pr_det > 0 else 0.0
        grace_det_red = 100.0 * (1.0 - (grace_det / pr_det)) if pr_det > 0 else 0.0
        joint_det_red = 100.0 * (1.0 - (joint_det / pr_det)) if pr_det > 0 else 0.0

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(
                f"\n\nRegional Signal Separation ({regions_to_analyze[0]})\n"
            )
            f_metrics.write(
                "(Components sum to the regional mean sea level change; the\n"
                " dynamic component includes its GRD response.)\n"
            )
            f_metrics.write("=" * 135 + "\n")

            f_metrics.write(
                f"{'Metric':<22} | {'Prior':<12} | {'Alt-Only':<12} | {'Alt Red%':<10} | {'GRACE-Only':<12} | {'GRACE Red%':<10} | {'Joint':<12} | {'Joint Red%'}\n"
            )
            f_metrics.write("-" * 135 + "\n")
            f_metrics.write(
                f"{'Joint KL Div (nats)':<22} | {'-':<12} | {post_alt_meas.kl_divergence(prior_meas):<12.4f} | {'-':<10} | {post_grace_meas.kl_divergence(prior_meas):<12.4f} | {'-':<10} | {post_joint_meas.kl_divergence(prior_meas):<12.4f} | {'-'}\n"
            )
            f_metrics.write(
                f"{'Total Var (Trace) mm²':<22} | {pr_trace:<12.4f} | {alt_trace:<12.4f} | {alt_trace_red:>9.2f}% | {grace_trace:<12.4f} | {grace_trace_red:>9.2f}% | {joint_trace:<12.4f} | {joint_trace_red:>9.2f}%\n"
            )
            f_metrics.write(
                f"{'Generalized Var (Det)':<22} | {pr_det:<12.4e} | {alt_det:<12.4e} | {alt_det_red:>9.2f}% | {grace_det:<12.4e} | {grace_det_red:>9.2f}% | {joint_det:<12.4e} | {joint_det_red:>9.2f}%\n"
            )
            f_metrics.write("-" * 135 + "\n")

            f_metrics.write(
                f"{'Component':<22} | {'Prior Var':<12} | {'Alt Var':<12} | {'Alt Red%':<10} | {'GRACE Var':<12} | {'GRACE Red%':<10} | {'Joint Var':<12} | {'Joint Red%'}\n"
            )
            f_metrics.write("-" * 135 + "\n")

            comp_names = ["Ocean dynamic SL", "Steric SL", "Barystatic GRD"]
            for i, name in enumerate(comp_names):
                pr_v = prior_cov_mat[i, i]
                a_v = alt_cov_mat[i, i]
                g_v = grace_cov_mat[i, i]
                j_v = joint_cov_mat[i, i]

                a_red = 100.0 * (1.0 - (a_v / pr_v)) if pr_v > 0 else 0.0
                g_red = 100.0 * (1.0 - (g_v / pr_v)) if pr_v > 0 else 0.0
                j_red = 100.0 * (1.0 - (j_v / pr_v)) if pr_v > 0 else 0.0

                f_metrics.write(
                    f"{name:<22} | {pr_v:<12.4f} | {a_v:<12.4f} | {a_red:>9.2f}% | {g_v:<12.4f} | {g_red:>9.2f}% | {j_v:<12.4f} | {j_red:>9.2f}%\n"
                )
            f_metrics.write("-" * 135 + "\n")

        # Plots
        inf.plot_corner_distributions(
            post_alt_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner_altimetry"] = plt.gcf()

        inf.plot_corner_distributions(
            post_grace_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner_grace"] = plt.gcf()

        inf.plot_corner_distributions(
            post_joint_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner_joint"] = plt.gcf()

    # ------------------ SAVE ALL FIGURES ------------------
    if figures_to_save:
        print(f"\nSaving {len(figures_to_save)} plots to '{output_dir}/'...")
        for name, fig in figures_to_save.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filepath}")
            plt.close(fig)


if __name__ == "__main__":
    main()
