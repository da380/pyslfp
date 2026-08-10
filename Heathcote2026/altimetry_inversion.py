"""
Extended Bayesian Altimetry Inversion (3-Component Model)
=========================================================

This script performs a Bayesian inversion of synthetic satellite altimetry data.
It estimates the underlying ice thickness changes, the total ocean dynamic
sea surface height change, and vertically averaged ocean density changes,
while strictly enforcing ocean mass conservation. Quantities of interest are
post-processed into the standard barystatic-GRD / steric / dynamic-manometric
split (see altimetry_utils). Features spatial covariance mapping to visualize the
physical constraints, an optional push-forward onto the 2D split of GMSL
change into barystatic and steric contributions, and an optionally
anti-correlated (Dyn, Rho) prior.
"""

import argparse
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


import pygeoinf as inf

import altimetry_utils as utils
import pyslfp as sl
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points

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

matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extended Bayesian inversion of Altimetry data (Ice, Dyn, Rho)."
    )
    # --- Output Options ---
    parser.add_argument(
        "--all", action="store_true", help="Enable all plotting options."
    )
    parser.add_argument("--plot-pdfs", action="store_true", help="Plot 1D GMSL PDFs.")
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot 3-component spatial maps (includes covariance maps).",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot 3-way regional signal decomposition.",
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
        help="Number samples for pointwise std estimates.",
    )

    # --- Resolution Settings ---
    parser.add_argument(
        "--lmax", type=int, default=256, help="Exact model max SH degree."
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Preconditioner max SH degree."
    )
    parser.add_argument(
        "--load-order", type=float, default=2.0, help="Sobolev space order."
    )
    parser.add_argument(
        "--load-scale-km", type=float, default=500.0, help="Sobolev length scale."
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Altimetry observation spacing."
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
        help="Effective steric SL std as a fraction of the Sterodynamic SL change std.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=1.0,
        help="Local (uncorrelated) altimetry noise std as factor of GMSL std.",
    )
    parser.add_argument(
        "--noise-corr-std-factor",
        type=float,
        default=0.05,
        help=(
            "Std of the optional large-scale correlated altimetry error "
            "component, as a factor of the GMSL prior std (0 disables). "
            "Represents long-wavelength systematics such as orbit and "
            "reference-frame errors; because it barely averages down, even "
            "small values (e.g. 0.02-0.05) set the error floor for "
            "large-scale averages."
        ),
    )
    parser.add_argument(
        "--noise-corr-scale-factor",
        type=float,
        default=4.0,
        help=(
            "Correlation scale of the correlated altimetry error "
            "component, as a factor of the load scale (default 4.0 = "
            "2000 km at the default load scale). Follows --prior-kernel."
        ),
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
        "--prior-kernel",
        choices=["heat", "sobolev"],
        default="sobolev",
        help=(
            "Covariance family for the three model priors and any "
            "spatially correlated noise models (applied consistently; "
            "white-noise settings unaffected): 'heat' or "
            "'sobolev' (default)."
        ),
    )
    parser.add_argument(
        "--prior-order",
        type=float,
        default=1.0,
        help=(
            "Common spectral order for --prior-kernel sobolev (must be "
            "positive). Sample roughness combines this order with the "
            "model space order (--load-order): with the default order-2 "
            "spaces, 1.0 gives degree-variance tails ~ l^-5, comparable "
            "to observed mesoscale SSH spectra. Ignored for the heat "
            "kernel."
        ),
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        args.plot_gmsl_split = True

    output_dir = "output_plots_altimetry_inversion"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    metrics_file = os.path.join(output_dir, "altimetry_metrics.txt")
    with open(metrics_file, "w") as f_metrics:
        f_metrics.write("Altimetry Inversion Metrics\n")
        f_metrics.write("===========================\n\n")

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")
    inf.configure_threading(n_threads=1)

    regions_to_analyze = ["Tasman Sea"]

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT 3-Component physical operators (lmax={args.lmax})...")
    state, load_space, fp_op, continuous_ssh, continuous_sl, forward_op, mm_scale = (
        utils.build_physics_components(
            args.lmax, args.load_order, args.load_scale_km, points, is_surrogate=False
        )
    )

    ocean_mask_mm = mm_scale * state.ocean_projection(value=0.0)
    ice_mask_mm = mm_scale * state.ice_projection(value=0.0)

    # Unitless masks for plotting
    ocean_mask = state.ocean_projection(value=0.0)
    ice_mask = state.ice_projection(value=0.0)

    model_prior, noise_measure, GMSL_prior_std = utils.build_measures(
        state,
        load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.noise_corr_scale_factor,
        args.noise_std_factor,
        points,
        mm_scale,
        prior_shift=args.prior_shift,
        is_surrogate=False,
        ocean_corr=args.ocean_corr,
        ocean_corr_scale_factor=args.ocean_corr_scale_factor,
        prior_kernel=args.prior_kernel,
        prior_order=args.prior_order,
        noise_corr_std_factor=args.noise_corr_std_factor,
    )

    print(f"Implied GMSL prior standard deviation: {GMSL_prior_std * mm_scale:.3f} mm")
    if args.ocean_corr > 0.0:
        print(
            f"Correlated (Dyn, Rho) prior enabled: r0 = {args.ocean_corr:.2f}, "
            f"correlation scale factor = {args.ocean_corr_scale_factor:.2f}"
        )
    if args.prior_kernel == "sobolev":
        print(
            f"Sobolev-kernel (Matern) priors enabled: common order = "
            f"{args.prior_order:.2f}"
        )
    if args.noise_corr_std_factor > 0.0:
        print(
            f"Correlated altimetry error component enabled: std factor = "
            f"{args.noise_corr_std_factor:.3f}, scale factor = "
            f"{args.noise_corr_scale_factor:.1f}"
        )

    print("Setting up Bayesian Inversion...")
    forward_problem = inf.LinearForwardProblem(
        forward_op, data_error_measure=noise_measure
    )
    model, data = forward_problem.synthetic_model_and_data(model_prior)
    data_measure = noise_measure.affine_mapping(translation=data)
    inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

    # ------------------ 2. PRECONDITIONER SETUP ------------------
    print(
        f"\nBuilding SURROGATE operators (lmax={args.surrogate_degree}) for preconditioning..."
    )
    surr_state, surr_load_space, _, _, _, surr_forward_op, _ = (
        utils.build_physics_components(
            args.surrogate_degree,
            args.load_order,
            args.load_scale_km,
            points,
            is_surrogate=True,
        )
    )

    surr_model_prior, surr_noise_measure, _ = utils.build_measures(
        surr_state,
        surr_load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.noise_corr_scale_factor,
        args.noise_std_factor,
        points,
        mm_scale,
        prior_shift=args.prior_shift,
        is_surrogate=True,
        ocean_corr=args.ocean_corr,
        ocean_corr_scale_factor=args.ocean_corr_scale_factor,
        prior_kernel=args.prior_kernel,
        prior_order=args.prior_order,
        noise_corr_std_factor=args.noise_corr_std_factor,
    )

    print("Constructing Woodbury preconditioner from unconstrained surrogate model...")
    woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=8)
    woodbury_preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
        woodbury_solver,
        alternate_forward_operator=surr_forward_op,
        alternate_prior_measure=surr_model_prior,
        alternate_data_error_measure=surr_noise_measure,
    )

    alpha = 0.1
    preconditioner = (
        1 - alpha
    ) * woodbury_preconditioner + alpha * surr_noise_measure.inverse_covariance

    # ------------------ 3. POSTERIOR SOLVE ------------------
    callback = inf.ProgressCallback()
    solver = inf.CGSolver(callback=callback, rtol=0.01 * args.noise_std_factor)

    print("\nSolving for 3-component posterior expectation...")
    model_posterior = inverse_problem.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    # ------------------ 4. GMSL & METRICS ------------------
    true_gmsl_op = utils.true_gmsl_operator(state, load_space, continuous_sl) * mm_scale
    alt_avg_op = sl.linear_operators.altimetry_averaging_operator(points) * mm_scale
    joint_space = true_gmsl_op.domain

    if args.plot_pdfs:
        print("Extracting GMSL PDFs...")
        prior_gmsl_measure = model_prior.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()
        alt_gmsl_measure = data_measure.affine_mapping(
            operator=alt_avg_op
        ).with_dense_covariance()
        post_gmsl_measure = model_posterior.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        true_gmsl = true_gmsl_op(model)[0]
        kl_div = post_gmsl_measure.kl_divergence(prior_gmsl_measure)

        post_var = post_gmsl_measure.covariance.matrix(dense=True)[0, 0]
        prior_var = prior_gmsl_measure.covariance.matrix(dense=True)[0, 0]
        var_reduction = 100.0 * (1.0 - (post_var / prior_var))

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(
                f"{'Target':<16} | {'KL Divergence':<15} | {'Prior Var (mm²)':<17} | {'Post Var (mm²)':<16} | {'Variance Reduction':<18}\n"
            )
            f_metrics.write("-" * 92 + "\n")
            f_metrics.write(
                f"{'GMSL':<16} | {kl_div:<15.4f} | {prior_var:<17.4f} | {post_var:<16.4f} | {var_reduction:>16.4f}%\n"
            )

        fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5), layout="constrained")
        inf.plot_1d_distributions(
            [post_gmsl_measure, alt_gmsl_measure],
            true_value=true_gmsl,
            ax=ax_pdf,
            title="",
            xlabel="Global Mean Sea Level Change (mm)",
            posterior_labels=["Bayesian", "Simple averaging"],
        )
        figures_to_save["gmsl_pdf"] = fig_pdf

    # ---------- 4b. GMSL SPLIT: BARYSTATIC vs STERIC ----------
    if args.plot_gmsl_split:
        print("\nPushing forward onto the 2D GMSL split (barystatic, steric)...")
        bary_op, steric_gmsl_op, steric_direct_op, dyn_direct_op = (
            utils.gmsl_split_operators(state, load_space, continuous_sl)
        )
        split_op = inf.ColumnLinearOperator([bary_op, steric_gmsl_op]) * mm_scale
        final_split_op = split_op.codomain.coordinate_projection @ split_op

        true_split = final_split_op(model)

        prior_split = model_prior.affine_mapping(
            operator=final_split_op
        ).with_dense_covariance(parallel=True, n_jobs=2)
        post_split = model_posterior.affine_mapping(
            operator=final_split_op
        ).with_dense_covariance(parallel=True, n_jobs=2)

        kl_div = post_split.kl_divergence(prior_split)

        prior_cov_mat = prior_split.covariance.matrix(dense=True)
        post_cov_mat = post_split.covariance.matrix(dense=True)

        def correlation(cov):
            return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        # Consistency checks. The residual steric GMSL (total minus
        # barystatic) should equal the direct ocean average of the steric
        # sea level: their difference diagnoses the SLE mass balance
        # (solver convergence), independently of the prior constraint.
        # Separately, under the prior mass constraint the ocean average of
        # the Sterodynamic SL Change equals the direct steric average, so their
        # difference (the ocean mean of the manometric part, <zeta>_O)
        # diagnoses the constraint.
        steric_resid_mm = steric_gmsl_op(model)[0] * mm_scale
        steric_direct_mm = steric_direct_op(model)[0] * mm_scale
        dyn_direct_mm = dyn_direct_op(model)[0] * mm_scale

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write("\nGMSL Split: Barystatic vs Steric\n")
            f_metrics.write("=" * 70 + "\n")
            f_metrics.write(f"Joint KL Divergence:     {kl_div:.4f} nats\n")
            f_metrics.write(
                f"Bary-Steric correlation: prior {correlation(prior_cov_mat):+.4f} | "
                f"posterior {correlation(post_cov_mat):+.4f}\n"
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
            f_metrics.write("-" * 70 + "\n")
            f_metrics.write(
                f"{'Component':<18} | {'Prior Var':<12} | {'Post Var':<12} | {'Reduction'}\n"
            )
            f_metrics.write("-" * 70 + "\n")
            for i, name in enumerate(["Barystatic", "Steric"]):
                pr_v = prior_cov_mat[i, i]
                po_v = post_cov_mat[i, i]
                red = 100.0 * (1.0 - (po_v / pr_v)) if pr_v > 0 else 0.0
                f_metrics.write(
                    f"{name:<18} | {pr_v:<12.4f} | {po_v:<12.4f} | {red:>8.2f}%\n"
                )
            f_metrics.write("-" * 70 + "\n")

        axes = inf.plot_corner_distributions(
            post_split,
            prior_measure=prior_split,
            true_values=true_split,
            labels=["Barystatic GMSL (mm)", "Steric GMSL (mm)"],
            title="",
            fill_density=False,
        )
        fig = plt.gcf()
        figures_to_save["gmsl_split_corner"] = fig

    # ------------------ 5. SPATIAL & COVARIANCE MAPS ------------------
    if args.plot_maps:
        print("\nGenerating 3-component spatial maps...")
        cmap = "seismic"
        gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}
        cb_kwargs = {"orientation": "horizontal", "shrink": 0.8, "pad": 0.05}

        # Physical density units for the raw density maps (nondimensional
        # density x density_scale = kg/m^3; x1000 -> g/m^3).
        rho_scale_gm3 = 1.0e3 * state.model.parameters.density_scale
        ocean_mask_gm3 = rho_scale_gm3 * ocean_mask

        # =================================================================
        # A. Posterior Spatial Maps & STD
        # =================================================================
        true_ice, true_dyn, true_rho = model
        post_ice, post_dyn, post_rho = model_posterior.expectation

        vmax_ice = max(
            np.max(np.abs(true_ice.data * mm_scale)),
            np.max(np.abs(post_ice.data * mm_scale)),
        )
        vmax_dyn = max(
            np.max(np.abs(true_dyn.data * mm_scale)),
            np.max(np.abs(post_dyn.data * mm_scale)),
        )
        vmax_rho = max(
            np.max(np.abs(true_rho.data * rho_scale_gm3)),
            np.max(np.abs(post_rho.data * rho_scale_gm3)),
        )

        plot_std = args.std_samples > 0
        ncols = 3 if plot_std else 2

        if plot_std:
            print(
                f"\nComputing pointwise standard deviation from {args.std_samples} posterior samples..."
            )
            samples = model_posterior.samples(
                args.std_samples, parallel=True, n_jobs=14
            )
            var_ice, var_dyn, var_rho = (
                load_space.zero,
                load_space.zero,
                load_space.zero,
            )

            for s_ice, s_dyn, s_rho in samples:
                diff_ice = load_space.subtract(s_ice, post_ice)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_ice, diff_ice),
                    var_ice,
                )

                diff_dyn = load_space.subtract(s_dyn, post_dyn)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_dyn, diff_dyn),
                    var_dyn,
                )

                diff_rho = load_space.subtract(s_rho, post_rho)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_rho, diff_rho),
                    var_rho,
                )

            std_ice = load_space.vector_sqrt(var_ice)
            std_dyn = load_space.vector_sqrt(var_dyn)
            std_rho = load_space.vector_sqrt(var_rho)
            cmap_std = "Blues"

        fig_width = 22 if plot_std else 16
        fig_maps, axes = sl.subplots(
            3, ncols, figsize=(fig_width, 16), gridspec_kw={"hspace": 0.15}
        )

        # --- ROW 1: ICE ---
        _, im1 = sl.plot(
            true_ice * mm_scale,
            ax=axes[0, 0],
            colorbar=True,
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Ice Thickness Change (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        _, im2 = sl.plot(
            post_ice * mm_scale,
            ax=axes[0, 1],
            colorbar=True,
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Ice Thickness Change (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            _, im3 = sl.plot(
                std_ice * ice_mask_mm,
                ax=axes[0, 2],
                colorbar=True,
                cmap=cmap_std,
                colorbar_kwargs={**cb_kwargs, "label": "Ice thickness change STD (mm)"},
                gridlines_kwargs=gl_kwargs,
            )

        # --- ROW 2: DYNAMIC TOPOGRAPHY ---
        sl.plot(
            true_dyn * ocean_mask_mm,
            ax=axes[1, 0],
            colorbar=True,
            vmin=-vmax_dyn,
            vmax=vmax_dyn,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Sterodynamic SL Change (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        sl.plot(
            post_dyn * ocean_mask_mm,
            ax=axes[1, 1],
            colorbar=True,
            vmin=-vmax_dyn,
            vmax=vmax_dyn,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Sterodynamic SL Change (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            sl.plot(
                std_dyn * ocean_mask_mm,
                ax=axes[1, 2],
                colorbar=True,
                cmap=cmap_std,
                colorbar_kwargs={
                    **cb_kwargs,
                    "label": "Sterodynamic SL Change STD (mm)",
                },
                gridlines_kwargs=gl_kwargs,
            )

        # --- ROW 3: VERTICALLY AVERAGED DENSITY ---
        sl.plot(
            true_rho * ocean_mask_gm3,
            ax=axes[2, 0],
            colorbar=True,
            vmin=-vmax_rho,
            vmax=vmax_rho,
            cmap=cmap,
            colorbar_kwargs={
                **cb_kwargs,
                "label": r"Averaged Density Change (g m$^{-3}$)",
            },
            gridlines_kwargs=gl_kwargs,
        )
        sl.plot(
            post_rho * ocean_mask_gm3,
            ax=axes[2, 1],
            colorbar=True,
            vmin=-vmax_rho,
            vmax=vmax_rho,
            cmap=cmap,
            colorbar_kwargs={
                **cb_kwargs,
                "label": r"Averaged Density Change (g m$^{-3}$)",
            },
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            sl.plot(
                std_rho * ocean_mask_gm3,
                ax=axes[2, 2],
                colorbar=True,
                cmap=cmap_std,
                colorbar_kwargs={
                    **cb_kwargs,
                    "label": r"Averaged Density STD (g m$^{-3}$)",
                },
                gridlines_kwargs=gl_kwargs,
            )

        if args.plot_regions:
            for ax in axes.flatten():
                state.plot_boundaries(ax, regions_to_analyze)

        col_labels = ["True State", "Posterior Expectation", "Pointwise Std. Deviation"]
        for j in range(ncols):
            axes[0, j].set_title(col_labels[j], fontsize=16, fontweight="bold", pad=20)

        figures_to_save["posterior_maps"] = fig_maps

        # =================================================================
        # B. SSH Observation Map
        # =================================================================
        print("Generating Sea Surface Height maps with observation overlays...")
        true_ssh = continuous_ssh(model)
        data_mm = data * mm_scale
        vmax_ssh = max(
            np.max(np.abs(true_ssh.data * mm_scale)), np.max(np.abs(data_mm))
        )

        fig_ssh, axes_ssh = sl.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.1}
        )

        sl.plot(
            true_ssh * ocean_mask_mm,
            ax=axes_ssh[0],
            colorbar=True,
            vmin=-vmax_ssh,
            vmax=vmax_ssh,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "SSH Change (mm)"},
            gridlines_kwargs=gl_kwargs,
        )

        axes_ssh[1].set_global()
        axes_ssh[1].coastlines(linewidth=0.5, alpha=0.5, zorder=10)
        sl.plot_points(
            points,
            data=data_mm,
            ax=axes_ssh[1],
            cmap=cmap,
            vmin=-vmax_ssh,
            vmax=vmax_ssh,
            s=6,
            edgecolors="none",
            colorbar=True,
            colorbar_kwargs={**cb_kwargs, "label": "Observed SSH Data (mm)"},
            zorder=5,
        )

        if args.plot_regions:
            for ax in axes_ssh.flatten():
                state.plot_boundaries(ax, regions_to_analyze)

        figures_to_save["observed_ssh"] = fig_ssh

        # =================================================================
        # C. Point-wise Covariance Maps
        # =================================================================
        print("\nGenerating Point-wise Covariance Maps...")

        # Covariance maps show the raw model parameters, matching the
        # posterior maps: ice and Sterodynamic SL Change in mm, density in g/m^3.
        # Rows mixing a height component with the density component
        # therefore carry mixed units.
        comp_scales = [mm_scale, mm_scale, rho_scale_gm3]

        def cov_label(row_idx, pt_idx):
            mm_components = {0, 1}
            if row_idx in mm_components and pt_idx in mm_components:
                return "Covariance (mm\u00b2)"
            if row_idx not in mm_components and pt_idx not in mm_components:
                return r"Covariance ((g m$^{-3}$)$^{2}$)"
            return r"Covariance (mm $\cdot$ g m$^{-3}$)"

        scenarios = [
            ("Ice", 0, (-78.0, -110.0), "WAIS"),
            ("Ocean Dyn", 1, (30.0, -45.0), "North_Atlantic"),
        ]

        def plot_cov_row(ax_pr, ax_po, pr_field, po_field, pt, label):
            """Helper to plot side-by-side prior/posterior covariance."""
            vmax_pr = np.max(np.abs(pr_field.data))
            vmax_po = np.max(np.abs(po_field.data))

            # Fallback if one or both fields vanish completely
            if vmax_pr == 0 and vmax_po == 0:
                vmax_pr = vmax_po = 1.0
            elif vmax_pr == 0:
                vmax_pr = vmax_po
            elif vmax_po == 0:
                vmax_po = vmax_pr

            _, im_pr = sl.plot(
                pr_field,
                ax=ax_pr,
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-vmax_pr,
                vmax=vmax_pr,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )
            sl.plot_points(
                [pt],
                ax=ax_pr,
                color="black",
                zorder=10,
                gridlines=False,
            )

            _, im_po = sl.plot(
                po_field,
                ax=ax_po,
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-vmax_po,
                vmax=vmax_po,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )
            sl.plot_points(
                [pt],
                ax=ax_po,
                color="black",
                zorder=10,
                gridlines=False,
            )

        for comp_name, comp_idx, pt, pt_name in scenarios:
            print(f"  Evaluating perturbation in {comp_name} at {pt}...")

            dirac_rep = load_space.dirac_representation(pt)
            test_vec = [load_space.zero, load_space.zero, load_space.zero]
            test_vec[comp_idx] = dirac_rep

            prior_cov = model_prior.covariance(test_vec)
            post_cov = model_posterior.covariance(test_vec)

            pr_ice, pr_dyn, pr_rho = prior_cov
            po_ice, po_dyn, po_rho = post_cov

            perturb_scale = comp_scales[comp_idx]

            pr_ice_plot = pr_ice * (perturb_scale * mm_scale) * ice_mask
            po_ice_plot = po_ice * (perturb_scale * mm_scale) * ice_mask

            pr_dyn_plot = pr_dyn * (perturb_scale * mm_scale) * ocean_mask
            po_dyn_plot = po_dyn * (perturb_scale * mm_scale) * ocean_mask

            pr_rho_plot = pr_rho * (perturb_scale * rho_scale_gm3) * ocean_mask
            po_rho_plot = po_rho * (perturb_scale * rho_scale_gm3) * ocean_mask

            fig_cov, axes_cov = sl.subplots(
                3, 2, figsize=(16, 16), gridspec_kw={"hspace": 0.15}
            )

            plot_cov_row(
                axes_cov[0, 0],
                axes_cov[0, 1],
                pr_ice_plot,
                po_ice_plot,
                pt,
                cov_label(0, comp_idx),
            )

            plot_cov_row(
                axes_cov[1, 0],
                axes_cov[1, 1],
                pr_dyn_plot,
                po_dyn_plot,
                pt,
                cov_label(1, comp_idx),
            )

            plot_cov_row(
                axes_cov[2, 0],
                axes_cov[2, 1],
                pr_rho_plot,
                po_rho_plot,
                pt,
                cov_label(2, comp_idx),
            )

            # --- Apply Custom Titles to Grid Layout ---
            axes_cov[0, 0].set_title("Prior", fontsize=16, fontweight="bold", pad=20)
            axes_cov[0, 1].set_title(
                "Posterior", fontsize=16, fontweight="bold", pad=20
            )

            row_titles = [
                "Ice thickness Change",
                "Sterodynamic SL Change",
                "Averaged Density Change",
            ]
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

            figures_to_save[f"covariance_{comp_name.replace(' ', '_')}_{pt_name}"] = (
                fig_cov
            )

    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")
        masks = [state.get_projection(r, value=0.0) for r in regions_to_analyze]
        avg_op = sl.linear_operators.averaging_operator(state, load_space, masks)

        # Dynamic manometric SL = the manometric part (zeta = eta - eta_s) PLUS
        # the regional GRD response to the total ocean load (which is exactly
        # the load of zeta, the steric part being column-mass neutral).
        # Steric SL loads nothing and so has no induced response. Together
        # with the barystatic-GRD term, the three coordinates sum exactly to
        # the regional average of the true sea level change.
        steric_op_field = utils.steric_sea_level_operator(
            state, load_space
        ) @ joint_space.subspace_projection(2)
        op_steric = avg_op @ steric_op_field

        barystatic_sl_op = fp_op.codomain.subspace_projection(0) @ fp_op

        dyn_to_load = sl.linear_operators.sea_level_change_to_load_operator(
            state, load_space, load_space
        )
        rho_to_load = sl.linear_operators.ocean_density_change_to_load_operator(
            state, load_space, load_space
        )
        ocean_load_op = dyn_to_load @ joint_space.subspace_projection(
            1
        ) + rho_to_load @ joint_space.subspace_projection(2)

        zeta_field_op = joint_space.subspace_projection(1) - steric_op_field
        op_dyn = avg_op @ (zeta_field_op + barystatic_sl_op @ ocean_load_op)

        ice_to_load = sl.linear_operators.ice_thickness_change_to_load_operator(
            state, load_space, load_space
        ) @ sl.linear_operators.ice_projection_operator(state, load_space)
        op_ice_fp = (
            avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
        )

        combined_op = (
            inf.ColumnLinearOperator([op_dyn, op_steric, op_ice_fp]) * mm_scale
        )
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals = final_op(model)
        post_meas = model_posterior.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)
        prior_meas = model_prior.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)

        kl_div = post_meas.kl_divergence(prior_meas)

        # --- Extract Dense Covariance Matrices ---
        prior_cov_mat = prior_meas.covariance.matrix(dense=True)
        post_cov_mat = post_meas.covariance.matrix(dense=True)

        prior_trace = np.trace(prior_cov_mat)
        post_trace = np.trace(post_cov_mat)
        trace_reduction = 100.0 * (1.0 - (post_trace / prior_trace))

        prior_det = np.linalg.det(prior_cov_mat)
        post_det = np.linalg.det(post_cov_mat)

        # Log KL divergence and detailed covariance metrics to file
        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(f"\nRegional Signal Separation ({regions_to_analyze[0]})\n")
            f_metrics.write(
                "(Components sum to the regional mean sea level change; the\n"
                " dynamic component includes its GRD response.)\n"
            )
            f_metrics.write("=" * 65 + "\n")
            f_metrics.write(f"Joint KL Divergence:      {kl_div:.4f} nats\n")
            f_metrics.write(f"Prior Total Var (Trace):  {prior_trace:.4f} mm²\n")
            f_metrics.write(f"Post Total Var (Trace):   {post_trace:.4f} mm²\n")
            f_metrics.write(f"Overall Trace Reduction:  {trace_reduction:.2f}%\n")
            f_metrics.write(f"Prior Generalized Var:    {prior_det:.4e}\n")
            f_metrics.write(f"Post Generalized Var:     {post_det:.4e}\n")
            f_metrics.write("-" * 65 + "\n")
            f_metrics.write(
                f"{'Component':<18} | {'Prior Var':<12} | {'Post Var':<12} | {'Reduction'}\n"
            )
            f_metrics.write("-" * 65 + "\n")

            comp_names = ["Dynamic Manometric SL", "Steric SL", "Barystatic-GRD SL"]
            for i, name in enumerate(comp_names):
                pr_v = prior_cov_mat[i, i]
                po_v = post_cov_mat[i, i]
                red = 100.0 * (1.0 - (po_v / pr_v)) if pr_v > 0 else 0.0
                f_metrics.write(
                    f"{name:<18} | {pr_v:<12.4f} | {po_v:<12.4f} | {red:>8.2f}%\n"
                )
            f_metrics.write("-" * 65 + "\n")

        labels = [
            "Dynamic Manometric SL (mm)",
            "Steric SL (mm)",
            "Barystatic-GRD SL (mm)",
        ]

        inf.plot_corner_distributions(
            post_meas,
            prior_measure=prior_meas,
            true_values=true_vals,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner"] = plt.gcf()

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
