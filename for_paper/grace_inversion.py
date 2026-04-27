"""
Joint Bayesian Inversion (GRACE + Satellite Altimetry)
======================================================

This script performs a joint Bayesian inversion of synthetic GRACE gravimetry
and satellite altimetry data to simultaneously estimate ice sheet mass loss
and ocean dynamic topography.
"""

import argparse
import os
import numpy as np
import scipy.stats as stats
import matplotlib

# Force headless backend to avoid Wayland/Qt display errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl

from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator

import joint_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian Inversion of GRACE and Altimetry Data."
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
        help="Plot spatial maps of true loads, posterior expectations, and SSH.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot regional Corner Plot decomposing the signal.",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of Monte Carlo trials for statistical comparison.",
    )

    # --- Resolution & Physics Settings ---
    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Maximum spherical harmonic degree for the exact Earth model.",
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=32,
        help="Maximum spherical harmonic degree for the surrogate preconditioner.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Maximum spherical harmonic degree of the GRACE observations.",
    )
    parser.add_argument(
        "--load-order",
        type=float,
        default=2.0,
        help="Sobolev space order for the load.",
    )
    parser.add_argument(
        "--load-scale-km",
        type=float,
        default=500.0,
        help="Length scale (in km) defining the load space.",
    )

    # --- Preconditioner Options ---
    parser.add_argument(
        "--no-precond",
        action="store_true",
        help="Disable the surrogate sparse preconditioner entirely.",
    )
    parser.add_argument(
        "--full-woodbury",
        action="store_true",
        help="Use monolithic Joint Woodbury preconditioner instead of the Hybrid Block-Diagonal preconditioner.",
    )

    # --- Observation & Prior Settings ---
    parser.add_argument(
        "--spacing",
        type=float,
        default=4.0,
        help="Spacing in degrees for the altimetry observation points.",
    )
    parser.add_argument(
        "--ice-scale-km",
        type=float,
        default=500.0,
        help="Correlation length scale (in km) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ice-std-mm",
        type=float,
        default=10.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ocean-scale-km",
        type=float,
        default=100.0,
        help="Correlation length scale (in km) for the ocean dynamic prior.",
    )
    parser.add_argument(
        "--ocean-std-factor",
        type=float,
        default=2.0,
        help="Ocean dynamic standard deviation as a factor of GMSL std.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=1.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    # --- Noise Settings ---
    parser.add_argument(
        "--alt-noise-std-factor",
        type=float,
        default=1.0,
        help="Altimetry instrument noise std per point as a factor of GMSL std.",
    )
    parser.add_argument(
        "--grace-noise-scale-km",
        type=float,
        default=50.0,
        help="Correlation length scale (in km) for GRACE spatial noise.",
    )
    parser.add_argument(
        "--grace-noise-std-factor",
        type=float,
        default=0.1,
        help="GRACE spatial noise std as a factor of the ice prior std.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        if args.mc_trials == 0:
            args.mc_trials = 500

    # Setup directory to save plots
    output_dir = "output_plots_joint"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    inf.configure_threading(n_threads=1)

    # ==========================================
    # 1. BUILD EXACT PHYSICS
    # ==========================================
    print(f"\nBuilding EXACT joint physical operators (lmax={args.lmax})...")
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
        args.ice_scale_km,
        args.ice_std_mm,
        args.ocean_scale_km,
        args.ocean_std_factor,
        args.alt_noise_std_factor,
        args.grace_noise_scale_km,
        args.grace_noise_std_factor,
        args.obs_degree,
        points,
        exact_phys["scale_mm"],
        prior_shift=args.prior_shift,
    )

    scale_mm = exact_phys["scale_mm"]
    print(
        f"Implied GMSL prior standard deviation: {exact_meas['gmsl_std'] * scale_mm:.3f} mm"
    )

    print("Setting up Joint Bayesian Inversion...")
    forward_problem = inf.LinearForwardProblem(
        exact_phys["joint_forward"], data_error_measure=exact_meas["joint_noise"]
    )
    true_model, synthetic_data = forward_problem.synthetic_model_and_data(
        exact_meas["model_prior"]
    )
    inverse_problem = inf.LinearBayesianInversion(
        forward_problem, exact_meas["model_prior"]
    )

    # ==========================================
    # 2. PRECONDITIONER SETUP
    # ==========================================
    preconditioner = None
    if not args.no_precond:
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
            args.ice_scale_km,
            args.ice_std_mm,
            args.ocean_scale_km,
            args.ocean_std_factor,
            args.alt_noise_std_factor,
            args.grace_noise_scale_km,
            args.grace_noise_std_factor,
            args.obs_degree,
            points,
            surr_phys["scale_mm"],
            prior_shift=args.prior_shift,
            is_surrogate=True,
        )

        woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=8)
        alpha = 0.1

        if args.full_woodbury:
            print("Constructing Full Joint Woodbury Preconditioner...")
            woodbury_preconditioner = (
                inverse_problem.surrogate_woodbury_data_preconditioner(
                    woodbury_solver,
                    alternate_forward_operator=surr_phys["joint_forward"],
                    alternate_prior_measure=surr_meas["unmasked_prior"],
                    alternate_data_error_measure=exact_meas["joint_noise"],
                )
            )
            preconditioner = (1 - alpha) * woodbury_preconditioner + alpha * exact_meas[
                "joint_noise"
            ].inverse_covariance

        else:
            print("Constructing Hybrid Block-Diagonal Preconditioner...")

            # --- 1. Altimetry Block (Woodbury Surrogate) ---
            surr_alt_fwd = inf.LinearForwardProblem(
                surr_phys["alt_track"],
                data_error_measure=exact_meas["alt_noise"],
            )
            surr_alt_inv = inf.LinearBayesianInversion(
                surr_alt_fwd, surr_meas["unmasked_prior"]
            )
            woodbury_P_alt = surr_alt_inv.woodbury_data_preconditioner(woodbury_solver)
            P_alt = (1 - alpha) * woodbury_P_alt + alpha * exact_meas[
                "alt_noise"
            ].inverse_covariance

            # --- 2. GRACE Block (WMB Spectral) ---
            wmb_proxy_prior = exact_phys[
                "load_space"
            ].point_value_scaled_heat_kernel_gaussian_measure(
                exact_meas["ice_scale"], std=exact_meas["ice_std"]
            )
            P_grace = exact_meas["wmb"].bayesian_normal_operator_preconditioner(
                wmb_proxy_prior, exact_meas["grace_noise"]
            )

            # Fuse Blocks
            preconditioner = inf.BlockDiagonalLinearOperator([P_alt, P_grace])

    # ==========================================
    # 3. POSTERIOR SOLVE
    # ==========================================
    callback = inf.ProgressCallback()
    tolerance = 0.01 * min(args.alt_noise_std_factor, args.grace_noise_std_factor)
    solver = inf.CGSolver(callback=callback, rtol=tolerance)

    print("\nSolving for posterior expectation...")
    model_posterior = inverse_problem.model_posterior_measure(
        synthetic_data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    # ==========================================
    # 4. GMSL EXTRACTION & MAPPING
    # ==========================================
    true_gmsl_op = utils.true_gmsl_operator(
        exact_phys["state"], exact_phys["load_space"], exact_phys["continuous_ssh"]
    )

    alt_avg_op = altimetry_averaging_operator(points)
    grace_data_space = exact_meas["grace_noise"].domain

    # Extract GMSL strictly from the Altimetry subspace of the joint data
    joint_alt_avg_op = inf.RowLinearOperator(
        [alt_avg_op, grace_data_space.zero_operator(inf.EuclideanSpace(1))]
    )

    if args.plot_pdfs or args.mc_trials:
        post_gmsl_measure = model_posterior.affine_mapping(operator=true_gmsl_op)
        post_gmsl_std_mm = (
            np.sqrt(post_gmsl_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
        )

        std_noise_measure = exact_meas["joint_noise"].affine_mapping(
            operator=joint_alt_avg_op
        )
        std_noise_std_mm = (
            np.sqrt(std_noise_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
        )

        true_gmsl_val_mm = true_gmsl_op(true_model)[0] * scale_mm

    regions_to_analyze = ["Mediterranean Sea", "Tasman Sea"]

    # ------------------ OPTION 1: MAPS ------------------
    if args.plot_maps:
        print("Generating spatial maps...")
        cmap = "seismic"

        true_ice, true_ocean = true_model
        post_ice, post_ocean = model_posterior.expectation

        ocean_mask = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
        ice_mask = scale_mm * exact_phys["state"].ice_projection(value=0.0)

        vmax_ice = max(
            np.max(np.abs(true_ice.data * scale_mm)),
            np.max(np.abs(post_ice.data * scale_mm)),
        )
        vmax_ocean = max(
            np.max(np.abs(true_ocean.data * scale_mm)),
            np.max(np.abs(post_ocean.data * scale_mm)),
        )

        fig_maps, axes_maps = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        sl.plot(
            true_ice * ice_mask,
            ax=axes_maps[0, 0],
            colorbar=True,
            colorbar_kwargs={"label": "Ice Thickness Change (mm)"},
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
        )
        axes_maps[0, 0].set_title("True Ice Thickness Change")

        sl.plot(
            post_ice * ice_mask,
            ax=axes_maps[0, 1],
            colorbar=True,
            colorbar_kwargs={"label": "Ice Thickness Change (mm)"},
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
        )
        axes_maps[0, 1].set_title("Joint Posterior Expected Ice Thickness")

        sl.plot(
            true_ocean * ocean_mask,
            ax=axes_maps[1, 0],
            colorbar=True,
            colorbar_kwargs={"label": "Dynamic Ocean (mm)"},
            vmin=-vmax_ocean,
            vmax=vmax_ocean,
            cmap=cmap,
        )
        axes_maps[1, 0].set_title("True Dynamic Ocean Component")

        sl.plot(
            post_ocean * ocean_mask,
            ax=axes_maps[1, 1],
            colorbar=True,
            colorbar_kwargs={"label": "Dynamic Ocean (mm)"},
            vmin=-vmax_ocean,
            vmax=vmax_ocean,
            cmap=cmap,
        )
        axes_maps[1, 1].set_title("Joint Posterior Expected Dynamic Ocean")

        if args.plot_regions:
            for ax in axes_maps.flatten():
                exact_phys["state"].plot_boundaries(
                    ax, regions_to_analyze, edgecolor="black", linewidth=2.0, zorder=10
                )

        figures_to_save["joint_inversion_posterior_maps"] = fig_maps

        print("Generating Sea Surface Height maps with observation overlays...")
        true_ssh = exact_phys["continuous_ssh"](true_model)
        obs_data_mm = synthetic_data[0] * scale_mm
        vmax_ssh = max(
            np.max(np.abs(true_ssh.data * scale_mm)), np.max(np.abs(obs_data_mm))
        )

        fig_ssh, axes_ssh = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        sl.plot(
            true_ssh * scale_mm,
            ax=axes_ssh[0],
            colorbar=True,
            colorbar_kwargs={"label": "SSH Change (mm)"},
            vmin=-vmax_ssh,
            vmax=vmax_ssh,
            cmap=cmap,
        )
        axes_ssh[0].set_title("True Continuous SSH Change")

        axes_ssh[1].set_global()
        axes_ssh[1].coastlines(linewidth=0.5, alpha=0.5, zorder=10)
        sl.plot_points(
            points,
            data=obs_data_mm,
            ax=axes_ssh[1],
            cmap=cmap,
            vmin=-vmax_ssh,
            vmax=vmax_ssh,
            s=5,
            edgecolors="none",
            colorbar=True,
            colorbar_kwargs={
                "label": "Observed SSH (mm)",
                "orientation": "horizontal",
                "shrink": 0.7,
                "pad": 0.05,
            },
            zorder=5,
        )
        axes_ssh[1].set_title("Altimetry Observations")

        if args.plot_regions:
            for ax in axes_ssh:
                exact_phys["state"].plot_boundaries(
                    ax, regions_to_analyze, edgecolor="black", linewidth=2.0, zorder=10
                )

        figures_to_save["joint_inversion_ssh_maps"] = fig_ssh

    # ------------------ OPTION 2: PDF ------------------
    if args.plot_pdfs:
        print("Plotting Head-to-Head GMSL PDF...")

        class MockMeasure:
            def __init__(self, m, s):
                self.mean = np.array([m])
                self.cov = np.array([[s**2]])

        results = {
            "Joint Bayes (Alt + GRACE)": MockMeasure(
                post_gmsl_measure.expectation[0] * scale_mm, post_gmsl_std_mm
            ),
            "Standard Averaging (Alt Only)": MockMeasure(
                joint_alt_avg_op(synthetic_data)[0] * scale_mm, std_noise_std_mm
            ),
        }

        fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5), layout="constrained")
        inf.plot_1d_distributions(
            list(results.values()),
            true_value=true_gmsl_val_mm,
            ax=ax_pdf,
            xlabel="GMSL Change (mm)",
            title="Global Mean Sea Level Estimators",
            posterior_labels=list(results.keys()),
        )
        figures_to_save["joint_inversion_gmsl_pdf_comparison"] = fig_pdf

    # ------------------ OPTION 3: MONTE CARLO ------------------
    if args.mc_trials > 0:
        print(f"Running {args.mc_trials} MC trials via dense joint measure mapping...")
        post_exp_op = inverse_problem.posterior_expectation_operator(
            solver, preconditioner=preconditioner
        )

        if isinstance(post_exp_op, inf.AffineOperator):
            bayes_linear = post_exp_op.linear_part
            bayes_translation = true_gmsl_op(post_exp_op.translation_part)
        else:
            bayes_linear = post_exp_op
            bayes_translation = None

        std_err_op = inf.RowLinearOperator([-1.0 * true_gmsl_op, joint_alt_avg_op])
        bayes_err_op = inf.RowLinearOperator(
            [-1.0 * true_gmsl_op, true_gmsl_op @ bayes_linear]
        )
        joint_err_op = inf.ColumnLinearOperator([std_err_op, bayes_err_op])

        translation = (
            [true_gmsl_op.codomain.zero, bayes_translation]
            if bayes_translation is not None
            else None
        )
        joint_err_meas = inverse_problem.joint_prior_measure.affine_mapping(
            operator=joint_err_op, translation=translation
        )

        joint_err_dense = joint_err_meas.with_dense_covariance()
        samples = joint_err_dense.samples(args.mc_trials)

        std_errs, bayes_errs = np.zeros(args.mc_trials), np.zeros(args.mc_trials)
        for i, (s_err, b_err) in enumerate(samples):
            std_errs[i] = (s_err[0] * scale_mm) / std_noise_std_mm
            bayes_errs[i] = (b_err[0] * scale_mm) / post_gmsl_std_mm

        raw_cov = joint_err_dense.covariance.matrix(dense=True) * (scale_mm**2)
        raw_mean = joint_err_dense.expectation

        max_err = max(np.max(np.abs(std_errs)), np.max(np.abs(bayes_errs)))
        plot_limit = np.ceil(max_err) + 0.5

        fig_mc, ax_mc = plt.subplots(figsize=(7, 7), layout="constrained")
        ax_mc.scatter(
            std_errs,
            bayes_errs,
            alpha=0.6,
            color="purple",
            edgecolor="white",
            s=30,
            zorder=3,
        )

        mu_2d = np.array(
            [
                (raw_mean[0][0] * scale_mm) / std_noise_std_mm,
                (raw_mean[1][0] * scale_mm) / post_gmsl_std_mm,
            ]
        )
        cov_2d = np.array(
            [
                [
                    raw_cov[0, 0] / (std_noise_std_mm**2),
                    raw_cov[0, 1] / (std_noise_std_mm * post_gmsl_std_mm),
                ],
                [
                    raw_cov[0, 1] / (std_noise_std_mm * post_gmsl_std_mm),
                    raw_cov[1, 1] / (post_gmsl_std_mm**2),
                ],
            ]
        )

        x_grid, y_grid = np.mgrid[
            -plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j
        ]
        rv = stats.multivariate_normal(mu_2d, cov_2d)
        max_density = rv.pdf(mu_2d)
        ax_mc.contour(
            x_grid,
            y_grid,
            rv.pdf(np.dstack((x_grid, y_grid))),
            levels=[max_density * np.exp(-0.5 * k**2) for k in [4, 3, 2, 1]],
            colors="indigo",
            linewidths=[0.5, 1.0, 1.5],
            alpha=0.8,
            zorder=4,
        )

        ax_mc.axhline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
        ax_mc.axvline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
        ax_mc.axhspan(
            -1,
            1,
            color="blue",
            alpha=0.15,
            zorder=0,
            label=r"Joint Bayes 1$\sigma$ Expected",
        )
        ax_mc.axhspan(-2, 2, color="blue", alpha=0.05, zorder=0)
        ax_mc.axvspan(
            -1,
            1,
            color="red",
            alpha=0.15,
            zorder=0,
            label=r"Standard 1$\sigma$ Expected",
        )
        ax_mc.axvspan(-2, 2, color="red", alpha=0.05, zorder=0)

        ax_mc.set_xlim(-plot_limit, plot_limit)
        ax_mc.set_ylim(-plot_limit, plot_limit)
        ax_mc.set_aspect("equal")
        ax_mc.set_xlabel(r"Standard Estimator Normalized Error", fontsize=12)
        ax_mc.set_ylabel(r"Joint Bayesian Estimator Normalized Error", fontsize=12)
        ax_mc.set_title("GMSL MC Validation: Normalized Residuals", fontsize=16)

        ax_mc.plot([], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF")
        ax_mc.legend(loc="upper left", fontsize=10)
        figures_to_save["joint_inversion_mc_validation_scatter"] = fig_mc

    # ------------------ OPTION 4: REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals...")

        op_dynamic, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        combined_op = inf.ColumnLinearOperator([op_dynamic, op_ice_fp]) * scale_mm

        # Project block vector down to a native Euclidean Space for plotting
        flatten_op = combined_op.codomain.coordinate_projection
        final_op = flatten_op @ combined_op

        true_vals_mm = final_op(true_model)
        post_meas = model_posterior.affine_mapping(operator=final_op)
        prior_meas = exact_meas["model_prior"].affine_mapping(operator=final_op)

        labels = [
            f"{regions_to_analyze[0]}: Dynamic (mm)",
            f"{regions_to_analyze[1]}: Dynamic (mm)",
            f"{regions_to_analyze[0]}: Ice/SLE (mm)",
            f"{regions_to_analyze[1]}: Ice/SLE (mm)",
        ]

        inf.plot_corner_distributions(
            post_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="Joint Bayes Signal Separation: Dynamic Ocean vs. Ice Melt",
            fill_density=False,
        )
        figures_to_save["joint_inversion_regional_corner_plot"] = plt.gcf()

    # ------------------ SAVE ALL FIGURES ------------------
    if figures_to_save:
        print(f"\nSaving {len(figures_to_save)} plots to '{output_dir}/'...")
        for name, fig in figures_to_save.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filepath}")
            # Explicitly close the figure to free up memory
            plt.close(fig)


if __name__ == "__main__":
    main()
