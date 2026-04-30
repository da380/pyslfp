"""
Extended Joint Bayesian Inversion (GRACE + Satellite Altimetry)
===============================================================

This script performs a joint Bayesian inversion of synthetic GRACE gravimetry
and satellite altimetry data to estimate ice sheet mass loss, ocean dynamic
topography, and ocean density changes (steric expansion).

Use the `--compare` flag to run Head-to-Head Altimetry-only, GRACE-only,
and Joint inversions on the exact same physical scenario.
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
from pyslfp.linear_operators import ocean_altimetry_points

import joint_extended_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extended Joint Bayesian Inversion (Ice, Dyn, Rho) with Optional Comparison Mode."
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
        help="Plot spatial maps of true loads and posterior expectations.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot regional 3-way signal decomposition.",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of MC trials for error validation.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run Alt-only and GRACE-only alongside the Joint inversion.",
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
        default=0.5,
        help="Effective steric SL std as factor of GMSL std.",
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
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )
    parser.add_argument(
        "--no-precond", action="store_true", help="Disable the preconditioner entirely."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        if args.mc_trials == 0 and not args.compare:
            args.mc_trials = 500

    output_dir = "output_plots_joint_extended"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

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
    )

    scale_mm = exact_phys["scale_mm"]
    density_scale = exact_phys["state"].model.parameters.density_scale
    print(
        f"Implied GMSL prior standard deviation: {exact_meas['gmsl_std'] * scale_mm:.3f} mm"
    )

    print("\nDrawing MASTER synthetic model and dataset...")
    master_forward = inf.LinearForwardProblem(
        exact_phys["joint_forward"], data_error_measure=exact_meas["joint_noise"]
    )
    true_model, joint_data = master_forward.synthetic_model_and_data(
        exact_meas["model_prior"]
    )
    alt_data, grace_data = joint_data[0], joint_data[1]

    # ------------------ 2. SETUP INVERSE PROBLEMS ------------------
    joint_inv = inf.LinearBayesianInversion(master_forward, exact_meas["model_prior"])

    if args.compare:
        alt_fwd = inf.LinearForwardProblem(
            exact_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
        )
        alt_inv = inf.LinearBayesianInversion(alt_fwd, exact_meas["model_prior"])

        grace_fwd = inf.LinearForwardProblem(
            exact_phys["grace_track"], data_error_measure=exact_meas["grace_noise"]
        )
        grace_inv = inf.LinearBayesianInversion(grace_fwd, exact_meas["model_prior"])

    # ------------------ 3. PRECONDITIONER SETUP ------------------
    P_alt, P_grace, P_joint = None, None, None
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
        )

        woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=8)
        alpha = 0.1

        P_joint = (1 - alpha) * joint_inv.surrogate_woodbury_data_preconditioner(
            woodbury_solver,
            alternate_forward_operator=surr_phys["joint_forward"],
            alternate_prior_measure=surr_meas["unmasked_prior"],
            alternate_data_error_measure=exact_meas["joint_noise"],
        ) + alpha * exact_meas["joint_noise"].inverse_covariance

        if args.compare:
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

    if args.compare:
        print(" -> Solving Altimetry-only...")
        post_alt = alt_inv.model_posterior_measure(
            alt_data, solver, preconditioner=P_alt
        )
        print(" -> Solving GRACE-only...")
        post_grace = grace_inv.model_posterior_measure(
            grace_data, solver, preconditioner=P_grace
        )

    print(" -> Solving Joint Inversion...")
    post_joint = joint_inv.model_posterior_measure(
        joint_data, solver, preconditioner=P_joint
    )

    # ------------------ 5. GMSL EXTRACTION ------------------
    true_gmsl_op = utils.true_gmsl_operator(
        exact_phys["state"], exact_phys["load_space"], exact_phys["continuous_sl"]
    )
    true_gmsl_val_mm = true_gmsl_op(true_model)[0] * scale_mm

    alt_avg_op = sl.linear_operators.altimetry_averaging_operator(points)

    # Calculate GMSL Statistics
    std_noise_measure = exact_meas["alt_noise"].affine_mapping(operator=alt_avg_op)
    std_noise_std_mm = (
        np.sqrt(std_noise_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )
    std_alt_gmsl = alt_avg_op(alt_data)[0] * scale_mm

    post_gmsl_joint = post_joint.affine_mapping(operator=true_gmsl_op)
    if args.compare:
        post_gmsl_alt = post_alt.affine_mapping(operator=true_gmsl_op)
        post_gmsl_grace = post_grace.affine_mapping(operator=true_gmsl_op)

    # ------------------ 6. MAPPING ------------------
    if args.plot_maps:
        print("\nGenerating spatial maps...")
        cmap = "seismic"
        ocean_mask_mm = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
        ocean_mask_raw = exact_phys["state"].ocean_projection(value=0.0)
        ice_mask_mm = scale_mm * exact_phys["state"].ice_projection(value=0.0)

        true_ice, true_dyn, true_rho = true_model

        if args.compare:
            # --- COMPARISON MODE MAPS ---
            vmax_ice = max(
                np.max(np.abs(true_ice.data * scale_mm)),
                np.max(np.abs(post_alt.expectation[0].data * scale_mm)),
                np.max(np.abs(post_grace.expectation[0].data * scale_mm)),
                np.max(np.abs(post_joint.expectation[0].data * scale_mm)),
            )
            vmax_dyn = max(
                np.max(np.abs(true_dyn.data * scale_mm)),
                np.max(np.abs(post_alt.expectation[1].data * scale_mm)),
                np.max(np.abs(post_grace.expectation[1].data * scale_mm)),
                np.max(np.abs(post_joint.expectation[1].data * scale_mm)),
            )
            vmax_rho = max(
                np.max(np.abs(true_rho.data * density_scale)),
                np.max(np.abs(post_alt.expectation[2].data * density_scale)),
                np.max(np.abs(post_grace.expectation[2].data * density_scale)),
                np.max(np.abs(post_joint.expectation[2].data * density_scale)),
            )

            titles = ["True State", "Altimetry Only", "GRACE Only", "Joint Inversion"]
            models = [
                true_model,
                post_alt.expectation,
                post_grace.expectation,
                post_joint.expectation,
            ]
            grid_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

            # Ice Maps
            fig_ice, axes_ice = plt.subplots(
                2,
                2,
                figsize=(12, 10),
                subplot_kw={"projection": ccrs.Robinson()},
                layout="constrained",
            )
            for i in range(4):
                idx = grid_indices[i]
                sl.plot(
                    models[i][0] * ice_mask_mm,
                    ax=axes_ice[idx],
                    colorbar=True,
                    vmin=-vmax_ice,
                    vmax=vmax_ice,
                    cmap=cmap,
                    colorbar_kwargs={"label": "Ice Thickness (mm)"},
                )
                axes_ice[idx].set_title(f"{titles[i]}\nIce Thickness Change")
            figures_to_save["compare_maps_ice"] = fig_ice

            # Dyn Maps
            fig_dyn, axes_dyn = plt.subplots(
                2,
                2,
                figsize=(12, 10),
                subplot_kw={"projection": ccrs.Robinson()},
                layout="constrained",
            )
            for i in range(4):
                idx = grid_indices[i]
                sl.plot(
                    models[i][1] * ocean_mask_mm,
                    ax=axes_dyn[idx],
                    colorbar=True,
                    vmin=-vmax_dyn,
                    vmax=vmax_dyn,
                    cmap=cmap,
                    colorbar_kwargs={"label": "Dynamic Topo (mm)"},
                )
                axes_dyn[idx].set_title(f"{titles[i]}\nDynamic Topography")
            figures_to_save["compare_maps_dyn"] = fig_dyn

            # Rho Maps
            fig_rho, axes_rho = plt.subplots(
                2,
                2,
                figsize=(12, 10),
                subplot_kw={"projection": ccrs.Robinson()},
                layout="constrained",
            )
            for i in range(4):
                idx = grid_indices[i]
                sl.plot(
                    models[i][2] * density_scale * ocean_mask_raw,
                    ax=axes_rho[idx],
                    colorbar=True,
                    vmin=-vmax_rho,
                    vmax=vmax_rho,
                    cmap=cmap,
                    colorbar_kwargs={"label": r"Density Anomaly (kg/m$^3$)"},
                )
                axes_rho[idx].set_title(f"{titles[i]}\nOcean Density Change")
            figures_to_save["compare_maps_rho"] = fig_rho

        else:
            # --- DEFAULT JOINT ONLY MAPS ---
            post_ice, post_dyn, post_rho = post_joint.expectation
            vmax_ice = max(
                np.max(np.abs(true_ice.data * scale_mm)),
                np.max(np.abs(post_ice.data * scale_mm)),
            )
            vmax_dyn = max(
                np.max(np.abs(true_dyn.data * scale_mm)),
                np.max(np.abs(post_dyn.data * scale_mm)),
            )
            vmax_rho = max(
                np.max(np.abs(true_rho.data * density_scale)),
                np.max(np.abs(post_rho.data * density_scale)),
            )

            fig_maps, axes = plt.subplots(
                3,
                2,
                figsize=(14, 15),
                subplot_kw={"projection": ccrs.Robinson()},
                layout="constrained",
            )
            sl.plot(
                true_ice * ice_mask_mm,
                ax=axes[0, 0],
                colorbar=True,
                vmin=-vmax_ice,
                vmax=vmax_ice,
                cmap=cmap,
                colorbar_kwargs={"label": "Ice Thickness (mm)"},
            )
            axes[0, 0].set_title("True Ice Change")
            sl.plot(
                post_ice * ice_mask_mm,
                ax=axes[0, 1],
                colorbar=True,
                vmin=-vmax_ice,
                vmax=vmax_ice,
                cmap=cmap,
                colorbar_kwargs={"label": "Ice Thickness (mm)"},
            )
            axes[0, 1].set_title("Posterior Ice Change")

            sl.plot(
                true_dyn * ocean_mask_mm,
                ax=axes[1, 0],
                colorbar=True,
                vmin=-vmax_dyn,
                vmax=vmax_dyn,
                cmap=cmap,
                colorbar_kwargs={"label": "Dynamic Topo (mm)"},
            )
            axes[1, 0].set_title("True Dynamic Topography")
            sl.plot(
                post_dyn * ocean_mask_mm,
                ax=axes[1, 1],
                colorbar=True,
                vmin=-vmax_dyn,
                vmax=vmax_dyn,
                cmap=cmap,
                colorbar_kwargs={"label": "Dynamic Topo (mm)"},
            )
            axes[1, 1].set_title("Posterior Dynamic Topography")

            sl.plot(
                true_rho * density_scale * ocean_mask_raw,
                ax=axes[2, 0],
                colorbar=True,
                vmin=-vmax_rho,
                vmax=vmax_rho,
                cmap=cmap,
                colorbar_kwargs={"label": r"Density Anomaly (kg/m$^3$)"},
            )
            axes[2, 0].set_title("True Ocean Density Change")
            sl.plot(
                post_rho * density_scale * ocean_mask_raw,
                ax=axes[2, 1],
                colorbar=True,
                vmin=-vmax_rho,
                vmax=vmax_rho,
                cmap=cmap,
                colorbar_kwargs={"label": r"Density Anomaly (kg/m$^3$)"},
            )
            axes[2, 1].set_title("Posterior Ocean Density Change")

            if args.plot_regions:
                for ax in axes.flatten():
                    exact_phys["state"].plot_boundaries(ax, regions_to_analyze)
            figures_to_save["joint_posterior_maps"] = fig_maps

            # SSH Observation Map
            true_ssh = exact_phys["continuous_ssh"](true_model)
            obs_data_mm = alt_data * scale_mm
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
                true_ssh * ocean_mask_mm,
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
                s=4,
                edgecolors="none",
                colorbar=True,
                colorbar_kwargs={
                    "label": "Observed SSH Data (mm)",
                    "orientation": "horizontal",
                    "shrink": 0.7,
                    "pad": 0.05,
                },
                zorder=5,
            )
            axes_ssh[1].set_title("Altimetry Observations")

            if args.plot_regions:
                for ax in axes_ssh.flatten():
                    exact_phys["state"].plot_boundaries(ax, regions_to_analyze)
            figures_to_save["joint_observed_ssh"] = fig_ssh

    # ------------------ 7. PDF ------------------
    if args.plot_pdfs:
        print("Plotting GMSL PDFs...")

        class MockMeasure:
            def __init__(self, mean, std):
                self.mean = np.array([mean])
                self.cov = np.array([[std**2]])

        if args.compare:
            results = {
                "Standard Altimetry": MockMeasure(std_alt_gmsl, std_noise_std_mm),
                "Bayesian Altimetry Only": MockMeasure(
                    post_gmsl_alt.expectation[0] * scale_mm,
                    np.sqrt(post_gmsl_alt.covariance.matrix(dense=True)[0, 0])
                    * scale_mm,
                ),
                "Bayesian GRACE Only": MockMeasure(
                    post_gmsl_grace.expectation[0] * scale_mm,
                    np.sqrt(post_gmsl_grace.covariance.matrix(dense=True)[0, 0])
                    * scale_mm,
                ),
                "Joint Bayesian": MockMeasure(
                    post_gmsl_joint.expectation[0] * scale_mm,
                    np.sqrt(post_gmsl_joint.covariance.matrix(dense=True)[0, 0])
                    * scale_mm,
                ),
            }
        else:
            results = {
                "Standard Altimetry": MockMeasure(std_alt_gmsl, std_noise_std_mm),
                "Joint Bayesian": MockMeasure(
                    post_gmsl_joint.expectation[0] * scale_mm,
                    np.sqrt(post_gmsl_joint.covariance.matrix(dense=True)[0, 0])
                    * scale_mm,
                ),
            }

        fig_pdf, ax_pdf = plt.subplots(
            figsize=(10, 6) if args.compare else (8, 5), layout="constrained"
        )
        inf.plot_1d_distributions(
            list(results.values()),
            true_value=true_gmsl_val_mm,
            ax=ax_pdf,
            xlabel="Global Mean Sea Level Change (mm)",
            title="GMSL Estimators",
            posterior_labels=list(results.keys()),
        )
        figures_to_save["extended_gmsl_pdf"] = fig_pdf

    # ------------------ 8. REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")
        op_dyn, op_rho, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        combined_op = inf.ColumnLinearOperator([op_dyn, op_rho, op_ice_fp]) * scale_mm
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(true_model)
        prior_meas = exact_meas["model_prior"].affine_mapping(operator=final_op)
        labels = [
            f"{regions_to_analyze[0]}: Dynamic (mm)",
            f"{regions_to_analyze[0]}: Density",
            f"{regions_to_analyze[0]}: SLE/Ice (mm)",
        ]

        if args.compare:
            posteriors = {
                "altimetry": (
                    "Altimetry-Only",
                    post_alt.affine_mapping(operator=final_op),
                ),
                "grace": ("GRACE-Only", post_grace.affine_mapping(operator=final_op)),
                "joint": (
                    "Joint Inversion",
                    post_joint.affine_mapping(operator=final_op),
                ),
            }
            for key, (title_prefix, meas) in posteriors.items():
                inf.plot_corner_distributions(
                    meas,
                    prior_measure=prior_meas,
                    true_values=true_vals_mm,
                    labels=labels,
                    title=f"{title_prefix} 3-Component Signal Separation",
                    fill_density=False,
                )
                figures_to_save[f"extended_regional_corner_{key}"] = plt.gcf()
        else:
            post_meas = post_joint.affine_mapping(operator=final_op)
            inf.plot_corner_distributions(
                post_meas,
                prior_measure=prior_meas,
                true_values=true_vals_mm,
                labels=labels,
                title="Joint Bayes 3-Component Signal Separation",
                fill_density=False,
            )
            figures_to_save["extended_regional_corner"] = plt.gcf()

    # ------------------ 9. MC VALIDATION (JOINT ONLY) ------------------
    if args.mc_trials > 0 and not args.compare:
        print(f"Running {args.mc_trials} MC trials via dense joint measure mapping...")
        joint_alt_avg_op = inf.RowLinearOperator(
            [
                alt_avg_op,
                exact_meas["grace_noise"].domain.zero_operator(inf.EuclideanSpace(1)),
            ]
        )

        post_exp_op = joint_inv.posterior_expectation_operator(
            solver, preconditioner=P_joint
        )
        bayes_linear = (
            post_exp_op.linear_part
            if isinstance(post_exp_op, inf.AffineOperator)
            else post_exp_op
        )
        bayes_translation = (
            true_gmsl_op(post_exp_op.translation_part)
            if isinstance(post_exp_op, inf.AffineOperator)
            else None
        )

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
        joint_err_meas = joint_inv.joint_prior_measure.affine_mapping(
            operator=joint_err_op, translation=translation
        )
        joint_err_dense = joint_err_meas.with_dense_covariance(parallel=True, n_jobs=4)

        samples = joint_err_dense.samples(args.mc_trials)
        std_errs = np.array([(s[0] * scale_mm) / std_noise_std_mm for s, _ in samples])
        bayes_errs = np.array(
            [
                (b[0] * scale_mm)
                / np.sqrt(
                    post_gmsl_joint.covariance.matrix(dense=True)[0, 0] * scale_mm**2
                )
                for _, b in samples
            ]
        )

        raw_cov = joint_err_dense.covariance.matrix(dense=True) * (scale_mm**2)
        raw_mean = joint_err_dense.expectation
        post_gmsl_std = (
            np.sqrt(post_gmsl_joint.covariance.matrix(dense=True)[0, 0]) * scale_mm
        )

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
                (raw_mean[1][0] * scale_mm) / post_gmsl_std,
            ]
        )
        cov_2d = np.array(
            [
                [
                    raw_cov[0, 0] / (std_noise_std_mm**2),
                    raw_cov[0, 1] / (std_noise_std_mm * post_gmsl_std),
                ],
                [
                    raw_cov[0, 1] / (std_noise_std_mm * post_gmsl_std),
                    raw_cov[1, 1] / (post_gmsl_std**2),
                ],
            ]
        )

        x_grid, y_grid = np.mgrid[
            -plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j
        ]
        rv = stats.multivariate_normal(mu_2d, cov_2d)
        ax_mc.contour(
            x_grid,
            y_grid,
            rv.pdf(np.dstack((x_grid, y_grid))),
            levels=[rv.pdf(mu_2d) * np.exp(-0.5 * k**2) for k in [4, 3, 2, 1]],
            colors="indigo",
            linewidths=[0.5, 1.0, 1.5],
            alpha=0.8,
            zorder=4,
        )

        ax_mc.axhline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
        ax_mc.axvline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
        ax_mc.axhspan(
            -1, 1, color="blue", alpha=0.15, zorder=0, label=r"Bayes 1$\sigma$ Expected"
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
        ax_mc.set_xlabel(r"Standard Estimator Normalized Error", fontsize=12)
        ax_mc.set_ylabel(r"Joint Bayesian Normalized Error", fontsize=12)
        ax_mc.set_title("GMSL MC Validation: Normalized Residuals", fontsize=16)
        ax_mc.plot([], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF")
        ax_mc.legend(loc="upper left", fontsize=10)
        figures_to_save["joint_mc_validation"] = fig_mc

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
