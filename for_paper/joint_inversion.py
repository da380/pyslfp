"""
Joint Bayesian Inversion (GRACE + Satellite Altimetry)
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

import matplotlib

# Force headless backend to avoid Wayland/Qt display errors

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf


import joint_utils as utils
from plot_utils import plot_normalized_mc_errors

import pyslfp as sl
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points


matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian Inversion (Ice, Dyn, Rho) with Optional Comparison Mode."
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

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        args.compare = True
        if args.mc_trials <= 0:
            args.mc_trials = -1

    output_dir = "output_plots_joint_inversion"
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

    # ------------------ 5. GMSL & MC SETUP ------------------
    if args.plot_pdfs or (args.mc_trials != 0 and not args.compare):

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

        post_gmsl_joint = post_joint.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        if args.compare:
            post_gmsl_alt = post_alt.affine_mapping(
                operator=true_gmsl_op
            ).with_dense_covariance()
            post_gmsl_grace = post_grace.affine_mapping(
                operator=true_gmsl_op
            ).with_dense_covariance()

        true_gmsl_val_mm = true_gmsl_op(true_model)[0]

        if args.plot_pdfs:
            print("Plotting GMSL PDFs...")
            measures = [alt_gmsl_measure]
            labels = ["Simple averaging"]

            if args.compare:
                measures.extend([post_gmsl_alt, post_gmsl_grace, post_gmsl_joint])
                labels.extend(
                    [
                        f"Alt-Only ({post_gmsl_alt.kl_divergence(prior_gmsl_measure):.2f} nats)",
                        f"GRACE-Only ({post_gmsl_grace.kl_divergence(prior_gmsl_measure):.2f} nats)",
                        f"Joint Bayes ({post_gmsl_joint.kl_divergence(prior_gmsl_measure):.2f} nats)",
                    ]
                )
            else:
                measures.append(post_gmsl_joint)
                labels.append(
                    f"Joint Bayes ({post_gmsl_joint.kl_divergence(prior_gmsl_measure):.2f} nats)"
                )

            fig_pdf, ax_pdf = plt.subplots(
                figsize=(10, 6) if args.compare else (8, 5), layout="constrained"
            )
            inf.plot_1d_distributions(
                measures,
                true_value=true_gmsl_val_mm,
                ax=ax_pdf,
                title="Global Mean Sea Level Estimators",
                posterior_labels=labels,
            )
            figures_to_save["gmsl_pdf"] = fig_pdf

        if args.mc_trials != 0 and not args.compare:
            print(
                f"\nExtracting analytical distributions for MC validation (trials={'skipped' if args.mc_trials == -1 else args.mc_trials})..."
            )

            post_exp_op = joint_inv.posterior_expectation_operator(
                solver, preconditioner=P_joint
            )

            if isinstance(post_exp_op, inf.AffineOperator):
                bayes_linear = post_exp_op.linear_part
                bayes_translation = true_gmsl_op(post_exp_op.translation_part)
            else:
                bayes_linear = post_exp_op
                bayes_translation = None

            # Route standard altimetry averaging solely through the altimetry subspace
            grace_space = exact_meas["grace_noise"].domain
            joint_alt_avg_op = inf.RowLinearOperator(
                [
                    alt_avg_op,
                    grace_space.zero_operator(codomain=alt_avg_op.codomain),
                ]
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

            joint_err_dense = joint_err_meas.with_dense_covariance(
                parallel=True, n_jobs=4
            )

            if args.mc_trials > 0:
                samples = joint_err_dense.samples(args.mc_trials)
                raw_errs_x = np.zeros(args.mc_trials)
                raw_errs_y = np.zeros(args.mc_trials)
                for i, (s_err, b_err) in enumerate(samples):
                    raw_errs_x[i] = s_err[0]
                    raw_errs_y[i] = b_err[0]
            else:
                raw_errs_x, raw_errs_y = None, None

            std_noise_std_mm = np.sqrt(
                alt_gmsl_measure.covariance.matrix(dense=True)[0, 0]
            )
            post_gmsl_std_mm = np.sqrt(
                post_gmsl_joint.covariance.matrix(dense=True)[0, 0]
            )

            raw_cov = joint_err_dense.covariance.matrix(dense=True)
            raw_mean = joint_err_dense.expectation
            raw_mean_2d = [raw_mean[0][0], raw_mean[1][0]]

            fig_mc, ax_mc = plt.subplots(figsize=(7, 7), layout="constrained")

            plot_normalized_mc_errors(
                ax_mc,
                raw_errs_x,
                raw_errs_y,
                raw_mean_2d,
                raw_cov,
                std_noise_std_mm,
                post_gmsl_std_mm,
                title="GMSL MC Validation: Normalized Residuals",
                xlabel=r"Standard Estimator Normalized Error",
                ylabel=r"Joint Bayesian Normalized Error",
                label_x="Standard",
                label_y="Joint Bayes",
                show_legend=True,
            )
            figures_to_save["joint_mc_validation"] = fig_mc

    # ------------------ 6. MAPPING ------------------
    if args.plot_maps:
        print("\nGenerating spatial maps...")
        cmap = "seismic"

        # Steric scale conversion
        mean_ocean_depth = (
            exact_phys["state"].model.integrate(exact_phys["state"].sea_level)
            / exact_phys["state"].ocean_area
        )
        water_density = exact_phys["state"].model.parameters.water_density
        steric_scale = mean_ocean_depth / water_density

        ocean_mask_mm = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
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
                np.max(np.abs(true_rho.data * steric_scale * scale_mm)),
                np.max(np.abs(post_alt.expectation[2].data * steric_scale * scale_mm)),
                np.max(
                    np.abs(post_grace.expectation[2].data * steric_scale * scale_mm)
                ),
                np.max(
                    np.abs(post_joint.expectation[2].data * steric_scale * scale_mm)
                ),
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
                    models[i][2] * steric_scale * ocean_mask_mm,
                    ax=axes_rho[idx],
                    colorbar=True,
                    vmin=-vmax_rho,
                    vmax=vmax_rho,
                    cmap=cmap,
                    colorbar_kwargs={"label": "Steric SL (mm)"},
                )
                axes_rho[idx].set_title(f"{titles[i]}\nSteric Sea Level Change")
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
                np.max(np.abs(true_rho.data * steric_scale * scale_mm)),
                np.max(np.abs(post_rho.data * steric_scale * scale_mm)),
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
                true_rho * steric_scale * ocean_mask_mm,
                ax=axes[2, 0],
                colorbar=True,
                vmin=-vmax_rho,
                vmax=vmax_rho,
                cmap=cmap,
                colorbar_kwargs={"label": r"Steric SL (mm)"},
            )
            axes[2, 0].set_title("True Steric Sea Level Change")
            sl.plot(
                post_rho * steric_scale * ocean_mask_mm,
                ax=axes[2, 1],
                colorbar=True,
                vmin=-vmax_rho,
                vmax=vmax_rho,
                cmap=cmap,
                colorbar_kwargs={"label": r"Steric SL (mm)"},
            )
            axes[2, 1].set_title("Posterior Steric Sea Level Change")

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

    # ------------------ 7. REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")
        op_dyn, op_rho, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        mean_ocean_depth = (
            exact_phys["state"].model.integrate(exact_phys["state"].sea_level)
            / exact_phys["state"].ocean_area
        )
        water_density = exact_phys["state"].model.parameters.water_density
        steric_scale = mean_ocean_depth / water_density

        combined_op = (
            inf.ColumnLinearOperator([op_dyn, op_rho * steric_scale, op_ice_fp])
            * scale_mm
        )
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(true_model)
        prior_meas = exact_meas["model_prior"].affine_mapping(operator=final_op)
        labels = [
            f"{regions_to_analyze[0]}: Dynamic SL (mm)",
            f"{regions_to_analyze[0]}: Steric SL (mm)",
            f"{regions_to_analyze[0]}: Barystatic (mm)",
        ]

        if args.compare:
            posteriors = {
                "altimetry": (
                    "Altimetry-Only",
                    post_alt.affine_mapping(operator=final_op).with_dense_covariance(
                        parallel=True, n_jobs=3
                    ),
                ),
                "grace": (
                    "GRACE-Only",
                    post_grace.affine_mapping(operator=final_op).with_dense_covariance(
                        parallel=True, n_jobs=3
                    ),
                ),
                "joint": (
                    "Joint Inversion",
                    post_joint.affine_mapping(operator=final_op).with_dense_covariance(
                        parallel=True, n_jobs=3
                    ),
                ),
            }
            for key, (title_prefix, meas) in posteriors.items():
                kl_div = meas.kl_divergence(prior_meas)
                inf.plot_corner_distributions(
                    meas,
                    prior_measure=prior_meas,
                    true_values=true_vals_mm,
                    labels=labels,
                    title=f"{title_prefix} 3-Component Signal Separation ({kl_div:.2f} nats)",
                    fill_density=False,
                )
                figures_to_save[f"regional_corner_{key}"] = plt.gcf()
        else:
            post_meas = post_joint.affine_mapping(
                operator=final_op
            ).with_dense_covariance(parallel=True, n_jobs=3)
            kl_div = post_meas.kl_divergence(prior_meas)
            inf.plot_corner_distributions(
                post_meas,
                prior_measure=prior_meas,
                true_values=true_vals_mm,
                labels=labels,
                title=f"Joint Bayes 3-Component Signal Separation ({kl_div:.2f} nats)",
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
