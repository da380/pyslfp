"""
Extended Bayesian Altimetry Inversion (3-Component Model)
=========================================================

This script performs a Bayesian inversion of synthetic satellite altimetry data.
It estimates the underlying ice thickness changes, ocean dynamic topography,
and ocean density changes (effective steric sea level), while strictly enforcing
ocean mass conservation. Includes analytical MC error validation.
"""

import argparse
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

import pygeoinf as inf

import altimetry_utils as utils
from plot_utils import plot_normalized_mc_errors

import pyslfp as sl
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points

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
        "--plot-maps", action="store_true", help="Plot 3-component spatial maps."
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot 3-way regional signal decomposition.",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of MC trials for analytical error validation.",
    )

    # --- Resolution Settings ---
    parser.add_argument(
        "--lmax", type=int, default=128, help="Exact model max SH degree."
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
        "--noise-std-factor",
        type=float,
        default=1.0,
        help="Instrument noise std as factor of GMSL std.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.0,
        help="Instrument noise correlation scale.",
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        if args.mc_trials <= 0:
            args.mc_trials = -1

    output_dir = "output_plots_altimetry_inversion"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    inf.configure_threading(n_threads=1)

    regions_to_analyze = ["Tasman Sea"]

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT 3-Component physical operators (lmax={args.lmax})...")
    (state, load_space, fp_op, continuous_ssh, continuous_sl, forward_op, mm_scale) = (
        utils.build_physics_components(
            args.lmax, args.load_order, args.load_scale_km, points, is_surrogate=False
        )
    )

    model_prior, noise_measure, GMSL_prior_std = utils.build_measures(
        state,
        load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.noise_scale_factor,
        args.noise_std_factor,
        points,
        mm_scale,
        prior_shift=args.prior_shift,
        is_surrogate=False,
    )

    print(f"Implied GMSL prior standard deviation: {GMSL_prior_std * mm_scale:.3f} mm")

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
    (surr_state, surr_load_space, _, _, _, surr_forward_op, _) = (
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
        args.noise_scale_factor,
        args.noise_std_factor,
        points,
        mm_scale,
        prior_shift=args.prior_shift,
        is_surrogate=True,
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

    # ------------------ 4. GMSL & MC SETUP ------------------

    if args.plot_pdfs or args.mc_trials != 0:

        true_gmsl_op = (
            utils.true_gmsl_operator(state, load_space, continuous_sl) * mm_scale
        )
        alt_avg_op = sl.linear_operators.altimetry_averaging_operator(points) * mm_scale

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

        if args.plot_pdfs:

            kl_div = post_gmsl_measure.kl_divergence(prior_gmsl_measure)

            fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5), layout="constrained")

            inf.plot_1d_distributions(
                [post_gmsl_measure, alt_gmsl_measure],
                true_value=true_gmsl,
                ax=ax_pdf,
                title="Global Mean Sea Level Estimators",
                posterior_labels=[f"Bayesian ({kl_div:.2f} nats)", "Simple averaging"],
            )
            figures_to_save["gmsl_pdf"] = fig_pdf

        if args.mc_trials != 0:

            print(
                f"\nExtracting analytical distributions for MC validation (trials={'skipped' if args.mc_trials == -1 else args.mc_trials})..."
            )

            post_exp_op = inverse_problem.posterior_expectation_operator(
                solver, preconditioner=preconditioner
            )

            if isinstance(post_exp_op, inf.AffineOperator):
                bayes_linear = post_exp_op.linear_part
                bayes_translation = true_gmsl_op(post_exp_op.translation_part)
            else:
                bayes_linear = post_exp_op
                bayes_translation = None

            std_err_op = inf.RowLinearOperator([-1.0 * true_gmsl_op, alt_avg_op])
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
                post_gmsl_measure.covariance.matrix(dense=True)[0, 0]
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
                ylabel=r"3-Component Bayesian Normalized Error",
                label_x="Standard",
                label_y="Bayes",
                show_legend=True,
            )

            figures_to_save["mc_validation"] = fig_mc

    if args.plot_maps:
        print("Generating 3-component spatial maps...")
        cmap = "seismic"

        density_scale = state.model.parameters.density_scale

        ocean_mask_mm = mm_scale * state.ocean_projection(value=0.0)
        ocean_mask_raw = state.ocean_projection(value=0.0)

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
            true_ice * mm_scale,
            ax=axes[0, 0],
            colorbar=True,
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
            colorbar_kwargs={"label": "Ice Thickness (mm)"},
        )
        axes[0, 0].set_title("True Ice Change")
        sl.plot(
            post_ice * mm_scale,
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
                state.plot_boundaries(ax, regions_to_analyze)

        figures_to_save["posterior_maps"] = fig_maps

        print("Generating Sea Surface Height maps with observation overlays...")

        true_ssh = continuous_ssh(model)

        data_mm = data * mm_scale

        vmax_ssh = max(
            np.max(np.abs(true_ssh.data * mm_scale)), np.max(np.abs(data_mm))
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
            data=data_mm,
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

        figures_to_save["observed_ssh"] = fig_ssh

    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")

        masks = [state.get_projection(r, value=0.0) for r in regions_to_analyze]
        avg_op = sl.linear_operators.averaging_operator(state, load_space, masks)
        joint_space = inf.HilbertSpaceDirectSum([load_space, load_space, load_space])

        op_dyn = avg_op @ joint_space.subspace_projection(1)
        op_rho = avg_op @ joint_space.subspace_projection(2)

        ice_to_load = sl.linear_operators.ice_thickness_change_to_load_operator(
            state, load_space, load_space
        )
        barystatic_sl_op = fp_op.codomain.subspace_projection(0) @ fp_op
        op_ice_fp = (
            avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
        )

        combined_op = inf.ColumnLinearOperator([op_dyn, op_rho, op_ice_fp]) * mm_scale
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(model)
        post_meas = model_posterior.affine_mapping(operator=final_op)
        prior_meas = model_prior.affine_mapping(operator=final_op)

        labels = [
            f"{regions_to_analyze[0]}: Dynamic (mm)",
            f"{regions_to_analyze[0]}: Density",
            f"{regions_to_analyze[0]}: SLE/Ice (mm)",
        ]

        inf.plot_corner_distributions(
            post_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="3-Component Signal Separation",
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
