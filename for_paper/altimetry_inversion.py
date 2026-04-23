"""
Bayesian Inversion vs. Standard Averaging (Altimetry Analysis)
==============================================================

This script performs a Bayesian inversion of synthetic satellite altimetry data
to estimate the underlying ice thickness changes, dynamic ocean topography, and
resulting Global Mean Sea Level (GMSL).
"""

import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl

from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator

import altimetry_utils as utils


def parse_arguments():
    """Parses command-line arguments to toggle simulation and plot options."""
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of Altimetry data with Woodbury Preconditioning."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all plotting options and run a small sample batch for MC and posterior variance.",
    )
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot 1D analytical PDFs of the GMSL estimate (Bayesian vs Standard).",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps of true loads, posterior expectations, and sea surface heights.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot separated regional sea level drivers.",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of Monte Carlo trials for statistical comparison of estimators.",
    )

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
        help="Maximum spherical harmonic degree for the surrogate preconditioner model.",
    )
    parser.add_argument(
        "--no-precond",
        action="store_true",
        help="Disable the surrogate sparse preconditioner.",
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
    parser.add_argument(
        "--spacing",
        type=float,
        default=4.0,
        help="Spacing in degrees for the altimetry observation points.",
    )
    parser.add_argument(
        "--ice-scale-factor",
        type=float,
        default=1.0,
        help="Relative correlation length scale for the ice thickness prior.",
    )
    parser.add_argument(
        "--ice-std-mm",
        type=float,
        default=10.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ocean-scale-factor",
        type=float,
        default=0.2,
        help="Relative correlation length scale for the ocean dynamic thickness prior.",
    )
    parser.add_argument(
        "--ocean-std-factor",
        type=float,
        default=1.0,
        help="Ocean dynamic thickness noise standard deviation as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.2,
        help="Instrument noise standard deviation per point as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.0,
        help="Relative correlation length scale for the noise field.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=1.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        if args.mc_trials == 0:
            args.mc_trials = 500

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    # ------------------ EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT physical operators (lmax={args.lmax})...")
    (
        exact_state,
        exact_load_space,
        exact_fp_op,
        exact_continuous_ssh,
        exact_forward_op,
        scale_mm,
    ) = utils.build_physics_components(
        args.lmax, args.load_order, args.load_scale_km, points, is_surrogate=False
    )

    exact_model_prior, noise_meas, GMSL_prior_std = utils.build_measures(
        exact_state,
        exact_load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_scale_factor,
        args.ocean_std_factor,
        args.noise_scale_factor,
        args.noise_std_factor,
        points,
        scale_mm,
        prior_shift=args.prior_shift,
    )

    print(f"Implied GMSL prior standard deviation: {GMSL_prior_std * scale_mm:.3f} mm")

    print("Setting up Bayesian Inversion...")
    forward_problem = inf.LinearForwardProblem(
        exact_forward_op, data_error_measure=noise_meas
    )
    true_model, synthetic_data = forward_problem.synthetic_model_and_data(
        exact_model_prior
    )
    inverse_problem = inf.LinearBayesianInversion(forward_problem, exact_model_prior)

    # ------------------ PRECONDITIONER SETUP ------------------
    preconditioner = None
    if not args.no_precond:
        print(
            f"\nBuilding 'Physics-Lite' SURROGATE operators (lmax={args.surrogate_degree}) for preconditioning..."
        )
        (surr_state, surr_load_space, _, _, surr_forward_op, _) = (
            utils.build_physics_components(
                args.surrogate_degree,
                args.load_order,
                args.load_scale_km,
                points,
                is_surrogate=True,
            )
        )

        surr_prior, surr_noise_meas, _ = utils.build_measures(
            surr_state,
            surr_load_space,
            args.ice_scale_factor,
            args.ice_std_mm,
            args.ocean_scale_factor,
            args.ocean_std_factor,
            args.noise_scale_factor,
            args.noise_std_factor,
            points,
            scale_mm,
            prior_shift=args.prior_shift,
        )

        print("Constructing Woodbury preconditioner from surrogate model...")
        woodbury_solver = inf.CholeskySolver(galerkin=True)

        preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
            woodbury_solver,
            alternate_forward_operator=surr_forward_op,
            alternate_prior_measure=surr_prior,
            alternate_data_error_measure=surr_noise_meas,
        )

    # ------------------ POSTERIOR SOLVE ------------------
    callback = inf.ProgressCallback()
    solver = inf.CGMatrixSolver(callback=callback)

    print("\nSolving for posterior expectation...")
    model_posterior = inverse_problem.model_posterior_measure(
        synthetic_data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    # ------------------ EXTRACT GMSL OPERATORS ------------------
    if args.plot_pdfs or args.mc_trials:
        true_gmsl_op = utils.true_gmsl_operator(
            exact_state, exact_load_space, exact_continuous_ssh
        )
        alt_avg_op = altimetry_averaging_operator(points)

        post_gmsl_measure = model_posterior.affine_mapping(operator=true_gmsl_op)
        post_gmsl_std_mm = (
            np.sqrt(post_gmsl_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
        )

        std_noise_measure = noise_meas.affine_mapping(operator=alt_avg_op)
        std_noise_std_mm = (
            np.sqrt(std_noise_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
        )

        true_gmsl_val_mm = true_gmsl_op(true_model)[0] * scale_mm

    # ------------------ DEFINE REGIONS FOR ANALYSIS ------------------
    regions_to_analyze = ["Mediterranean Sea - Western Basin", "South Atlantic Ocean"]

    # ------------------ OPTION 1: MAPS ------------------
    if args.plot_maps:
        print("Generating spatial maps...")
        cmap = "seismic"

        true_ice, true_ocean = true_model
        post_ice, post_ocean = model_posterior.expectation

        ocean_mask = scale_mm * exact_state.ocean_projection(value=0.0)
        ice_mask = scale_mm * exact_state.ice_projection(value=0.0)

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
        axes_maps[0, 1].set_title("Posterior Expected Ice Thickness")

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
        axes_maps[1, 1].set_title("Posterior Expected Dynamic Ocean Component")

        # Overlay boundaries on the component maps if regional analysis is active
        if args.plot_regions:
            for ax in axes_maps.flatten():
                exact_state.plot_boundaries(ax, regions_to_analyze)

        # Sea Surface Height Observations
        print("Generating Sea Surface Height maps with observation overlays...")
        true_ssh = exact_continuous_ssh(true_model)
        obs_data_mm = synthetic_data * scale_mm
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

        # Overlay boundaries on the SSH maps if regional analysis is active
        if args.plot_regions:
            for ax in axes_ssh:
                exact_state.plot_boundaries(
                    ax, regions_to_analyze, edgecolor="black", linewidth=2.0, zorder=10
                )

    # ------------------ OPTION 2: PDF ------------------
    if args.plot_pdfs:
        print("Plotting Head-to-Head GMSL PDF...")

        class MockMeasure:
            def __init__(self, m, s):
                self.mean = np.array([m])
                self.cov = np.array([[s**2]])

        results = {
            "Bayesian Inversion": MockMeasure(
                post_gmsl_measure.expectation[0] * scale_mm, post_gmsl_std_mm
            ),
            "Standard Averaging": MockMeasure(
                alt_avg_op(synthetic_data)[0] * scale_mm, std_noise_std_mm
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
        ax_mc.set_aspect("equal")
        ax_mc.set_xlabel(r"Standard Estimator Normalized Error", fontsize=12)
        ax_mc.set_ylabel(r"Bayesian Estimator Normalized Error", fontsize=12)
        ax_mc.set_title("GMSL MC Validation: Normalized Residuals", fontsize=16)

        ax_mc.plot([], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF")
        ax_mc.legend(loc="upper left", fontsize=10)

    # ------------------ OPTION 4: REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals...")
        # Using built-in IHO sea boundaries

        op_dynamic, op_ice_fp = utils.regional_decomposition_operators(
            exact_state, exact_load_space, exact_fp_op, regions_to_analyze
        )

        # Combine into a single operator and scale directly to mm
        combined_op = inf.ColumnLinearOperator([op_dynamic, op_ice_fp]) * scale_mm

        # Flatten for convenience to an operator on Euclidean space.
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        # Apply the completely flattened operator to the True Model
        true_vals_mm = final_op(
            true_model
        )  # Append the flattening step to the end of the chain

        # Extract Analytical Marginal Posteriors and Priors (now guaranteed to be flat)
        post_meas = model_posterior.affine_mapping(operator=final_op)
        prior_meas = exact_model_prior.affine_mapping(operator=final_op)

        labels = [
            "Med: Dynamic (mm)",
            "Carib: Dynamic (mm)",
            "Med: Ice/SLE (mm)",
            "Carib: Ice/SLE (mm)",
        ]

        # Plotting using pygeoinf's corner plot
        inf.plot_corner_distributions(
            post_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="Bayesian Signal Separation: Dynamic Ocean vs. Ice Melt",
            fill_density=False,
        )

    # ----------------------------------------------------------------------
    if any([args.plot_maps, args.plot_pdfs, args.mc_trials, args.plot_regions]):
        plt.show()


if __name__ == "__main__":
    main()
