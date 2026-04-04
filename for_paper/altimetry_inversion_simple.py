"""
Bayesian Inversion vs. Standard Averaging (Altimetry Analysis)
==============================================================

This script performs a Bayesian inversion of synthetic satellite altimetry data
to estimate the underlying ice thickness changes and resulting Global Mean Sea Level (GMSL).

(Stripped Down Version: Ice Thickness Only)
"""

import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    """Parses command-line arguments to toggle simulation and plot options."""
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of Altimetry data (Ice Only)."
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
        "--mc-trials",
        type=int,
        default=0,
        help="Number of Monte Carlo trials for statistical comparison of estimators.",
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=64,
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
        "--spacing-degrees",
        type=float,
        default=10.0,
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
        default=200.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.5,
        help="Instrument noise std per point as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=0.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    return parser.parse_args()


def build_physics_components(lmax, points, args, max_iterations=None):
    """
    Constructs the physical operators and Bayesian measures for a specific SH degree.
    """
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    scale_mm = 1000.0 * fp.length_scale

    load_order = args.load_order
    load_scale = args.load_scale_km * 1000.0 / fp.length_scale
    finger_print_operator = fp.as_sobolev_linear_operator(
        load_order, load_scale, max_iterations=max_iterations
    )

    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    ice_to_load_operator = sl.ice_thickness_change_to_load_operator(fp, load_space)

    response_space_to_ssh_operator = sl.sea_surface_height_operator(fp, response_space)
    ssh_inclusion = response_space_to_ssh_operator.codomain.order_inclusion_operator(
        load_space.order
    )

    point_evaluation_operator = load_space.point_evaluation_operator(points)

    # Simplified forward model chain: Ice -> Load -> Response -> SSH
    model_to_ssh_operator = (
        ssh_inclusion
        @ response_space_to_ssh_operator
        @ finger_print_operator
        @ ice_to_load_operator
    )
    forward_op = point_evaluation_operator @ model_to_ssh_operator

    # Priors
    ice_thickness_scale = args.ice_scale_km * 1000.0 / fp.length_scale
    ice_thickness_std = args.ice_std_mm / scale_mm

    # Model prior is now just the ice thickness prior
    model_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_thickness_scale, std=ice_thickness_std
    )

    if args.prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(args.prior_shift, offset_shape)
        )

    # GMSL Operator for noise scaling
    GMSL_weighting_function = (
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        / (fp.water_density * fp.ocean_area)
    )

    B = sl.averaging_operator(load_space, [GMSL_weighting_function])
    GMSL_prior_measure = model_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    # Noise
    noise_std = args.noise_std_factor * GMSL_prior_std
    n_points = len(points)
    data_space = inf.EuclideanSpace(n_points)
    noise_meas = inf.GaussianMeasure.from_standard_deviations(
        data_space, np.full(n_points, noise_std)
    )

    return (
        fp,
        load_space,
        model_to_ssh_operator,
        forward_op,
        model_prior,
        noise_meas,
        GMSL_prior_std,
        scale_mm,
    )


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = True
        if args.mc_trials == 0:
            args.mc_trials = 500

    # Create a dummy fingerprint just to extract the standard altimetry points
    print("Generating altimetry points...")
    dummy_fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    dummy_fp.set_state_from_ice_ng()
    points = dummy_fp.ocean_altimetry_points(spacing_degrees=args.spacing_degrees)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    # ------------------ EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT physical operators (lmax={args.lmax})...")
    (
        exact_fp,
        exact_load_space,
        exact_model_to_ssh,
        exact_forward_op,
        exact_model_prior,
        noise_meas,
        GMSL_prior_std,
        scale_mm,
    ) = build_physics_components(args.lmax, points, args)

    print(
        f"Implied GMSL standard deviation from ice prior: {GMSL_prior_std * scale_mm:.3f} mm"
    )

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
            f"\nBuilding SURROGATE physical operators (lmax={args.surrogate_degree}) for preconditioning..."
        )
        _, _, _, surr_forward_op, surr_prior, _, _, _ = build_physics_components(
            args.surrogate_degree, points, args, max_iterations=3
        )

        print("Constructing Woodbury preconditioner from surrogate model...")
        surrogate_inv = inverse_problem.surrogate_inversion(
            alternate_forward_operator=surr_forward_op,
            alternate_prior_measure=surr_prior,
        )
        # Using parallel=True to spread the SLE solver evaluations across CPU cores
        preconditioner = surrogate_inv.woodbury_data_preconditioner(
            parallel=True, n_jobs=-1
        )

    # ------------------ POSTERIOR SOLVE ------------------

    callback = inverse_problem.normal_residual_callback(synthetic_data)
    solver = inf.CGMatrixSolver(callback=callback)

    print("\nSolving for posterior expectation...")
    model_posterior = inverse_problem.model_posterior_measure(
        synthetic_data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    # ------------------ EXTRACT GMSL OPERATORS ------------------
    true_avg_weight = exact_fp.ocean_function / exact_fp.ocean_area
    true_avg_op = sl.averaging_operator(exact_load_space, [true_avg_weight])

    # Operator mapping full model directly to GMSL
    true_gmsl_op = true_avg_op @ exact_model_to_ssh

    # Standard Estimator Operator (acts on data)
    alt_avg_op = sl.altimetry_averaging_operator(points)

    # Posterior GMSL Distribution
    post_gmsl_measure = model_posterior.affine_mapping(operator=true_gmsl_op)
    post_gmsl_std_mm = (
        np.sqrt(post_gmsl_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    # Standard Estimator Noise Distribution (Variance around the estimated mean)
    std_noise_measure = noise_meas.affine_mapping(operator=alt_avg_op)
    std_noise_std_mm = (
        np.sqrt(std_noise_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    # Exact True GMSL
    true_gmsl_val_mm = true_gmsl_op(true_model)[0] * scale_mm

    # ------------------ OPTION 1: MAPS ------------------
    if args.plot_maps:
        print("Generating spatial maps...")

        true_ice = true_model
        post_ice = model_posterior.expectation

        true_ssh = exact_model_to_ssh(true_model)
        post_ssh = exact_model_to_ssh(post_ice)

        ocean_mask = scale_mm * exact_fp.ocean_projection(value=0)
        ice_mask = scale_mm * exact_fp.ice_projection(value=0)

        vmax_ice = max(
            np.max(np.abs(true_ice.data * scale_mm)),
            np.max(np.abs(post_ice.data * scale_mm)),
        )

        vmax_ssh = max(
            np.max(np.abs(true_ssh.data * scale_mm)),
            np.max(np.abs(post_ssh.data * scale_mm)),
        )

        fig_maps, axes_maps = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        # Plot True Ice
        sl.plot(
            true_ice * ice_mask,
            ax=axes_maps[0],
            colorbar=True,
            colorbar_kwargs={"label": "True Ice Thickness Change (mm)"},
            vmin=-vmax_ice,
            vmax=vmax_ice,
            symmetric=True,
        )
        axes_maps[0].set_title("True Ice Thickness Change")

        # Plot Posterior Expected Ice
        sl.plot(
            post_ice * ice_mask,
            ax=axes_maps[1],
            colorbar=True,
            colorbar_kwargs={"label": "Expected Ice Thickness Change (mm)"},
            vmin=-vmax_ice,
            vmax=vmax_ice,
            symmetric=True,
        )
        axes_maps[1].set_title("Posterior Expected Ice Thickness")

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

        # Provide the preconditioner to the expectation operator so MC trials solve quickly!
        post_exp_op = inverse_problem.posterior_expectation_operator(
            solver, preconditioner=preconditioner
        )

        if isinstance(post_exp_op, inf.AffineOperator):
            bayes_linear = post_exp_op.linear_part
            bayes_translation = true_gmsl_op(post_exp_op.translation_part)
        else:
            bayes_linear = post_exp_op
            bayes_translation = None

        # Build combined linear mappings for errors on the joint space [model, data]
        std_err_op = inf.RowLinearOperator([-1.0 * true_gmsl_op, alt_avg_op])

        bayes_err_op = inf.RowLinearOperator(
            [-1.0 * true_gmsl_op, true_gmsl_op @ bayes_linear]
        )

        joint_err_op = inf.ColumnLinearOperator([std_err_op, bayes_err_op])

        if bayes_translation is not None:
            translation = [true_gmsl_op.codomain.zero, bayes_translation]
        else:
            translation = None

        joint_meas = inverse_problem.joint_prior_measure
        joint_err_meas = joint_meas.affine_mapping(
            operator=joint_err_op, translation=translation
        )

        print("Constructing dense error covariance...")
        joint_err_dense = joint_err_meas.with_dense_covariance()
        samples = joint_err_dense.samples(args.mc_trials)

        std_errs = np.zeros(args.mc_trials)
        bayes_errs = np.zeros(args.mc_trials)

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

        # Analytical 2D PDF Contours
        mu_2d = np.array(
            [
                (raw_mean[0][0] * scale_mm) / std_noise_std_mm,
                (raw_mean[1][0] * scale_mm) / post_gmsl_std_mm,
            ]
        )

        var_s = raw_cov[0, 0] / (std_noise_std_mm**2)
        var_b = raw_cov[1, 1] / (post_gmsl_std_mm**2)
        cov_sb = raw_cov[0, 1] / (std_noise_std_mm * post_gmsl_std_mm)
        cov_2d = np.array([[var_s, cov_sb], [cov_sb, var_b]])

        x_grid, y_grid = np.mgrid[
            -plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j
        ]
        pos = np.dstack((x_grid, y_grid))
        rv = stats.multivariate_normal(mu_2d, cov_2d)
        Z = rv.pdf(pos)

        max_density = rv.pdf(mu_2d)
        levels = [max_density * np.exp(-0.5 * k**2) for k in [4, 3, 2, 1]]
        ax_mc.contour(
            x_grid,
            y_grid,
            Z,
            levels=levels,
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
            label=r"Bayes 1$\sigma$ Expected",
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

        ax_mc.plot([], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF")
        ax_mc.legend(loc="upper left", fontsize=10)
        ax_mc.set_title("GMSL MC Validation: Normalized Residuals", fontsize=16)

    if any([args.plot_maps, args.plot_pdfs, args.mc_trials]):
        plt.show()


if __name__ == "__main__":
    main()
