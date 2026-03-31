"""
Bayesian Inversion vs. WMB Method (Comprehensive Analysis)
==========================================================

This script performs a Bayesian inversion of synthetic GRACE gravimetry data to
estimate regional surface mass changes and compares the results head-to-head
with the standard WMB method.
"""

import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl
import grace_utils as utils


def parse_arguments():
    """Parses command-line arguments to toggle simulation and plot options."""
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of GRACE data vs WMB method."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all plotting options and run a small sample batch for MC and posterior variance.",
    )
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot head-to-head analytical PDFs of regional averages (Bayesian vs WMB).",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps of true loads, posterior expectations, spatial error residuals, and regional averaging masks.",
    )
    parser.add_argument(
        "--plot-corner",
        action="store_true",
        help="Plot a corner plot demonstrating Bayesian recovery of degree-1 coefficients.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=0,
        help="Number of samples to draw from the posterior to estimate pointwise standard deviation.",
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
        help="Maximum spherical harmonic degree for the Earth model.",
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
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Scale (in km) for spatial smoothing. Defaults to --load-scale-km.",
    )

    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=250.0,
        help="Correlation length scale (in km) for the prior measure.",
    )
    parser.add_argument(
        "--direct-std-m",
        type=float,
        default=0.01,
        help="Pointwise standard deviation (in m EWT) for the prior.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Factor scaling the noise correlation length relative to the prior.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.1,
        help="Factor scaling the noise standard deviation relative to the prior.",
    )
    parser.add_argument(
        "--remove-degree-1",
        action="store_true",
        help="Remove degree 1 components from the prior measure.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.smoothing_scale_km is None:
        args.smoothing_scale_km = args.load_scale_km

    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_corner = True
        if args.posterior_samples == 0:
            args.posterior_samples = 10
        if args.mc_trials == 0:
            args.mc_trials = 500

    print("Initializing models and operators...")
    fp, load_space, response_space, fp_op, scale_mm = utils.build_physics_components(
        args.lmax, args.load_order, args.load_scale_km
    )

    init_prior, cond_prior, noise = utils.build_measures(
        fp,
        load_space,
        args.direct_scale_km,
        args.direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
        remove_degree_1=args.remove_degree_1,
    )

    wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
    data_error_measure = wmb.load_measure_to_observation_measure(noise)
    forward_op = sl.grace_operator(response_space, args.obs_degree) @ fp_op

    forward_problem = inf.LinearForwardProblem(
        forward_op, data_error_measure=data_error_measure
    )
    true_direct_load, synthetic_grace_data = forward_problem.synthetic_model_and_data(
        cond_prior
    )

    inverse_problem = inf.LinearBayesianInversion(forward_problem, cond_prior)
    preconditioner = wmb.bayesian_normal_operator_preconditioner(
        init_prior, data_error_measure
    )
    solver = inf.CGMatrixSolver()

    print("Solving for posterior...")
    load_posterior = inverse_problem.model_posterior_measure(
        synthetic_grace_data, solver, preconditioner=preconditioner
    )
    print(f"Solution in {solver.iterations} iterations")

    tot_op = utils.build_total_load_operator(fp, response_space, load_space, fp_op)
    region_names, avg_op, weighting_functions = utils.get_regional_averaging(
        fp, load_space, args.smoothing_scale_km
    )

    tot_avg_op = avg_op @ tot_op
    wmb_avg_op = wmb.potential_coefficient_to_load_operator(load_space)
    wmb_direct_avg_op = avg_op @ wmb_avg_op

    if args.plot_pdfs or args.mc_trials > 0:
        print("Forming load average estimates")
        post_avg_measure = load_posterior.affine_mapping(operator=tot_avg_op)
        post_stds_mm = (
            np.sqrt(np.diag(post_avg_measure.covariance.matrix(dense=True))) * scale_mm
        )

        wmb_noise_measure = data_error_measure.affine_mapping(
            operator=wmb_direct_avg_op
        )
        wmb_stds_mm = (
            np.sqrt(np.diag(wmb_noise_measure.covariance.matrix(dense=True))) * scale_mm
        )

    # ------------------ OPTION 1: Maps ------------------
    if args.plot_maps:
        print("Generating spatial maps...")

        true_dir_mm = true_direct_load * scale_mm
        post_dir_mm = load_posterior.expectation * scale_mm

        vmax_dir = max(
            np.max(np.abs(true_dir_mm.data)), np.max(np.abs(post_dir_mm.data))
        )

        fig_maps, axes_maps = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        sl.plot(
            true_dir_mm,
            ax=axes_maps[0],
            colorbar=True,
            colorbar_kwargs={"label": "EWT (mm)"},
            vmin=-vmax_dir,
            vmax=vmax_dir,
            symmetric=True,
        )
        axes_maps[0].set_title("True Direct Load")

        sl.plot(
            post_dir_mm,
            ax=axes_maps[1],
            colorbar=True,
            colorbar_kwargs={"label": "EWT (mm)"},
            vmin=-vmax_dir,
            vmax=vmax_dir,
            symmetric=True,
        )
        axes_maps[1].set_title("Posterior Expectation (Direct Load)")

    # ------------------ OPTION 2: PDFs ------------------
    if args.plot_pdfs:
        print("Plotting Head-to-Head PDFs...")

        error_map_mm = (
            load_posterior.expectation * scale_mm - true_direct_load * scale_mm
        )

        results = {
            "Bayesian": {
                "means": post_avg_measure.expectation * scale_mm,
                "stds": post_stds_mm,
            },
            "WMB": {
                "means": wmb_direct_avg_op(synthetic_grace_data) * scale_mm,
                "stds": wmb_stds_mm,
            },
        }

        utils.plot_regional_pdfs(
            results,
            region_names,
            tot_avg_op(true_direct_load) * scale_mm,
            error_map_mm=error_map_mm,
        )

    # ------------------ OPTION 3: Corner Plot ------------------
    if args.plot_corner:
        print("Generating Degree-1 Corner Plot...")
        deg1_op = load_space.to_coefficient_operator(1, lmin=1) * scale_mm @ tot_op
        inf.plot_corner_distributions(
            load_posterior.affine_mapping(operator=deg1_op),
            prior_measure=cond_prior.affine_mapping(operator=deg1_op),
            true_values=deg1_op(true_direct_load),
            labels=[
                r"$\zeta_{1-1}$ (mm)",
                r"$\zeta_{10}$ (mm)",
                r"$\zeta_{11}$ (mm)",
            ],
        )

    # ------------------ OPTION 4: Samples ------------------
    if args.posterior_samples > 0:
        pointwise_variance = load_posterior.sample_pointwise_variance(
            args.posterior_samples
        )
        pointwise_std = pointwise_variance.copy()
        pointwise_std.data[:, :] = np.sqrt(pointwise_variance.data[:, :])
        pointwise_std_mm = pointwise_std * scale_mm

        fig_samples, ax_samples = plt.subplots(
            figsize=(14, 5),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        _, im_samples = sl.plot(
            pointwise_std_mm,
            ax=ax_samples,
            colorbar=True,
            colorbar_kwargs={"label": "Std Dev EWT (mm)"},
            vmin=0,
            cmap="viridis",
            symmetric=False,
        )

    # ------------------ OPTION 5: Monte Carlo ------------------
    if args.mc_trials > 0:
        print(f"Running {args.mc_trials} MC trials via dense joint measure mapping...")
        w_errs, b_errs = np.zeros((args.mc_trials, len(region_names))), np.zeros(
            (args.mc_trials, len(region_names))
        )

        post_exp_op = inverse_problem.posterior_expectation_operator(
            solver, preconditioner=preconditioner
        )

        if isinstance(post_exp_op, inf.AffineOperator):
            bayes_linear = post_exp_op.linear_part
            bayes_translation = tot_avg_op(post_exp_op.translation_part)
        else:
            bayes_linear = post_exp_op
            bayes_translation = None

        wmb_err_op = inf.RowLinearOperator([-1 * tot_avg_op, wmb_direct_avg_op])
        bayes_err_op = inf.RowLinearOperator(
            [-1 * tot_avg_op, tot_avg_op @ bayes_linear]
        )

        joint_err_op = inf.ColumnLinearOperator([wmb_err_op, bayes_err_op])

        joint_meas = inverse_problem.joint_prior_measure

        translation = (
            [avg_op.codomain.zero, bayes_translation]
            if bayes_translation is not None
            else None
        )

        joint_err_meas = joint_meas.affine_mapping(
            operator=joint_err_op, translation=translation
        )

        print("Constructing 6x6 dense error covariance...")
        joint_err_dense = joint_err_meas.with_dense_covariance()
        samples = joint_err_dense.samples(args.mc_trials)

        for i, (w_err, b_err) in enumerate(samples):
            w_errs[i, :] = (w_err * scale_mm) / wmb_stds_mm
            b_errs[i, :] = (b_err * scale_mm) / post_stds_mm

        raw_cov_6x6 = joint_err_dense.covariance.matrix(dense=True) * (scale_mm**2)
        if joint_err_dense.has_zero_expectation:
            raw_mean_w = np.zeros(len(region_names))
            raw_mean_b = np.zeros(len(region_names))
        else:
            raw_mean_w = joint_err_dense.expectation[0] * scale_mm
            raw_mean_b = joint_err_dense.expectation[1] * scale_mm

        max_err = max(np.max(np.abs(w_errs)), np.max(np.abs(b_errs)))
        plot_limit = np.ceil(max_err) + 0.5

        fig_mc, axes_mc = plt.subplots(
            nrows=1,
            ncols=len(region_names),
            figsize=(15, 5),
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        for j, region in enumerate(region_names):
            ax = axes_mc.flatten()[j]

            ax.scatter(
                w_errs[:, j],
                b_errs[:, j],
                alpha=0.6,
                color="purple",
                edgecolor="white",
                s=20,
                zorder=3,
            )

            # --- Analytical 2D PDF Contours ---
            mu_2d = np.array(
                [raw_mean_w[j] / wmb_stds_mm[j], raw_mean_b[j] / post_stds_mm[j]]
            )

            var_w = raw_cov_6x6[j, j] / (wmb_stds_mm[j] ** 2)
            var_b = raw_cov_6x6[j + 3, j + 3] / (post_stds_mm[j] ** 2)
            cov_wb = raw_cov_6x6[j, j + 3] / (wmb_stds_mm[j] * post_stds_mm[j])
            cov_2d = np.array([[var_w, cov_wb], [cov_wb, var_b]])

            x_grid, y_grid = np.mgrid[
                -plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j
            ]
            pos = np.dstack((x_grid, y_grid))
            rv = stats.multivariate_normal(mu_2d, cov_2d)
            Z = rv.pdf(pos)

            max_density = rv.pdf(mu_2d)
            levels = [max_density * np.exp(-0.5 * k**2) for k in [3, 2, 1]]
            ax.contour(
                x_grid,
                y_grid,
                Z,
                levels=levels,
                colors="indigo",
                linewidths=[0.5, 1.0, 1.5],
                alpha=0.8,
                zorder=4,
            )

            ax.axhline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
            ax.axvline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
            ax.axhspan(
                -1,
                1,
                color="blue",
                alpha=0.15,
                zorder=0,
                label=r"Bayes 1$\sigma$ Expected",
            )
            ax.axhspan(-2, 2, color="blue", alpha=0.05, zorder=0)
            ax.axvspan(
                -1,
                1,
                color="red",
                alpha=0.15,
                zorder=0,
                label=r"WMB 1$\sigma$ Expected",
            )
            ax.axvspan(-2, 2, color="red", alpha=0.05, zorder=0)

            ax.set_xlim(-plot_limit, plot_limit)
            ax.set_ylim(-plot_limit, plot_limit)
            ax.set_aspect("equal")

            ax.set_title(region, fontsize=14)
            ax.set_xlabel(r"WMB Normalized Error", fontsize=11)
            ax.grid(True, linestyle=":", alpha=0.4)

            if j == 0:
                ax.set_ylabel(r"Bayes Normalized Error", fontsize=11)
                ax.plot(
                    [], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF"
                )
                ax.legend(loc="upper left", fontsize=9)

        fig_mc.suptitle(
            "Monte Carlo Validation: Distribution of Normalized Residuals",
            fontsize=18,
            fontweight="bold",
        )

    if any(
        [
            args.plot_maps,
            args.plot_pdfs,
            args.plot_corner,
            args.posterior_samples,
            args.mc_trials,
        ]
    ):
        plt.show()


if __name__ == "__main__":
    main()
