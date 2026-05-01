"""
Bayesian GRACE Inversion
========================

This script performs a Bayesian inversion of synthetic GRACE gravimetry data
to estimate the causative surface mass load. It highlights the differences
between the exact elastic Sea Level Equation response and the purely spectral
WMB approximation, particularly focusing on how these differences impact
inversion results and preconditioning.
"""

import argparse
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats


import pygeoinf as inf
import grace_utils as utils
import pyslfp as sl


matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of GRACE data for total mass load."
    )
    # --- Output Options ---
    parser.add_argument(
        "--all", action="store_true", help="Enable all plotting options."
    )
    parser.add_argument("--plot-maps", action="store_true", help="Plot spatial maps.")
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot head-to-head analytical PDFs of regional averages (Bayesian vs WMB).",
    )

    parser.add_argument(
        "--plot-deg1",
        action="store_true",
        help="Plot Degree 1 coefficient corner plots.",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of MC trials for error validation.",
    )

    # --- Resolution & Physics Settings ---
    parser.add_argument(
        "--lmax", type=int, default=128, help="Exact model max SH degree."
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
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Scale (in km) for spatial smoothing. Defaults to --load-scale-km.",
    )

    # --- Prior Settings ---
    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=250.0,
        help="Prior correlation scale (km).",
    )
    parser.add_argument(
        "--direct-std-m", type=float, default=0.01, help="Prior std dev (m EWT)."
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )
    parser.add_argument(
        "--remove-degree-1", action="store_true", help="Remove degree 1 from prior."
    )

    # --- Noise Settings ---
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Noise correlation scale factor.",
    )
    parser.add_argument(
        "--noise-std-factor", type=float, default=0.1, help="Noise std factor."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_maps = args.plot_pdfs = args.plot_deg1 = True
        if args.mc_trials == 0:
            args.mc_trials = 500

    if args.smoothing_scale_km is None:
        args.smoothing_scale_km = args.load_scale_km

    output_dir = "output_plots_grace_inversion"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    inf.configure_threading(n_threads=1)

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT physical operators (lmax={args.lmax})...")
    state, load_space, response_space, fp_op, ewt_mm_scale = (
        utils.build_physics_components(args.lmax, args.load_order, args.load_scale_km)
    )

    initial_prior, model_prior, noise_spatial, noise_scale = utils.build_measures(
        state,
        load_space,
        args.direct_scale_km,
        args.direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
        remove_degree_1=args.remove_degree_1,
        prior_shift=args.prior_shift,
    )

    region_names, avg_op, _, regions_dict = utils.get_regional_averaging(
        state, load_space, smoothing_scale_km=args.smoothing_scale_km
    )

    # Construct the exact forward observation operator
    total_load_op = utils.build_total_load_operator(
        state, response_space, load_space, fp_op
    )
    grace_obs_op = sl.linear_operators.grace_observation_operator(
        response_space, args.obs_degree
    )
    exact_forward_op = grace_obs_op @ fp_op @ total_load_op

    # Build the exact data noise measure using WMB spectral mapping
    wmb_method = sl.linear_operators.WMBMethod(state.model, args.obs_degree)
    data_error_measure = wmb_method.load_measure_to_observation_measure(noise_spatial)

    print("\nDrawing synthetic model and dataset...")
    forward_problem = inf.LinearForwardProblem(
        exact_forward_op, data_error_measure=data_error_measure
    )
    model, data = forward_problem.synthetic_model_and_data(model_prior)
    data_measure = data_error_measure.affine_mapping(translation=data)
    inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

    # ------------------ 2. PRECONDITIONER SETUP ------------------
    preconditioner = wmb_method.bayesian_normal_operator_preconditioner(
        initial_prior, data_error_measure
    )

    # ------------------ 3. POSTERIOR SOLVE ------------------
    print("\nSolving for posterior expectation...")
    callback = inf.ProgressCallback()
    solver = inf.CGSolver(callback=callback, rtol=0.01 * args.noise_std_factor)

    model_posterior = inverse_problem.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    if args.plot_pdfs or args.mc_trials > 0 or args.plot_deg1:
        print("Forming load average estimates")

        tot_avg_op = ewt_mm_scale * avg_op @ total_load_op

        wmb_op = wmb_method.potential_coefficient_to_load_operator(load_space)
        wmb_avg_op = ewt_mm_scale * avg_op @ wmb_op

        wmb_avg_measure = data_measure.affine_mapping(
            operator=wmb_avg_op
        ).with_dense_covariance()

        post_avg_measure = model_posterior.affine_mapping(
            operator=tot_avg_op
        ).with_dense_covariance(parallel=True, n_jobs=4)

        true_avg = tot_avg_op(model)

        # ------------------ 5. REGIONAL ANALYSIS ------------------
        if args.plot_pdfs:
            print("\nDecomposing Regional Signals...")

            ncols = 2
            nrows = int(np.ceil(len(region_names) / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(14, 5 * nrows),
                layout="constrained",
            )
            axes_flat = axes.flatten()

            labels = list(regions_dict.keys())

            for i, region in enumerate(region_names):
                ax = axes_flat[i]
                coordinate_projection = post_avg_measure.domain.subspace_projection(i)

                inf.plot_1d_distributions(
                    [
                        post_avg_measure.affine_mapping(operator=coordinate_projection),
                        wmb_avg_measure.affine_mapping(operator=coordinate_projection),
                    ],
                    ax=ax,
                    true_value=true_avg[i],
                    xlabel="Regional Average Mass (mm EWT)",
                    title=f"{region}",
                    posterior_labels=labels,
                )

            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            figures_to_save["grace_regional_pdfs"] = plt.gcf()

        if args.plot_deg1:
            print("Generating Degree-1 Corner Plot...")
            deg1_op = (
                load_space.to_coefficient_operator(1, lmin=1)
                * ewt_mm_scale
                @ total_load_op
            )

            deg1_prior = model_prior.affine_mapping(
                operator=deg1_op
            ).with_dense_covariance(parallel=True, n_jobs=3)

            deg1_post = model_posterior.affine_mapping(
                operator=deg1_op
            ).with_dense_covariance(parallel=True, n_jobs=3)

            kl_div = deg1_post.kl_divergence(deg1_prior)

            inf.plot_corner_distributions(
                deg1_post,
                prior_measure=deg1_prior,
                true_values=deg1_op(model),
                labels=[
                    r"$\zeta_{1-1}$ (mm)",
                    r"$\zeta_{10}$ (mm)",
                    r"$\zeta_{11}$ (mm)",
                ],
                title=f"Degree 1 Recovery (Information Gained: {kl_div:.2f} nats)",
            )
            figures_to_save["grace_degree_1_corner"] = plt.gcf()

    # ------------------ 4. MAPPING ------------------
    if args.plot_maps:
        print("Generating spatial maps...")
        cmap = "seismic"

        post_model = model_posterior.expectation

        wmb_inv_op = wmb_method.potential_coefficient_to_load_operator(load_space)
        wmb_estimate = wmb_inv_op(data)

        smoothing_operator = load_space.heat_kernel_gaussian_measure(
            2 * noise_scale
        ).covariance
        smoothed_wmb_estimate = smoothing_operator(wmb_estimate)

        vmax = max(
            np.max(np.abs(model.data * ewt_mm_scale)),
            np.max(np.abs(post_model.data * ewt_mm_scale)),
            np.max(np.abs(wmb_estimate.data * ewt_mm_scale)),
        )

        fig_maps, axes = sl.subplots(
            2,
            2,
            figsize=(20, 12),
        )

        sl.plot(
            model * ewt_mm_scale,
            ax=axes[0, 0],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[0, 0].set_title("True direct Load")

        sl.plot(
            wmb_estimate * ewt_mm_scale,
            ax=axes[0, 1],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[0, 1].set_title("WMB Spectral Estimate)")

        sl.plot(
            smoothed_wmb_estimate * ewt_mm_scale,
            ax=axes[1, 0],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[1, 0].set_title("Smoothed WMB Spectral Estimate)")

        sl.plot(
            post_model * ewt_mm_scale,
            ax=axes[1, 1],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[1, 1].set_title("Bayesian Posterior")

        for ax in axes.flatten():
            utils.draw_region_boundaries(state, ax, regions_dict)

        figures_to_save["grace_posterior_maps"] = fig_maps

    # ------------------ 6: Corner Plot ------------------

    # ------------------ OPTION 4: Monte Carlo ------------------
    if args.mc_trials > 0:
        print(f"Running {args.mc_trials} MC trials via dense joint measure mapping...")
        w_errs, b_errs = (
            np.zeros((args.mc_trials, len(region_names))),
            np.zeros((args.mc_trials, len(region_names))),
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

        print("Constructing dense error covariance...")
        joint_err_dense = joint_err_meas.with_dense_covariance(parallel=True, n_jobs=8)
        samples = joint_err_dense.samples(args.mc_trials)

        for i, (w_err, b_err) in enumerate(samples):
            w_errs[i, :] = (w_err * ewt_mm_scale) / wmb_stds_mm
            b_errs[i, :] = (b_err * ewt_mm_scale) / post_stds_mm

        n_reg = len(region_names)
        raw_cov = joint_err_dense.covariance.matrix(dense=True) * (ewt_mm_scale**2)

        if joint_err_dense.has_zero_expectation:
            raw_mean_w = np.zeros(n_reg)
            raw_mean_b = np.zeros(n_reg)
        else:
            raw_mean_w = joint_err_dense.expectation[0] * ewt_mm_scale
            raw_mean_b = joint_err_dense.expectation[1] * ewt_mm_scale

        max_err = max(np.max(np.abs(w_errs)), np.max(np.abs(b_errs)))
        plot_limit = np.ceil(max_err) + 0.5

        ncols = int(np.ceil(np.sqrt(n_reg)))
        nrows = int(np.ceil(n_reg / ncols))

        fig_mc, axes_mc = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5 * ncols, 5 * nrows),
            sharex=True,
            sharey=True,
            layout="constrained",
        )

        axes_flat = np.atleast_1d(axes_mc).flatten()

        for j, region in enumerate(region_names):
            ax = axes_flat[j]

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

            var_w = raw_cov[j, j] / (wmb_stds_mm[j] ** 2)
            var_b = raw_cov[j + n_reg, j + n_reg] / (post_stds_mm[j] ** 2)
            cov_wb = raw_cov[j, j + n_reg] / (wmb_stds_mm[j] * post_stds_mm[j])
            cov_2d = np.array([[var_w, cov_wb], [cov_wb, var_b]])

            x_grid, y_grid = np.mgrid[
                -plot_limit:plot_limit:500j, -plot_limit:plot_limit:500j
            ]
            pos = np.dstack((x_grid, y_grid))
            rv = stats.multivariate_normal(mu_2d, cov_2d)
            Z = rv.pdf(pos)

            max_density = rv.pdf(mu_2d)
            levels = [max_density * np.exp(-0.5 * k**2) for k in [4, 3, 2, 1]]
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
            ax.grid(True, linestyle=":", alpha=0.4)

            if j >= (nrows - 1) * ncols or (j + ncols >= n_reg):
                ax.set_xlabel(r"WMB Normalized Error", fontsize=11)
            if j % ncols == 0:
                ax.set_ylabel(r"Bayes Normalized Error", fontsize=11)

            if j == 0:
                ax.plot(
                    [], [], color="indigo", linewidth=1.5, label="Analytical 2D PDF"
                )
                ax.legend(loc="upper left", fontsize=9)

        for j in range(n_reg, len(axes_flat)):
            axes_flat[j].set_visible(False)

        figures_to_save["grace_mc_validation"] = fig_mc

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
