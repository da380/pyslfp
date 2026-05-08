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


import pygeoinf as inf
import grace_utils as utils
from plot_utils import plot_normalized_mc_errors
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
        "--lmax", type=int, default=256, help="Exact model max SH degree."
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
        if args.mc_trials <= 0:
            args.mc_trials = -1

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

    if args.plot_pdfs or args.mc_trials != 0 or args.plot_deg1:
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

        prior_avg_measure = model_prior.affine_mapping(
            operator=tot_avg_op
        ).with_dense_covariance(parallel=True, n_jobs=4)

        true_avg = tot_avg_op(model)

        # ------------------ REGIONAL ANALYSIS ------------------
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

            for i, region in enumerate(region_names):
                ax = axes_flat[i]
                coordinate_projection = post_avg_measure.domain.subspace_projection(i)

                coord_prior = prior_avg_measure.affine_mapping(
                    operator=coordinate_projection
                )
                coord_post = post_avg_measure.affine_mapping(
                    operator=coordinate_projection
                )

                kl_div = coord_post.kl_divergence(coord_prior)

                coord_wmb = wmb_avg_measure.affine_mapping(
                    operator=coordinate_projection
                )

                inf.plot_1d_distributions(
                    [coord_post, coord_wmb],
                    ax=ax,
                    true_value=true_avg[i],
                    xlabel="Regional Average Mass (mm EWT)",
                    title=f"{region}",
                    posterior_labels=[
                        f"Bayesian ({kl_div:.2f} nats)",
                        "WMB",
                    ],
                )

            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            figures_to_save["grace_regional_pdfs"] = plt.gcf()

        # ------------------ DEGREE ONE ------------------
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
                title=f"Bayesian Degree 1 Recovery ({kl_div:.2f} nats)",
            )
            figures_to_save["grace_degree_1_corner"] = plt.gcf()

        # ------------------ OPTION 4: Monte Carlo ------------------
        if args.mc_trials != 0:
            print(
                f"\nExtracting analytical distributions for MC validation (trials={'skipped' if args.mc_trials == -1 else args.mc_trials})..."
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

            wmb_err_op = inf.RowLinearOperator([-1 * tot_avg_op, wmb_avg_op])
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

            # 1. Map to dense covariance
            joint_err_dense = joint_meas.affine_mapping(
                operator=joint_err_op, translation=translation
            ).with_dense_covariance(parallel=True, n_jobs=8)

            # Conditional Sampling
            if args.mc_trials > 0:
                w_errs = np.zeros((args.mc_trials, len(region_names)))
                b_errs = np.zeros((args.mc_trials, len(region_names)))

                samples = joint_err_dense.samples(args.mc_trials)
                for i, (w_err, b_err) in enumerate(samples):
                    w_errs[i, :] = w_err
                    b_errs[i, :] = b_err
            else:
                w_errs, b_errs = None, None

            # 2. Extract standard deviations
            wmb_avg_stds = np.sqrt(wmb_avg_measure.covariance.extract_diagonal())
            post_avg_stds = np.sqrt(post_avg_measure.covariance.extract_diagonal())

            # 3. Extract the full analytical mean and covariance
            n_reg = len(region_names)
            raw_cov_full = joint_err_dense.covariance.matrix(dense=True)
            raw_mean_full = joint_err_dense.expectation

            # Setup the subplots
            ncols = int(np.ceil(np.sqrt(n_reg)))
            nrows = int(np.ceil(n_reg / ncols))

            fig_mc, axes_mc = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(6 * ncols, 6 * nrows),
                layout="constrained",
            )

            axes_flat = np.atleast_1d(axes_mc).flatten()

            # 4. Loop through regions and plot
            for j, region in enumerate(region_names):
                ax = axes_flat[j]

                # Extract the 2x2 mean and covariance block for this region
                raw_mean_2d = [raw_mean_full[0][j], raw_mean_full[1][j]]
                var_w = raw_cov_full[j, j]
                var_b = raw_cov_full[n_reg + j, n_reg + j]
                cov_wb = raw_cov_full[j, n_reg + j]
                raw_cov_2d = np.array([[var_w, cov_wb], [cov_wb, var_b]])

                # Conditionally extract raw errors if sampling was performed
                raw_errs_w = w_errs[:, j] if w_errs is not None else None
                raw_errs_b = b_errs[:, j] if b_errs is not None else None

                plot_normalized_mc_errors(
                    ax,
                    raw_errs_w,
                    raw_errs_b,
                    raw_mean_2d,
                    raw_cov_2d,
                    wmb_avg_stds[j],
                    post_avg_stds[j],
                    title=f"{region} Error Distribution",
                    xlabel=r"WMB Normalized Error",
                    ylabel=r"Bayesian Normalized Error",
                    label_x="WMB",
                    label_y="Bayes",
                    show_legend=(j == 0),
                    show_samples=(args.mc_trials > 0),  # Pass the flag down
                )

            # Hide any empty subplots
            for k in range(j + 1, len(axes_flat)):
                axes_flat[k].set_visible(False)

            figures_to_save["grace_mc_validation"] = fig_mc

    # ------------------ MAPS ------------------
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
        axes[0, 1].set_title("WMB Estimate")

        sl.plot(
            smoothed_wmb_estimate * ewt_mm_scale,
            ax=axes[1, 0],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[1, 0].set_title("Smoothed WMB Estimate")

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
