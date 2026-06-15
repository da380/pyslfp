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
import pyslfp as sl
from pygeoinf.symmetric_space.sphere import plot_points


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
        "--prior-sensitivity", action="store_true", help="Estimate prior sensitivity"
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
        args.plot_maps = args.plot_pdfs = args.plot_deg1 = args.prior_sensitivity = True

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

    # Set up the averaging operators
    tot_avg_op = ewt_mm_scale * avg_op @ total_load_op
    wmb_op = wmb_method.potential_coefficient_to_load_operator(load_space)
    wmb_avg_op = ewt_mm_scale * avg_op @ wmb_op

    metrics_file = os.path.join(output_dir, "grace_metrics.txt")

    if args.plot_pdfs or args.plot_deg1:
        print("Forming load average estimates")

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

            with open(metrics_file, "w") as f_metrics:
                f_metrics.write(
                    f"{'Region':<16} | {'KL Divergence':<15} | {'Prior Var (mm²)':<17} | {'Post Var (mm²)':<16} | {'Variance Reduction':<18}\n"
                )
                f_metrics.write("-" * 92 + "\n")

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
                    coordinate_projection = post_avg_measure.domain.subspace_projection(
                        i
                    )

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

                    basis_vec = coord_prior.domain.basis_vector(0)
                    prior_var = coord_prior.directional_variance(basis_vec)
                    post_var = coord_post.directional_variance(basis_vec)

                    var_reduction = 100.0 * (1.0 - (post_var / prior_var))

                    f_metrics.write(
                        f"{region:<16} | {kl_div:<15.2f} | {prior_var:<17.2f} | {post_var:<16.2f} | {var_reduction:>16.2f}%\n"
                    )

                    inf.plot_1d_distributions(
                        [coord_post, coord_wmb],
                        ax=ax,
                        true_value=true_avg[i],
                        xlabel="Regional Average Total Load (mm EWT)",
                        title=f"{region}",
                        posterior_labels=["Bayesian", "WMB"],
                    )

                    if ax.get_legend() is not None:
                        ax.get_legend().remove()

                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                figures_to_save["grace_regional_pdfs"] = plt.gcf()
                print(f"  Regional metrics logged to: {metrics_file}")

        # ------------------ DEGREE ONE ------------------
        if args.plot_deg1:
            print("Generating Degree-1 Corner Plot...")
            deg1_op = load_space.to_coefficient_operator(1, lmin=1) * ewt_mm_scale

            deg1_prior = model_prior.affine_mapping(
                operator=deg1_op
            ).with_dense_covariance(parallel=True, n_jobs=3)

            deg1_post = model_posterior.affine_mapping(
                operator=deg1_op
            ).with_dense_covariance(parallel=True, n_jobs=3)

            kl_div = deg1_post.kl_divergence(deg1_prior)

            # --- Extract Dense Covariance Matrices ---
            prior_cov_mat = deg1_prior.covariance.matrix(dense=True)
            post_cov_mat = deg1_post.covariance.matrix(dense=True)

            prior_trace = np.trace(prior_cov_mat)
            post_trace = np.trace(post_cov_mat)
            trace_reduction = 100.0 * (1.0 - (post_trace / prior_trace))

            prior_det = np.linalg.det(prior_cov_mat)
            post_det = np.linalg.det(post_cov_mat)

            with open(metrics_file, "a") as f_metrics:
                f_metrics.write("\n\nDegree-1 Spherical Harmonic Coefficients\n")
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

                comp_names = ["Zeta(1,-1)", "Zeta(1,0)", "Zeta(1,1)"]
                for i, name in enumerate(comp_names):
                    pr_v = prior_cov_mat[i, i]
                    po_v = post_cov_mat[i, i]
                    red = 100.0 * (1.0 - (po_v / pr_v)) if pr_v > 0 else 0.0
                    f_metrics.write(
                        f"{name:<18} | {pr_v:<12.4f} | {po_v:<12.4f} | {red:>8.2f}%\n"
                    )
                f_metrics.write("-" * 65 + "\n")

            inf.plot_corner_distributions(
                deg1_post,
                # prior_measure=deg1_prior,
                true_values=deg1_op(model),
                labels=[
                    r"$\zeta_{1-1}$ (mm)",
                    r"$\zeta_{10}$ (mm)",
                    r"$\zeta_{11}$ (mm)",
                ],
                title="",
            )
            figures_to_save["grace_degree_1_corner"] = plt.gcf()

    # ------------------- Prior sensitivity analysis ---------------
    if args.prior_sensitivity:
        print("Generating Prior & Estimator Sensitivity Plots and Metrics...")
        metrics_file = os.path.join(output_dir, "grace_metrics.txt")

        # 1. Bayesian Resolution Operators
        kalman_operator = inverse_problem.kalman_operator(
            solver, preconditioner=preconditioner
        )
        exact_fwd_op = inverse_problem.forward_problem.forward_operator

        model_resolution_operator = (
            kalman_operator @ exact_fwd_op
        ) - load_space.identity_operator()

        property_resolution_operator = tot_avg_op @ model_resolution_operator

        # 2. WMB Resolution Operator (CA - B)
        # C = wmb_avg_op, A = exact_fwd_op, B = tot_avg_op
        wmb_property_resolution_operator = (wmb_avg_op @ exact_fwd_op) - tot_avg_op

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write("\n\n" + "=" * 115 + "\n")
            f_metrics.write(
                "REGIONAL ESTIMATOR KERNEL METRICS (PRIOR-FREE COMPARISON)\n"
            )
            f_metrics.write("-" * 115 + "\n")
            f_metrics.write(
                f"{'Region':<16} | {'Target Norm':<15} | {'Bayes Err Norm':<15} | {'Bayes Ratio':<15} | {'WMB Err Norm':<15} | {'WMB Ratio':<15}\n"
            )
            f_metrics.write("-" * 115 + "\n")

            for i, region in enumerate(region_names):
                # e_i (Basis vector)
                basis_vec = property_resolution_operator.codomain.basis_vector(i)

                # Target kernel: b_i = B* e_i
                target_vector = tot_avg_op.adjoint(basis_vec)

                # Bayesian Kernel error: (KA - I)* b_i
                bayes_res_vector = property_resolution_operator.adjoint(basis_vec)

                # WMB Kernel error: (CA - B)* e_i
                wmb_res_vector = wmb_property_resolution_operator.adjoint(basis_vec)

                # Calculate norms
                target_norm = load_space.norm(target_vector)
                bayes_error_norm = load_space.norm(bayes_res_vector)
                wmb_error_norm = load_space.norm(wmb_res_vector)

                bayes_ratio = bayes_error_norm / target_norm if target_norm > 0 else 0.0
                wmb_ratio = wmb_error_norm / target_norm if target_norm > 0 else 0.0

                f_metrics.write(
                    f"{region:<16} | {target_norm:<15.4e} | {bayes_error_norm:<15.4e} | {bayes_ratio:<15.4f} | {wmb_error_norm:<15.4e} | {wmb_ratio:<15.4f}\n"
                )

                # Normalize relative to pointwise max of the target
                max_abs_val = np.max(np.abs(target_vector.data))
                if max_abs_val > 0:
                    target_normed = target_vector / max_abs_val
                    bayes_error_pct = (bayes_res_vector / max_abs_val) * 100
                    wmb_error_pct = (wmb_res_vector / max_abs_val) * 100
                else:
                    target_normed = load_space.zero
                    bayes_error_pct = load_space.zero
                    wmb_error_pct = load_space.zero

                # 1x3 Figure setup
                fig_sens, axes = sl.subplots(
                    1, 3, figsize=(22, 5), gridspec_kw={"wspace": 0.15}
                )

                gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}

                # --- Plot 1: Target Kernel ---
                _, im_target = sl.plot(
                    target_normed,
                    ax=axes[0],
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-1.0,
                    vmax=1.0,
                    colorbar_kwargs={
                        "shrink": 0.8,
                        "pad": 0.05,
                        "orientation": "horizontal",
                    },
                    gridlines_kwargs=gl_kwargs,
                )
                axes[0].set_title("Target Kernel", fontsize=16)
                im_target.colorbar.set_label("Relative Amplitude", fontsize=14)
                utils.draw_region_boundaries(state, axes[0], regions_dict)

                # --- Find shared max for the error plots ---
                vmax_error = max(
                    np.max(np.abs(bayes_error_pct.data)),
                    np.max(np.abs(wmb_error_pct.data)),
                )
                if vmax_error == 0:
                    vmax_error = 1.0

                # --- Plot 2: Bayesian Kernel Error ---
                _, im_bayes = sl.plot(
                    bayes_error_pct,
                    ax=axes[1],
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-vmax_error,
                    vmax=vmax_error,
                    colorbar_kwargs={
                        "shrink": 0.8,
                        "pad": 0.05,
                        "orientation": "horizontal",
                    },
                    gridlines_kwargs=gl_kwargs,
                )
                axes[1].set_title("Bayesian Kernel Error", fontsize=16)
                im_bayes.colorbar.set_label("Relative Error (%)", fontsize=14)
                utils.draw_region_boundaries(state, axes[1], regions_dict)

                # --- Plot 3: WMB Kernel Error ---
                _, im_wmb = sl.plot(
                    wmb_error_pct,
                    ax=axes[2],
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-vmax_error,
                    vmax=vmax_error,
                    colorbar_kwargs={
                        "shrink": 0.8,
                        "pad": 0.05,
                        "orientation": "horizontal",
                    },
                    gridlines_kwargs=gl_kwargs,
                )
                axes[2].set_title("WMB Kernel Error", fontsize=16)
                im_wmb.colorbar.set_label("Relative Error (%)", fontsize=14)
                utils.draw_region_boundaries(state, axes[2], regions_dict)

                # Format filename
                safe_region_name = (
                    region.replace(" ", "_").replace("(", "").replace(")", "")
                )
                figures_to_save[f"estimator_kernels_{safe_region_name}"] = fig_sens

        # ------------------ DEGREE ONE KERNEL ERROR ------------------
        print("  Evaluating Degree-1 Estimator Kernels...")
        deg1_op = load_space.to_coefficient_operator(1, lmin=1) * ewt_mm_scale
        deg1_resolution_operator = deg1_op @ model_resolution_operator

        deg1_names = ["Zeta(1,-1)", "Zeta(1,0)", "Zeta(1,1)"]

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write("\n\n" + "=" * 80 + "\n")
            f_metrics.write("DEGREE-1 ESTIMATOR KERNEL METRICS (BAYESIAN ONLY)\n")
            f_metrics.write("-" * 80 + "\n")
            f_metrics.write(
                f"{'Component':<16} | {'Target Norm':<15} | {'Bayes Err Norm':<15} | {'Bayes Ratio':<15}\n"
            )
            f_metrics.write("-" * 80 + "\n")

            for i, comp_name in enumerate(deg1_names):
                # e_i (Basis vector)
                basis_vec = deg1_resolution_operator.codomain.basis_vector(i)

                # Extract Target and Bayes Error spatial vectors
                target_vector = deg1_op.adjoint(basis_vec)
                bayes_res_vector = deg1_resolution_operator.adjoint(basis_vec)

                target_norm = load_space.norm(target_vector)
                bayes_error_norm = load_space.norm(bayes_res_vector)

                bayes_ratio = bayes_error_norm / target_norm if target_norm > 0 else 0.0

                f_metrics.write(
                    f"{comp_name:<16} | {target_norm:<15.4e} | {bayes_error_norm:<15.4e} | {bayes_ratio:<15.4f}\n"
                )

                # Normalize relative to pointwise max of the target for plotting
                max_abs_val = np.max(np.abs(target_vector.data))
                if max_abs_val > 0:
                    target_normed = target_vector / max_abs_val
                    bayes_error_pct = (bayes_res_vector / max_abs_val) * 100
                else:
                    target_normed = load_space.zero
                    bayes_error_pct = load_space.zero

                fig_sens, axes = sl.subplots(
                    1, 2, figsize=(15, 5), gridspec_kw={"wspace": 0.15}
                )

                gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}

                # --- Plot 1: Target Kernel (Degree 1 Spherical Harmonic) ---
                _, im_target = sl.plot(
                    target_normed,
                    ax=axes[0],
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-1.0,
                    vmax=1.0,
                    colorbar_kwargs={
                        "shrink": 0.8,
                        "pad": 0.05,
                        "orientation": "horizontal",
                    },
                    gridlines_kwargs=gl_kwargs,
                )
                axes[0].set_title(f"Target Kernel: {comp_name}", fontsize=16)
                im_target.colorbar.set_label("Relative Amplitude", fontsize=14)
                utils.draw_region_boundaries(state, axes[0], regions_dict)

                # --- Find Max Error Bound ---
                vmax_error = np.max(np.abs(bayes_error_pct.data))
                if vmax_error == 0:
                    vmax_error = 1.0

                # --- Plot 2: Bayesian Kernel Error ---
                _, im_bayes = sl.plot(
                    bayes_error_pct,
                    ax=axes[1],
                    cmap="seismic",
                    colorbar=True,
                    symmetric=True,
                    vmin=-vmax_error,
                    vmax=vmax_error,
                    colorbar_kwargs={
                        "shrink": 0.8,
                        "pad": 0.05,
                        "orientation": "horizontal",
                    },
                    gridlines_kwargs=gl_kwargs,
                )
                axes[1].set_title("Bayesian Kernel Error", fontsize=16)
                im_bayes.colorbar.set_label("Relative Error (%)", fontsize=14)
                utils.draw_region_boundaries(state, axes[1], regions_dict)

                # Format filename uniquely for degree 1
                safe_comp_name = (
                    comp_name.replace("(", "").replace(")", "").replace(",", "_")
                )
                figures_to_save[f"estimator_kernels_deg1_{safe_comp_name}"] = fig_sens

        print(f"  Estimator metrics appended to: {metrics_file}")

    # ------------------ MAPS ------------------
    if args.plot_maps:
        print("Generating spatial maps...")
        cmap = "seismic"

        total_load_prior = model_prior.affine_mapping(operator=total_load_op)
        total_load_posterior = model_posterior.affine_mapping(operator=total_load_op)

        true_model = total_load_op(model)
        post_model = total_load_posterior.expectation

        wmb_inv_op = wmb_method.potential_coefficient_to_load_operator(load_space)
        wmb_estimate = wmb_inv_op(data)

        smoothing_operator = load_space.heat_kernel_gaussian_measure(
            2 * noise_scale
        ).covariance
        smoothed_wmb_estimate = smoothing_operator(wmb_estimate)

        vmax = max(
            np.max(np.abs(true_model.data * ewt_mm_scale)),
            np.max(np.abs(post_model.data * ewt_mm_scale)),
            np.max(np.abs(wmb_estimate.data * ewt_mm_scale)),
        )

        fig_maps, axes = sl.subplots(
            2, 2, figsize=(18, 10), gridspec_kw={"hspace": 0.10}
        )

        gl_kwargs = {
            "xlabel_style": {"size": 12},
            "ylabel_style": {"size": 12},
        }

        # 1. True total load
        _, im = sl.plot(
            true_model * ewt_mm_scale,
            ax=axes[0, 0],
            colorbar=False,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            gridlines_kwargs=gl_kwargs,
        )
        axes[0, 0].set_title("True Total Load", fontsize=16)

        # 2. WMB Estimate
        sl.plot(
            wmb_estimate * ewt_mm_scale,
            ax=axes[0, 1],
            colorbar=False,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            gridlines_kwargs=gl_kwargs,
        )
        axes[0, 1].set_title("WMB Estimate", fontsize=16)

        # 3. Smoothed WMB Estimate
        sl.plot(
            smoothed_wmb_estimate * ewt_mm_scale,
            ax=axes[1, 0],
            colorbar=False,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            gridlines_kwargs=gl_kwargs,
        )
        axes[1, 0].set_title("Smoothed WMB Estimate", fontsize=16)

        # 4. Bayesian Posterior
        sl.plot(
            post_model * ewt_mm_scale,
            ax=axes[1, 1],
            colorbar=False,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            gridlines_kwargs=gl_kwargs,
        )
        axes[1, 1].set_title("Bayesian Posterior", fontsize=16)

        for ax in axes.flatten():
            utils.draw_region_boundaries(state, ax, regions_dict)

        cb = fig_maps.colorbar(
            im, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.6, pad=0.05
        )
        cb.set_label("Load (mm EWT)", fontsize=16)
        cb.ax.tick_params(labelsize=14)

        figures_to_save["grace_posterior_maps"] = fig_maps

        # ==============================================================
        # 3x2 Covariance Maps
        # ==============================================================

        # Define three distinct, deterministic points representing different geophysical regimes
        point_ocean = (30.0, -45.0)  # Open Ocean (North Atlantic)
        point_hydro = (-5.0, -60.0)  # Hydrology (Amazon Basin)
        point_polar = (-75.0, -110.0)  # Ice Mass (West Antarctica)

        points_to_plot = [
            (point_ocean, "Ocean"),
            (point_hydro, "Hydrology (Amazon)"),
            (point_polar, "Ice (WAIS)"),
        ]

        fig_maps, axes = sl.subplots(
            3, 2, figsize=(16, 15), gridspec_kw={"hspace": 0.1}
        )

        def plot_covariance_map(cov_data, ax, pt, title):
            """Helper function to keep plotting code clean and DRY."""
            _, im_cov = sl.plot(
                cov_data * ewt_mm_scale**2,
                ax=ax,
                colorbar=True,
                cmap="seismic",
                symmetric=True,
                colorbar_kwargs={
                    "shrink": 0.8,
                    "pad": 0.05,
                    "orientation": "horizontal",
                },
                gridlines_kwargs=gl_kwargs,
            )
            ax.set_title(title, fontsize=16)

            plot_points([pt], ax=ax, color="black", gridlines=False)

            im_cov.colorbar.set_label("Covariance (mm² EWT)", fontsize=14)
            im_cov.colorbar.ax.tick_params(labelsize=12)

        for row_idx, (pt, label) in enumerate(points_to_plot):
            prior_cov = total_load_prior.two_point_covariance(pt)
            post_cov = total_load_posterior.two_point_covariance(pt)

            plot_covariance_map(
                prior_cov, axes[row_idx, 0], pt, f"Prior Covariance: {label}"
            )
            plot_covariance_map(
                post_cov, axes[row_idx, 1], pt, f"Posterior Covariance: {label}"
            )

        figures_to_save["grace_covariance_maps"] = fig_maps

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
