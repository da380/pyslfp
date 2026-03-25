"""
Bayesian Inversion vs. WMB Method (Comprehensive Analysis)
==========================================================

This script performs a Bayesian inversion of synthetic GRACE gravimetry data to
estimate regional surface mass changes and compares the results head-to-head
with the standard WMB method.

It evaluates the efficacy of the full-physics Bayesian approach using toggleable
visualizations:
1. Spatial Maps (--plot-maps): True vs Posterior fields and spatial residuals.
2. Regional PDFs (--plot-pdfs): Analytical error distributions for specific regions.
3. Degree-1 Recovery (--plot-corner): Joint posterior of degree-1 coefficients.
4. Pointwise Uncertainty (--posterior-samples N): Spatial standard deviation from N posterior samples.
5. Monte Carlo Validation (--mc-trials N): Frequentist validation of the Bayesian estimator.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    """Parses command-line arguments to toggle simulation and plot options."""
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of GRACE data vs WMB method."
    )

    # --- Plotting Options (All default to False/0) ---
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all plotting options (--plot-pdfs, --plot-maps, --plot-corner) and run a small sample batch for MC and posterior variance.",
    )
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot head-to-head analytical PDFs of regional averages (Bayesian vs WMB).",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps of true loads, posterior expectations, and spatial error residuals.",
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
        help="Number of samples to draw from the posterior to estimate and plot the pointwise standard deviation (default: 0).",
    )
    parser.add_argument(
        "--mc-trials",
        type=int,
        default=0,
        help="Number of Monte Carlo trials to run for statistical comparison of estimators (default: 0).",
    )

    # --- Resolution Parameters ---
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

    # --- Load Space Parameters ---
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

    # --- Prior & Noise Parameters ---
    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=250.0,
        help="Correlation length scale (in km) for the direct load prior measure.",
    )
    parser.add_argument(
        "--direct-std-m",
        type=float,
        default=0.01,
        help="Pointwise standard deviation (in m EWT) for the direct load prior.",
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.all:
        args.plot_pdfs = True
        args.plot_maps = True
        args.plot_corner = True
        if args.posterior_samples == 0:
            args.posterior_samples = 10
        if args.mc_trials == 0:
            args.mc_trials = 20  # Reasonable default for --all without taking forever

    # =========================================================================
    # 1. Earth Model and Physical Operators
    # =========================================================================
    print("Initializing Earth model and physical spaces...")
    fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    load_to_water_thickness_mm = 1000 * fp.length_scale / fp.water_density
    load_space_scale = args.load_scale_km * 1000 / fp.length_scale

    finger_print_operator = fp.as_sobolev_linear_operator(
        args.load_order, load_space_scale
    )
    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    # =========================================================================
    # 2. Prior and Data Error Measures
    # =========================================================================
    print("Setting up prior and noise measures...")
    direct_load_measure_scale = args.direct_scale_km * 1000 / fp.length_scale
    direct_load_measure_std = fp.water_density * args.direct_std_m / fp.length_scale

    initial_direct_load_prior = (
        load_space.point_value_scaled_heat_kernel_gaussian_measure(
            direct_load_measure_scale, std=direct_load_measure_std
        )
    )

    constraint_operator = load_space.to_coefficient_operator(0)
    constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
    direct_load_prior = constraint_subspace.condition_gaussian_measure(
        initial_direct_load_prior
    )

    noise_load_measure_scale = args.noise_scale_factor * direct_load_measure_scale
    noise_load_measure_std = args.noise_std_factor * direct_load_measure_std
    noise_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        noise_load_measure_scale, std=noise_load_measure_std
    )

    wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
    data_error_measure = wmb.load_measure_to_observation_measure(noise_load_measure)

    # =========================================================================
    # 3. Full-Physics Forward Problem Setup
    # =========================================================================
    print("Constructing full-physics forward operator...")
    grace_operator = sl.grace_operator(response_space, args.obs_degree)
    forward_operator = grace_operator @ finger_print_operator

    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    true_direct_load, synthetic_grace_data = forward_problem.synthetic_model_and_data(
        direct_load_prior
    )

    # =========================================================================
    # 4. Initial Bayesian Inversion (Reference State)
    # =========================================================================
    print("Performing baseline Bayesian inversion...")
    inverse_problem = inf.LinearBayesianInversion(forward_problem, direct_load_prior)
    preconditioner = wmb.bayesian_normal_operator_preconditioner(
        initial_direct_load_prior, data_error_measure
    )
    solver = inf.CGMatrixSolver()

    load_posterior = inverse_problem.model_posterior_measure(
        synthetic_grace_data, solver, preconditioner=preconditioner
    )

    # =========================================================================
    # 5. Build Total Load Post-Processing Operator
    # =========================================================================
    sea_level_projection = response_space.subspace_projection(0)
    sea_level_to_load = sl.sea_level_change_to_load_operator(
        fp, sea_level_projection.codomain, load_space
    )
    induced_load_operator = (
        sea_level_to_load @ sea_level_projection @ finger_print_operator
    )
    total_load_operator = load_space.identity_operator() + induced_load_operator

    # =========================================================================
    # 6. Common Regional Averaging Operators & Uncertainties
    # =========================================================================
    sle_factor = -1.0 / (fp.water_density * fp.ocean_area)
    selected_regions = [
        "Greenland/Iceland",
        "W.Antarctica",
        "S.Indic-Ocean",
        "South-American-Monsoon",
    ]
    target_regions = {
        region: fp.regionmask_projection(region, value=0) * sle_factor
        for region in selected_regions
    }

    region_names = list(target_regions.keys())
    weighting_functions = list(target_regions.values())
    averaging_operator = sl.averaging_operator(load_space, weighting_functions)

    total_averaging_operator = averaging_operator @ total_load_operator
    wmb_average_operator = wmb.potential_coefficient_to_load_average_operator(
        load_space, weighting_functions
    )

    # Extract strictly analytical covariances (independent of the synthetic data!)
    posterior_averages_measure = load_posterior.affine_mapping(
        operator=total_averaging_operator
    )
    post_stds_mm = (
        np.sqrt(np.diag(posterior_averages_measure.covariance.matrix(dense=True)))
        * fp.length_scale
        * 1000
    )

    wmb_noise_measure = data_error_measure.affine_mapping(operator=wmb_average_operator)
    wmb_stds_mm = (
        np.sqrt(np.diag(wmb_noise_measure.covariance.matrix(dense=True)))
        * fp.length_scale
        * 1000
    )

    # =========================================================================
    # OPTION 1: Spatial Map Visualization
    # =========================================================================
    if args.plot_maps:
        print("Generating spatial maps and residuals...")
        true_total_load = total_load_operator(true_direct_load)
        post_total_load = total_load_operator(load_posterior.expectation)

        scale = load_to_water_thickness_mm
        true_dir_mm = true_direct_load * scale
        post_dir_mm = load_posterior.expectation * scale
        true_tot_mm = true_total_load * scale
        post_tot_mm = post_total_load * scale
        res_tot_mm = true_tot_mm - post_tot_mm

        wmb_spatial_operator = wmb.potential_coefficient_to_load_operator(load_space)
        wmb_spatial_load = wmb_spatial_operator(synthetic_grace_data)
        wmb_spatial_mm = wmb_spatial_load * scale
        res_wmb_mm = true_tot_mm - wmb_spatial_mm

        vmax_dir = max(
            np.max(np.abs(true_dir_mm.data)), np.max(np.abs(post_dir_mm.data))
        )
        fig_d1, ax_d1, im_d1 = sl.plot(
            true_dir_mm,
            colorbar_label="EWT (mm)",
            vmin=-vmax_dir,
            vmax=vmax_dir,
            symmetric=True,
        )
        ax_d1.set_title("True Direct Load")

        fig_d2, ax_d2, im_d2 = sl.plot(
            post_dir_mm,
            colorbar_label="EWT (mm)",
            vmin=-vmax_dir,
            vmax=vmax_dir,
            symmetric=True,
        )
        ax_d2.set_title("Posterior Expectation (Direct Load)")

        vmax_tot = max(
            np.max(np.abs(true_tot_mm.data)), np.max(np.abs(post_tot_mm.data))
        )
        fig_t1, ax_t1, im_t1 = sl.plot(
            true_tot_mm,
            colorbar_label="EWT (mm)",
            vmin=-vmax_tot,
            vmax=vmax_tot,
            symmetric=True,
        )
        ax_t1.set_title("True Total Load (Direct + Induced)")

        fig_t2, ax_t2, im_t2 = sl.plot(
            post_tot_mm,
            colorbar_label="EWT (mm)",
            vmin=-vmax_tot,
            vmax=vmax_tot,
            symmetric=True,
        )
        ax_t2.set_title("Posterior Expectation (Total Load)")

        vmax_res = np.max(np.abs(res_tot_mm.data))
        if res_wmb_mm is not None:
            vmax_res = max(vmax_res, np.max(np.abs(res_wmb_mm.data)))

        fig_r1, ax_r1, im_r1 = sl.plot(
            res_tot_mm,
            colorbar_label="Error (mm)",
            vmin=-vmax_res,
            vmax=vmax_res,
            symmetric=True,
        )
        ax_r1.set_title("Bayesian Estimation Error (True Total - Posterior)")

        if res_wmb_mm is not None:
            fig_r2, ax_r2, im_r2 = sl.plot(
                res_wmb_mm,
                colorbar_label="Error (mm)",
                vmin=-vmax_res,
                vmax=vmax_res,
                symmetric=True,
            )
            ax_r2.set_title("WMB Estimation Error (True Total - WMB)")

    # =========================================================================
    # OPTION 2: Head-to-Head PDFs (Regional Averages)
    # =========================================================================
    if args.plot_pdfs:
        print("Plotting Head-to-Head PDFs...")

        # Calculate specific estimates for the baseline true state
        true_averages_mm = (
            total_averaging_operator(true_direct_load) * fp.length_scale * 1000
        )
        post_means_mm = posterior_averages_measure.expectation * fp.length_scale * 1000
        wmb_estimates_mm = (
            wmb_average_operator(synthetic_grace_data) * fp.length_scale * 1000
        )

        def gaussian_pdf(x, mean, std):
            return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mean) / std) ** 2
            )

        ncols = 2
        nrows = int(np.ceil(len(region_names) / ncols))
        fig_pdf, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), layout="constrained"
        )
        axes_flat = axes.flatten()

        for i, region in enumerate(region_names):
            ax = axes_flat[i]
            post_mean, post_std = post_means_mm[i], post_stds_mm[i]
            wmb_mean, wmb_std = wmb_estimates_mm[i], wmb_stds_mm[i]
            true_val = true_averages_mm[i]

            plot_min = min(
                post_mean - 4 * post_std, wmb_mean - 4 * wmb_std, true_val - post_std
            )
            plot_max = max(
                post_mean + 4 * post_std, wmb_mean + 4 * wmb_std, true_val + post_std
            )
            x_vals = np.linspace(plot_min, plot_max, 400)

            y_vals_post = gaussian_pdf(x_vals, post_mean, post_std)
            ax.plot(
                x_vals,
                y_vals_post,
                "b-",
                linewidth=2,
                label=rf"Bayesian ($\mu$={post_mean:.2f}, $\sigma$={post_std:.3f})",
            )
            ax.fill_between(
                x_vals,
                0,
                y_vals_post,
                where=(
                    (x_vals >= post_mean - 1.96 * post_std)
                    & (x_vals <= post_mean + 1.96 * post_std)
                ),
                color="blue",
                alpha=0.15,
            )

            y_vals_wmb = gaussian_pdf(x_vals, wmb_mean, wmb_std)
            ax.plot(
                x_vals,
                y_vals_wmb,
                "r-",
                linewidth=2,
                label=rf"WMB ($\mu$={wmb_mean:.2f}, $\sigma$={wmb_std:.3f})",
            )
            ax.fill_between(
                x_vals,
                0,
                y_vals_wmb,
                where=(
                    (x_vals >= wmb_mean - 1.96 * wmb_std)
                    & (x_vals <= wmb_mean + 1.96 * wmb_std)
                ),
                color="red",
                alpha=0.15,
            )

            ax.axvline(
                true_val,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"True Value ({true_val:.2f})",
            )
            ax.set_title(f"{region}", fontsize=14)
            ax.set_xlabel("Regional Average Mass (mm EWT)", fontsize=12)
            ax.set_ylabel("Probability Density", fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=10)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle(
            "Bayesian vs. WMB Efficacy: Regional Averages",
            fontsize=18,
            fontweight="bold",
        )

    # =========================================================================
    # OPTION 3: Degree-1 Coefficient Recovery & Corner Plot
    # =========================================================================
    if args.plot_corner:
        print("Evaluating Degree-1 Coefficient recovery...")
        degree_1_extraction_operator = (
            load_space.to_coefficient_operator(1, lmin=1) * load_to_water_thickness_mm
        )
        total_degree_1_operator = degree_1_extraction_operator @ total_load_operator

        true_degree_1 = total_degree_1_operator(true_direct_load)
        posterior_degree_1_measure = load_posterior.affine_mapping(
            operator=total_degree_1_operator
        )
        prior_degree_1_measure = direct_load_prior.affine_mapping(
            operator=total_degree_1_operator
        )

        print("Generating Degree-1 Corner Plot...")
        fig_corner, axes_corner = sl.plot_corner_distributions(
            posterior_degree_1_measure,
            prior_measure=prior_degree_1_measure,
            true_values=true_degree_1,
            labels=[
                r"$C_{1,-1}$ (mm EWT)",
                r"$C_{1,0}$ (mm EWT)",
                r"$C_{1,1}$ (mm EWT)",
            ],
            title="Joint Posterior Distributions of Total Load Degree-1 Coefficients",
            show_plot=False,
        )

    # =========================================================================
    # OPTION 4: Posterior Pointwise Standard Deviation
    # =========================================================================
    if args.posterior_samples > 0:
        print(
            f"Estimating posterior pointwise standard deviation using {args.posterior_samples} samples..."
        )
        pointwise_std = load_posterior.sample_pointwise_std(args.posterior_samples)

        pointwise_std_mm = pointwise_std * load_to_water_thickness_mm

        vmax_std = np.nanmax(np.abs(pointwise_std_mm.data))
        fig_std, ax_std, im_std = sl.plot(
            pointwise_std_mm,
            colorbar_label="Std Dev EWT (mm)",
            vmin=0,
            vmax=vmax_std,
            cmap="viridis",
            symmetric=False,
        )
        ax_std.set_title(
            f"Posterior Pointwise Standard Deviation (N={args.posterior_samples})"
        )

    # =========================================================================
    # OPTION 5: Monte Carlo Statistical Validation (Frequentist vs Bayesian)
    # =========================================================================
    if args.mc_trials > 0:
        print(
            f"\nRunning {args.mc_trials} Monte Carlo trials for statistical comparison..."
        )

        wmb_norm_errors = np.zeros((args.mc_trials, len(region_names)))
        bayes_norm_errors = np.zeros((args.mc_trials, len(region_names)))

        for i in range(args.mc_trials):
            if (i + 1) % max(1, args.mc_trials // 10) == 0 or i == 0:
                print(f"  Solving Trial {i + 1}/{args.mc_trials}...")

            # 1. Draw a new "True" State and "Noisy" Data
            mc_true_load, mc_data = forward_problem.synthetic_model_and_data(
                direct_load_prior
            )

            # 2. Extract Regional Averages for the new True state
            mc_true_avgs = (
                total_averaging_operator(mc_true_load) * fp.length_scale * 1000
            )

            # 3. WMB Estimation
            mc_wmb_est = wmb_average_operator(mc_data) * fp.length_scale * 1000

            # 4. Bayesian Inversion
            mc_post = inverse_problem.model_posterior_measure(
                mc_data, solver, preconditioner=preconditioner
            )
            # The Bayesian estimator is the posterior expectation mapped to regional averages
            mc_bayes_est = (
                total_averaging_operator(mc_post.expectation) * fp.length_scale * 1000
            )

            # 5. Record Normalized Residuals
            wmb_norm_errors[i, :] = (mc_wmb_est - mc_true_avgs) / wmb_stds_mm
            bayes_norm_errors[i, :] = (mc_bayes_est - mc_true_avgs) / post_stds_mm

        # --- Scatter Plot Visualization ---
        print("Generating Monte Carlo Scatter Plots...")
        ncols = 2
        nrows = int(np.ceil(len(region_names) / ncols))
        fig_mc, axes_mc = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(12, 12), layout="constrained"
        )
        axes_mc_flat = axes_mc.flatten()

        for j, region in enumerate(region_names):
            ax = axes_mc_flat[j]

            ax.scatter(
                wmb_norm_errors[:, j],
                bayes_norm_errors[:, j],
                alpha=0.8,
                color="purple",
                edgecolor="white",
                s=50,
                zorder=3,
            )

            # Draw origin reference lines
            ax.axhline(0, color="black", linestyle="-", alpha=0.5, zorder=1)
            ax.axvline(0, color="black", linestyle="-", alpha=0.5, zorder=1)

            # Identify expected target zones using independent crossing strips
            # Bayesian Expected Zones (Horizontal bands)
            ax.axhspan(
                -1,
                1,
                color="blue",
                alpha=0.15,
                zorder=0,
                label=r"Bayes 1$\sigma$ Expected",
            )
            ax.axhspan(
                -2, 2, color="blue", alpha=0.05, zorder=0
            )  # 2-sigma (unlabeled to save legend space)

            # WMB Expected Zones (Vertical bands)
            ax.axvspan(
                -1,
                1,
                color="red",
                alpha=0.15,
                zorder=0,
                label=r"WMB 1$\sigma$ Expected",
            )
            ax.axvspan(-2, 2, color="red", alpha=0.05, zorder=0)  # 2-sigma

            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(region, fontsize=14)
            ax.set_xlabel(
                r"WMB Normalized Error: $(\hat{x}_{wmb} - x_{true}) / \sigma_{wmb}$",
                fontsize=11,
            )
            ax.set_ylabel(
                r"Bayes Normalized Error: $(\hat{x}_{bayes} - x_{true}) / \sigma_{bayes}$",
                fontsize=11,
            )
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(loc="best", fontsize=9)

        for k in range(j + 1, len(axes_mc_flat)):
            axes_mc_flat[k].set_visible(False)

        fig_mc.suptitle(
            "Monte Carlo Validation: Distribution of Normalized Residuals",
            fontsize=18,
            fontweight="bold",
        )

    # =========================================================================
    # Display Outputs
    # =========================================================================
    if any(
        [
            args.plot_maps,
            args.plot_pdfs,
            args.plot_corner,
            args.posterior_samples > 0,
            args.mc_trials > 0,
        ]
    ):
        print("Rendering figures...")
        plt.show()
    else:
        print("Inversion complete. No plotting options were selected.")
        print(
            "Tip: Run with --plot-pdfs, --plot-maps, --plot-corner, --posterior-samples N, or --mc-trials N."
        )


if __name__ == "__main__":
    main()
