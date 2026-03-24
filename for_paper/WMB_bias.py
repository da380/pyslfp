"""
WMB Method Bias Evaluation
==========================

This script quantifies the systematic bias introduced by the Wahr, Molenaar, & Bryan (1998)
(WMB) method when estimating regional surface mass changes from satellite gravimetry
(e.g., GRACE / GRACE-FO).

The WMB method isolates surface mass anomalies by applying a purely spectral scaling
(via load Love numbers) to observed gravitational potential coefficients. However, this
approach inherently neglects gravitational self-attraction and loading (SAL) effects—most
notably the induced redistribution of water mass across the global ocean (governed by the
sea-level equation).

Methodology:
------------
This script evaluates the resulting estimation bias within a rigorous Bayesian statistical
framework using infinite-dimensional Gaussian measures:
    1. Defines a spatial prior for the "true" direct surface mass load and a colored
       observational noise model.
    2. Constructs a forward physical block-operator that computes the true total load
       (direct load + induced ocean response) and the resulting truncated satellite observations.
    3. Applies the WMB estimation operator to the simulated observations.
    4. Derives the exact analytical probability density functions (PDFs) of the estimation
       error (True Regional Average - WMB Estimated Average) for predefined IPCC AR6 regions.

Outputs:
--------
- Computes the exact analytical covariances and means of the WMB estimation errors.
- Generates visualizations overlaying the theoretical error PDFs onto Monte Carlo sample
  histograms.
- Evaluates the magnitude of the estimation error relative to the natural standard deviation
  of the region's true mass signal.
- Optionally plots spatial maps demonstrating the physical reality of the induced ocean load.

Usage:
------
Run `python WMB_bias.py --help` to see all available command-line configuration options,
including spherical harmonic resolution limits, noise scaling factors, and Monte Carlo
sampling flags.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import regionmask

import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    """Parses command-line arguments to toggle simulation options."""
    parser = argparse.ArgumentParser(
        description="Calculate and plot WMB method bias for GRACE gravimetry."
    )

    # --- Run Options ---
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples. Set to 0 to only plot analytical PDFs.",
    )
    parser.add_argument(
        "--remove-degree-1",
        action="store_true",
        help="Condition the prior measure to remove degree-1 (center of mass) variations.",
    )
    parser.add_argument(
        "--plot-map",
        action="store_true",
        help="Plot a map showing a spatial sample of the direct load and the induced difference.",
    )
    parser.add_argument(
        "--sample-expectation",
        action="store_true",
        help="Sample a non-zero expectation (mean) for the direct load measure from its own prior.",
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

    # --- Direct Load Measure (Prior) Parameters ---
    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=250.0,
        help="Correlation length scale (in km) for the direct load measure.",
    )
    parser.add_argument(
        "--direct-std-m",
        type=float,
        default=0.01,
        help="Pointwise standard deviation (in meters Equivalent Water Thickness) for the direct load.",
    )

    # --- Noise Measure Parameters ---
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Factor to scale the noise correlation length relative to the direct load scale.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.01,
        help="Factor to scale the noise standard deviation relative to the direct load std.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # =========================================================================
    # 1. Physical Parameters and Earth Model Setup
    # =========================================================================
    fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # Conversion factor from non-dimensional load to millimeters of Equivalent Water Thickness
    load_to_water_thickness_mm = 1000 * fp.length_scale / fp.water_density

    load_space_scale = args.load_scale_km * 1000 / fp.length_scale

    finger_print_operator = fp.as_sobolev_linear_operator(
        args.load_order,
        load_space_scale,
    )
    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    # =========================================================================
    # 2. Define Prior and Noise Gaussian Measures
    # =========================================================================
    direct_load_measure_scale = args.direct_scale_km * 1000 / fp.length_scale
    direct_load_measure_std = fp.water_density * args.direct_std_m / fp.length_scale
    direct_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        direct_load_measure_scale, std=direct_load_measure_std
    )

    lmax_con = 1 if args.remove_degree_1 else 0
    constraint_operator = load_space.to_coefficient_operator(lmax_con)
    constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
    direct_load_measure = constraint_subspace.condition_gaussian_measure(
        direct_load_measure
    )

    if args.sample_expectation:
        print("Sampling a non-zero expectation for the direct load...")
        sampled_mean = direct_load_measure.sample()
        # Translate the measure to center it on the sampled expectation
        direct_load_measure = direct_load_measure.affine_mapping(
            translation=sampled_mean
        )

    noise_load_measure_scale = args.noise_scale_factor * direct_load_measure_scale
    noise_load_measure_std = args.noise_std_factor * direct_load_measure_std
    noise_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        noise_load_measure_scale, std=noise_load_measure_std
    )

    # =========================================================================
    # 3. Regional Averaging Setup (WMB Method)
    # =========================================================================
    wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
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

    # =========================================================================
    # 4. Construct the Forward Model via Block Operators
    # =========================================================================
    data_error_measure = wmb.load_measure_to_observation_measure(noise_load_measure)
    data_space = data_error_measure.domain

    sea_level_projection = response_space.subspace_projection(0)
    sea_level_to_load = sl.sea_level_change_to_load_operator(
        fp, sea_level_projection.codomain, load_space
    )
    grace_operator = sl.grace_operator(response_space, args.obs_degree)
    averaging_operator = sl.averaging_operator(load_space, weighting_functions)
    wmb_average_operator = wmb.potential_coefficient_to_load_average_operator(
        load_space, weighting_functions
    )

    induced_load_operator = (
        sea_level_to_load @ sea_level_projection @ finger_print_operator
    )

    op1 = inf.BlockLinearOperator(
        [
            [load_space.identity_operator(), data_space.zero_operator(load_space)],
            [finger_print_operator, data_space.zero_operator(response_space)],
            [load_space.zero_operator(data_space), data_space.identity_operator()],
        ]
    )
    op2 = inf.BlockLinearOperator(
        [
            [
                load_space.identity_operator(),
                sea_level_to_load @ sea_level_projection,
                data_space.zero_operator(load_space),
            ],
            [
                load_space.zero_operator(data_space),
                grace_operator,
                data_space.identity_operator(),
            ],
        ]
    )

    averages_space = averaging_operator.codomain

    op3_true = inf.RowLinearOperator(
        [averaging_operator, data_space.zero_operator(averages_space)]
    )

    op3_error = inf.RowLinearOperator([averaging_operator, -1 * wmb_average_operator])

    true_averages_operator = op3_true @ op2 @ op1
    error_operator = op3_error @ op2 @ op1

    joint_measure = inf.GaussianMeasure.from_direct_sum(
        [direct_load_measure, data_error_measure]
    )

    true_averages_measure = joint_measure.affine_mapping(
        operator=true_averages_operator
    )
    error_measure = joint_measure.affine_mapping(operator=error_operator)

    # =========================================================================
    # 5. Optional Map Plotting (Physics Check)
    # =========================================================================
    if args.plot_map:
        print("Generating spatial sample maps...")

        total_load_operator = load_space.identity_operator() + induced_load_operator

        sample_direct = direct_load_measure.sample()
        sample_total = total_load_operator(sample_direct)
        sample_diff = sample_total - sample_direct

        ar6_regions = regionmask.defined_regions.ar6.all
        target_ar6 = ar6_regions[selected_regions]

        label_style = {
            "color": "black",
            "fontweight": "bold",
            "fontsize": 12,
            "bbox": dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        }

        fig1, ax1, im1 = sl.plot(
            sample_direct * load_to_water_thickness_mm,
            colorbar_label="Equivalent water thickness (mm)",
            symmetric=True,
        )
        ax1.set_title("Direct load sample")
        target_ar6.plot(
            ax=ax1,
            add_label=True,
            label="abbrev",
            line_kws={"color": "black", "linewidth": 1.5},
            text_kws=label_style,
        )

        fig2, ax2, im2 = sl.plot(
            sample_diff * load_to_water_thickness_mm,
            colorbar_label="Equivalent water thickness (mm)",
            symmetric=True,
        )
        ax2.set_title("Induced water load")
        target_ar6.plot(
            ax=ax2,
            add_label=True,
            label="abbrev",
            line_kws={"color": "black", "linewidth": 1.5},
            text_kws=label_style,
        )

    # =========================================================================
    # 6. Extract Analytical Covariances & Means
    # =========================================================================
    print("Determining analytical covariances and means...")

    true_averages_covariance = true_averages_measure.covariance.matrix(dense=True)
    error_covariance = error_measure.covariance.matrix(dense=True)

    true_variances = np.diag(true_averages_covariance)
    true_stds_mm = np.sqrt(true_variances) * fp.length_scale * 1000

    error_variances = np.diag(error_covariance)
    error_stds_mm = np.sqrt(error_variances) * fp.length_scale * 1000

    error_means_mm = error_measure.expectation * fp.length_scale * 1000

    # =========================================================================
    # 7. Monte Carlo Sampling & Histogram Visualization
    # =========================================================================
    n_regions = len(region_names)

    if args.samples > 0:
        print(f"Drawing {args.samples} Monte Carlo samples...")

        raw_samples_list = error_measure.samples(args.samples)

        wmb_errors_samples = np.vstack(raw_samples_list)

    print("Generating histogram plots...")

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    def make_forward(std_true):
        return lambda x: (x / std_true) * 100

    def make_inverse(std_true):
        return lambda percent: (percent / 100.0) * std_true

    ncols = 2
    nrows = int(np.ceil(n_regions / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes_flat = axes.flatten()

    for i, region in enumerate(region_names):
        ax_errs = axes_flat[i]

        mean_val = error_means_mm[i]
        std_val = error_stds_mm[i]

        if args.samples > 0:
            err_mm = wmb_errors_samples[:, i] * fp.length_scale * 1000
            ax_errs.hist(
                err_mm,
                bins=50,
                alpha=0.3,
                color="red",
                density=True,
                label="MC Samples",
            )

        x_err = np.linspace(mean_val - 4 * std_val, mean_val + 4 * std_val, 200)
        y_err = gaussian_pdf(x_err, mean_val, std_val)

        ax_errs.plot(
            x_err,
            y_err,
            "r-",
            linewidth=2,
            label=rf"$\mu$={mean_val:.3f}, $\sigma$={std_val:.3f} mm",
        )

        ax_errs.set_title(f"{region}", fontsize=14)
        ax_errs.set_xlabel("Error (mm)", fontsize=12)
        ax_errs.grid(True, linestyle=":", alpha=0.6)
        ax_errs.legend()

        ax_errs.axvline(0, color="black", linestyle="--", linewidth=1.5)

        sec_ax = ax_errs.secondary_xaxis(
            "top",
            functions=(make_forward(true_stds_mm[i]), make_inverse(true_stds_mm[i])),
        )
        sec_ax.set_xlabel(
            r"Error relative to True $\sigma$ (%)", fontsize=11, color="darkred"
        )
        sec_ax.tick_params(axis="x", colors="darkred")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        "WMB Method: Distribution of Estimation Errors",
        fontsize=18,
        fontweight="bold",
    )
    plt.show()


if __name__ == "__main__":
    main()
