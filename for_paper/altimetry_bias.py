"""
Altimetry Bias Evaluation
=========================

This script quantifies the systematic bias and variance introduced when estimating
Global Mean Sea Level (GMSL) changes from a discrete set of satellite altimetry tracks.

It compares a "True" GMSL—defined as the exact spatial average of the continuous sea
surface height over the global oceans—against a "Standard" estimator that averages
point-wise observations along altimetry tracks, accounting for the solid Earth's elastic
response, sea level equation feedbacks, dynamic ocean changes, and instrument noise.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    """Parses command-line arguments to configure the altimetry bias evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate Altimetry method bias using analytical Gaussian measures."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples to draw for validating analytical error distributions.",
    )
    parser.add_argument(
        "--plot-loads",
        action="store_true",
        help="Plot an example of the SSH sample and altimetry points.",
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Maximum spherical harmonic degree for the Earth model.",
    )
    parser.add_argument(
        "--load-order",
        type=float,
        default=1.25,
        help="Sobolev space order for the load.",
    )
    parser.add_argument(
        "--load-scale-km",
        type=float,
        default=100.0,
        help="Length scale (in km) defining the load space.",
    )
    parser.add_argument(
        "--spacing-degrees",
        type=float,
        default=5.0,
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
        default=20.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ocean-scale-km",
        type=float,
        default=100.0,
        help="Correlation length scale (in km) for the ocean dynamic thickness prior.",
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
        default=1.0,
        help="Instrument noise standard deviation per point as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=0.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Initializing Earth Model and Fingerprint Operators...")
    fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    scale_mm = 1000.0 * fp.length_scale

    # ------------------ SET UP SPACES AND OPERATORS ------------------
    load_order = args.load_order
    load_scale = args.load_scale_km * 1000.0 / fp.length_scale
    finger_print_operator = fp.as_sobolev_linear_operator(load_order, load_scale)

    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    ice_projection_operator = sl.ice_projection_operator(fp, load_space)
    ocean_projection_operator = sl.ocean_projection_operator(fp, load_space)

    ice_to_load_operator = sl.ice_thickness_change_to_load_operator(fp, load_space)
    sea_level_to_load_operator = sl.sea_level_change_to_load_operator(
        fp, load_space, load_space
    )

    response_space_to_ssh_operator = sl.sea_surface_height_operator(fp, response_space)
    ssh_inclusion = response_space_to_ssh_operator.codomain.order_inclusion_operator(
        load_space.order
    )
    load_identity = load_space.identity_operator()

    points = fp.ocean_altimetry_points(spacing_degrees=args.spacing_degrees)
    point_evaluation_operator = load_space.point_evaluation_operator(points)

    print(f"Generated {len(points)} ocean altimetry observation points.")

    # ------------------ CONSTRUCT MODEL MAPPINGS ------------------
    # Map [ice_thickness, ocean_thickness] -> [total_load, ocean_thickness]
    op1 = inf.BlockLinearOperator(
        [
            [
                ice_to_load_operator @ ice_projection_operator,
                sea_level_to_load_operator @ ocean_projection_operator,
            ],
            [load_space.zero_operator(), load_identity],
        ]
    )

    # Map [total_load, ocean_thickness] -> [static_ssh, ocean_thickness]
    op2 = inf.BlockDiagonalLinearOperator(
        [
            ssh_inclusion @ response_space_to_ssh_operator @ finger_print_operator,
            load_identity,
        ]
    )

    # Map [static_ssh, ocean_thickness] -> total_ssh
    op3 = inf.RowLinearOperator([load_identity, load_identity])

    model_to_ssh_operator = op3 @ op2 @ op1
    model_space = model_to_ssh_operator.domain

    # ------------------ SET UP THE PRIORS ------------------
    ice_thickness_scale = args.ice_scale_km * 1000.0 / fp.length_scale
    ice_thickness_std = args.ice_std_mm / scale_mm

    ice_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_thickness_scale, std=ice_thickness_std
    )

    # Calculate implied GMSL standard deviation from the ice prior
    GMSL_weighting_function = (
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        / (fp.water_density * fp.ocean_area)
    )

    B = sl.averaging_operator(load_space, [GMSL_weighting_function])
    GMSL_prior_measure = ice_thickness_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    print(
        f"Implied GMSL standard deviation from ice prior: {GMSL_prior_std * scale_mm:.3f} mm"
    )

    # Set dependent standard deviations
    ocean_thickness_scale = args.ocean_scale_km * 1000.0 / fp.length_scale
    ocean_thickness_std = args.ocean_std_factor * GMSL_prior_std
    noise_std = args.noise_std_factor * GMSL_prior_std

    ocean_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_thickness_scale, std=ocean_thickness_std
    )

    # Enforce zero ocean mean for dynamic ocean thickness
    ocean_thickness_prior = ocean_thickness_prior.affine_mapping(
        operator=sl.remove_ocean_average_operator(fp, load_space)
    )

    model_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_thickness_prior, ocean_thickness_prior]
    )
    model_prior = model_prior.affine_mapping(
        operator=inf.BlockDiagonalLinearOperator(
            [ice_projection_operator, ocean_projection_operator]
        )
    )

    # Apply the prior shift if requested
    if args.prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(args.prior_shift, offset_shape)
        )

    # ------------------ NOISE MEASURE ------------------
    n_points = len(points)
    data_space = inf.EuclideanSpace(n_points)
    noise_meas = inf.GaussianMeasure.from_standard_deviations(
        data_space, np.full(n_points, noise_std)
    )

    joint_meas = inf.GaussianMeasure.from_direct_sum([model_prior, noise_meas])

    # ------------------ TRUE VS ESTIMATOR OPERATORS ------------------
    # 1. True GMSL: Spatial average of continuous SSH over the oceans
    true_avg_weight = fp.ocean_function / fp.ocean_area
    true_avg_op = sl.averaging_operator(load_space, [true_avg_weight])
    true_gmsl_op = true_avg_op @ model_to_ssh_operator

    # 2. Estimated GMSL: Point evaluation averaged via latitude weighting
    alt_avg_op = sl.altimetry_averaging_operator(points)
    est_gmsl_op = alt_avg_op @ point_evaluation_operator @ model_to_ssh_operator

    # 3. Error Operator (True - Estimated)
    err_gmsl_op = true_gmsl_op - est_gmsl_op

    # Full Operators acting on the joint space [model, noise]
    # Map to True GMSL (ignores noise)
    op_true = inf.RowLinearOperator(
        [true_gmsl_op, data_space.zero_operator(inf.EuclideanSpace(1))]
    )
    # Map to Error = True - (Est_Signal + Est_Noise)
    op_err = inf.RowLinearOperator([err_gmsl_op, -1.0 * alt_avg_op])

    # ------------------ CALCULATE ANALYTICAL DISTRIBUTIONS ------------------
    print("Calculating analytical moments...")
    true_meas = joint_meas.affine_mapping(operator=op_true)
    err_meas = joint_meas.affine_mapping(operator=op_err)
    alt_noise_meas = noise_meas.affine_mapping(operator=alt_avg_op)

    true_std = np.sqrt(true_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    err_std = np.sqrt(err_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    err_mean = err_meas.expectation[0] * scale_mm
    alt_noise_std = (
        np.sqrt(alt_noise_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    # ------------------ OPTION: PLOT EXAMPLE LOADS ------------------
    if args.plot_loads:
        print("Plotting example loads, SSH fields, and observation tracks...")
        model_sample = model_prior.sample()
        ssh_sample = model_to_ssh_operator(model_sample)

        # Extract individual components from the joint model space
        ice_thickness, ocean_thickness = model_sample

        # Create scaling masks to isolate the regions and convert to mm
        ocean_mask = scale_mm * fp.ocean_projection(value=0)
        ice_mask = scale_mm * fp.ice_projection(value=0)

        # 1. Plot Ice Thickness Change
        fig1, ax1 = sl.create_map_figure(figsize=(12, 6))
        sl.plot(
            ice_thickness * ice_mask,
            ax=ax1,
            colorbar_kwargs={"label": "Ice Thickness Change (mm)"},
            symmetric=True,
        )
        ax1.set_title("Example Ice Thickness Sample")

        # 2. Plot Ocean Dynamic Thickness
        fig2, ax2 = sl.create_map_figure(figsize=(12, 6))
        sl.plot(
            ocean_thickness * ocean_mask,
            ax=ax2,
            colorbar_kwargs={"label": "Ocean Dynamic Thickness (mm)"},
            symmetric=True,
        )
        ax2.set_title("Example Ocean Dynamic Thickness Sample")

        # 3. Plot Total Sea Surface Height with Observations
        fig3, ax3 = sl.create_map_figure(figsize=(12, 6))

        # We capture the image artist (im3) to reuse its colormap and limits
        ax3, im3 = sl.plot(
            ssh_sample * ocean_mask,
            ax=ax3,
            colorbar_kwargs={"label": "SSH Anomaly (mm)"},
            symmetric=True,
        )
        ax3.set_title("Example Sea Surface Height Sample with Observations")

        # Generate the synthetic observations: clean data at points + noise
        noise_sample = noise_meas.sample()
        clean_data = point_evaluation_operator(ssh_sample)
        observed_data = clean_data + noise_sample
        observed_data_mm = observed_data * scale_mm

        marker_size = 10

        # Extract exact limits and colormap from the background plot
        vmin, vmax = im3.get_clim()
        cmap = im3.get_cmap()

        # Scatter the altimetry points on the SSH map
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]

        ax3.scatter(
            lons,
            lats,
            c=observed_data_mm,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            marker="o",  # Circles generally show internal color better than pixels
            edgecolors=(
                "black" if marker_size > 5 else "none"
            ),  # Only outline if big enough
            linewidths=0.5 if marker_size > 5 else 0,
            transform=ccrs.PlateCarree(),
            label="Altimetry Observations",
            zorder=5,  # Ensure markers sit above the background map
        )

        # Override the legend handle to make it clearly visible
        lgnd = ax3.legend(loc="lower left")
        for handle in lgnd.legend_handles:
            handle.set_alpha(1.0)
            handle.set_sizes([20.0])
            handle.set_edgecolor("black")

    # ------------------ OPTION: MONTE CARLO VALIDATION ------------------
    err_samples = None
    if args.samples > 0:
        print(f"Drawing {args.samples} MC samples...")
        joint_samples_list = joint_meas.samples(args.samples)
        err_samples = np.zeros(args.samples)
        for idx, sample in enumerate(joint_samples_list):
            err_samples[idx] = op_err(sample)[0] * scale_mm

    # ------------------ BIAS EVALUATION PLOTTING ------------------
    print("Plotting Bias PDFs...")

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    # Transformation functions for the secondary axis
    def make_forward(std_val):
        return lambda x: x / std_val

    def make_inverse(std_val):
        return lambda x: x * std_val

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    # Pad plotting bounds based on standard deviation
    plot_min = err_mean - 4 * err_std
    plot_max = err_mean + 4 * err_std
    x_vals = np.linspace(plot_min, plot_max, 300)

    # Plot MC Histogram if generated
    if err_samples is not None:
        ax.hist(
            err_samples,
            bins=50,
            alpha=0.3,
            color="red",
            density=True,
            label="MC Samples",
        )

    # Plot True vs Estimated Distributions
    ax.plot(
        x_vals,
        gaussian_pdf(x_vals, err_mean, err_std),
        "r-",
        linewidth=2.5,
        label=rf"Total Estimator Error ($\mu$={err_mean:.3f}, $\sigma$={err_std:.3f})",
    )

    ax.plot(
        x_vals,
        gaussian_pdf(x_vals, 0, alt_noise_std),
        "b--",
        linewidth=2,
        label=rf"Theoretical Noise Floor ($\mu$=0.000, $\sigma$={alt_noise_std:.3f})",
    )

    # Format Primary Axes
    ax.set_title("GMSL Altimetry Estimator Bias", fontsize=16)
    ax.set_xlabel("Error in GMSL estimate (mm)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11)

    # Add the secondary x-axis standardized by the true signal standard deviation
    norm_label = r"Error Standardized by True Signal $\sigma$"
    sec_ax = ax.secondary_xaxis(
        "top", functions=(make_forward(true_std), make_inverse(true_std))
    )
    sec_ax.set_xlabel(norm_label, fontsize=12, color="darkgreen")
    sec_ax.tick_params(axis="x", colors="darkgreen")

    plt.show()


if __name__ == "__main__":
    main()
