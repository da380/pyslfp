"""
Bayesian Inversion of Satellite Altimetry Data for Sea Level Fingerprints.

This script performs a joint inversion of sea surface height changes to
infer ice sheet mass loss and the resulting global mean sea level (GMSL) change.
It utilizes the pygeoinf library and pyslfp to handle the massive, dense
operator networks by employing spatial and low-rank surrogate preconditioners.

Usage:
    Run `python altimetry.py --help` to see all command-line arguments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import cartopy.crs as ccrs
from pygeoinf import plot_1d_distributions, plot_corner_distributions

import pyslfp as sl


def build_physics_components(lmax, altimetry_points, order, scale_km, pointwise_std_m):
    """
    Helper function to build the FingerPrint model, Sobolev spaces, forward
    operators, and prior measures for a specific spherical harmonic truncation degree.
    """
    # 1. Initialize the core fingerprint model at the requested resolution
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # 2. Define the model space for the unknown ice thickness change
    scale = scale_km * 1000 / fp.length_scale
    model_space = inf.symmetric_space.sphere.Sobolev(
        fp.lmax, order, scale, radius=fp.mean_sea_floor_radius
    )

    # 3. Construct the network of physical operators
    op1 = sl.ice_projection_operator(fp, model_space)
    op2 = sl.ice_thickness_change_to_load_operator(fp, model_space)
    op3 = fp.as_sobolev_linear_operator(order, scale)
    op4 = sl.ocean_altimetry_operator(fp, op3.codomain, altimetry_points)

    A = op4 @ op3 @ op2 @ op1

    response_to_ssh = sl.sea_surface_height_operator(fp, op3.codomain)
    A_ssh = response_to_ssh @ op3 @ op2 @ op1

    # 4. Set the prior measure
    pointwise_std = pointwise_std_m / fp.length_scale
    initial_model_prior_measure = (
        model_space.point_value_scaled_heat_kernel_gaussian_measure(
            scale, std=pointwise_std
        )
    )
    model_prior_measure = initial_model_prior_measure.affine_mapping(operator=op1)

    return fp, model_space, A, A_ssh, model_prior_measure


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of altimetry data with surrogate preconditioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics & Grid Parameters
    parser.add_argument(
        "--lmax", type=int, default=128, help="Exact physics truncation degree"
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=64, help="Surrogate truncation degree"
    )
    parser.add_argument(
        "--spacing", type=float, default=4.0, help="Altimetry points spacing (degrees)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Preconditioner Parameters
    parser.add_argument(
        "--precond",
        type=str,
        choices=["none", "dense", "block", "banded", "spectral", "sparse"],
        default="none",
        help="Type of preconditioner to apply",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=20, help="Bandwidth for banded preconditioner"
    )
    parser.add_argument(
        "--rank", type=int, default=10, help="Rank for spectral preconditioner"
    )
    parser.add_argument(
        "--block-size", type=float, default=10.0, help="Grid size for block partitioner"
    )

    # Low-Rank Surrogate Parameters
    parser.add_argument(
        "--lr-forward",
        type=int,
        default=None,
        help="Rank for low-rank forward operator surrogate",
    )
    parser.add_argument(
        "--lr-prior",
        type=int,
        default=None,
        help="Rank for low-rank prior measure surrogate",
    )
    parser.add_argument(
        "--lr-data-error",
        type=int,
        default=None,
        help="Rank for low-rank data error measure surrogate",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Physical scaling constants
    order = 2.0
    scale_km = 250.0
    pointwise_std_m = 0.1
    altimetry_std_factor = 0.1

    # ==========================================
    # Setup Data Grid
    # ==========================================
    print("Generating altimetry points...")
    # Use a dummy FingerPrint to grab the exact altimetry coordinates
    dummy_fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    dummy_fp.set_state_from_ice_ng()
    altimetry_points = dummy_fp.ocean_altimetry_points(spacing_degrees=args.spacing)
    lats = [p[0] for p in altimetry_points]
    lons = [p[1] for p in altimetry_points]
    print(f"Using {len(altimetry_points)} altimetry points for the inversion.")

    # ==========================================
    # 1. Exact Problem Components
    # ==========================================
    print(f"Building exact physical operators (lmax={args.lmax})...")
    exact_fp, model_space, A, A_ssh, model_prior_measure = build_physics_components(
        args.lmax, altimetry_points, order, scale_km, pointwise_std_m
    )

    GMSL_weighting_function = (
        -exact_fp.ice_density
        * exact_fp.one_minus_ocean_function
        * exact_fp.ice_projection(value=0)
        / (exact_fp.water_density * exact_fp.ocean_area)
    )

    B = sl.averaging_operator(model_space, [GMSL_weighting_function])
    GMSL_prior_measure = model_prior_measure.affine_mapping(operator=B)

    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    data_space = A.codomain
    altimetry_std = altimetry_std_factor * GMSL_prior_std
    data_error_measure = inf.GaussianMeasure.from_standard_deviation(
        data_space, altimetry_std
    )

    forward_problem = inf.LinearForwardProblem(A, data_error_measure=data_error_measure)
    bayesian_inversion = inf.LinearBayesianInversion(
        forward_problem, model_prior_measure
    )

    # Generate synthetic ground truth
    print("Generating synthetic truth and data...")
    model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)

    # ==========================================
    # 2. Surrogate Problem Components
    # ==========================================
    print(f"Building spatial surrogate operators (lmax={args.surrogate_degree})...")
    _, _, surrogate_A, _, surrogate_prior = build_physics_components(
        args.surrogate_degree, altimetry_points, order, scale_km, pointwise_std_m
    )

    surrogate_inv = bayesian_inversion.surrogate_inversion(
        alternate_forward_operator=surrogate_A,
        alternate_prior_measure=surrogate_prior,
    )

    if args.lr_forward or args.lr_prior or args.lr_data_error:
        print("Applying low-rank approximations to the spatial surrogate...")
        surrogate_inv = surrogate_inv.low_rank_surrogate(
            forward_rank=args.lr_forward,
            prior_rank=args.lr_prior,
            data_error_rank=args.lr_data_error,
        )

    surrogate_normal_operator = surrogate_inv.normal_operator

    # ==========================================
    # 3. Preconditioner Routing
    # ==========================================
    preconditioner = None
    if args.precond != "none":
        print(f"Initializing {args.precond} preconditioner...")
        if args.precond == "block":
            blocks = sl.partition_points_by_grid(altimetry_points, args.block_size)
            print(f"Forming block preconditioner with {len(blocks)} blocks...")
            solver_wrapper = inf.ExactBlockPreconditioningMethod(
                blocks, incomplete=True
            )
        elif args.precond == "banded":
            solver_wrapper = inf.BandedPreconditioningMethod(
                args.bandwidth, incomplete=args.incomplete
            )
        elif args.precond == "spectral":
            solver_wrapper = inf.SpectralPreconditioningMethod(
                rank=args.rank, method="variable", max_rank=20 * args.rank
            )
        elif args.precond == "sparse":
            solver_wrapper = inf.ColumnThresholdedPreconditioningMethod(
                1e-2, max_nnz=100, incomplete=True
            )
        elif args.precond == "dense":
            solver_wrapper = inf.CholeskySolver()

        preconditioner = solver_wrapper(surrogate_normal_operator)

    # ==========================================
    # 4. Solve the Linear System
    # ==========================================
    print("Solving the linear system...")
    solver = inf.CGMatrixSolver()
    model_posterior_measure = bayesian_inversion.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )

    print(f"Number of iterations = {solver.iterations}")
    model_posterior_expectation = model_posterior_measure.expectation

    # ==========================================
    # 5. Visualization and Inference Extraction
    # ==========================================
    print("Visualizing the expected values...")

    max_abs_ice_change = (
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        (model_true).data.flatten(),
                        (model_posterior_expectation).data.flatten(),
                    ]
                )
            )
        )
        * 1000
        * exact_fp.length_scale
    )

    # Map Group 1: Ice Thickness
    fig_ice, axes_ice = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    sl.plot(
        1000 * model_true * exact_fp.length_scale,
        ax=axes_ice[0],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_ice_change,
        vmax=max_abs_ice_change,
        colorbar=True,
        colorbar_kwargs={"label": "Ice Thickness Change (mm)"},
    )
    axes_ice[0].plot(
        lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1
    )
    axes_ice[0].set_title("a) True Ice Thickness Change")

    sl.plot(
        1000 * model_posterior_expectation * exact_fp.length_scale,
        ax=axes_ice[1],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_ice_change,
        vmax=max_abs_ice_change,
        colorbar=True,
        colorbar_kwargs={"label": "Ice Thickness Change (mm)"},
    )
    axes_ice[1].plot(
        lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1
    )
    axes_ice[1].set_title("b) Posterior Expectation")

    # Map Group 2: SSH and Residuals
    ssh_posterior = A_ssh(model_posterior_expectation)
    ssh_true = A_ssh(model_true)
    ocean_mask = exact_fp.ocean_projection()

    max_abs_sl_change = (
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        (ssh_true * ocean_mask).data.flatten(),
                        (ssh_posterior * ocean_mask).data.flatten(),
                    ]
                )
            )
        )
        * 1000
        * exact_fp.length_scale
    )

    fig_ssh, axes_ssh = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    sl.plot(
        1000 * ssh_true * ocean_mask * exact_fp.length_scale,
        ax=axes_ssh[0],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_sl_change,
        vmax=max_abs_sl_change,
        colorbar=True,
        colorbar_kwargs={"label": "SSH Change (mm)"},
    )
    axes_ssh[0].set_title("a) True Sea surface height change")

    # Scale the synthetic data back to millimeters to match the background
    scaled_data = data * 1000 * exact_fp.length_scale

    # Scatter the noisy data points using the exact same colormap and limits
    axes_ssh[0].scatter(
        lons,
        lats,
        c=scaled_data,
        cmap="seismic",
        vmin=-max_abs_sl_change,
        vmax=max_abs_sl_change,
        s=40,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Calculate the predicted data and normalized residuals
    predicted_data = A(model_posterior_expectation)
    normalized_residuals = (data - predicted_data) / altimetry_std

    sl.plot(
        1000 * ssh_posterior * exact_fp.ocean_projection() * exact_fp.length_scale,
        ax=axes_ssh[1],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_sl_change,
        vmax=max_abs_sl_change,
        colorbar=True,
        colorbar_kwargs={"label": "SSH Change (mm)"},
    )
    axes_ssh[1].set_title("b) Predicted SSH Fingerprint & Normalized Residuals")

    # Scatter the normalized residuals on top
    sc4 = axes_ssh[1].scatter(
        lons,
        lats,
        c=normalized_residuals,
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
        s=40,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Add a secondary colorbar specifically for the residuals
    fig_ssh.colorbar(
        sc4,
        ax=axes_ssh[1],
        label="Normalized Residuals (σ)",
        shrink=0.7,
        pad=0.04,
        orientation="horizontal",
    )

    """


    # --- PDF for GMSL change ---
    print("Extracting PDFs for GMSL and Ice Sheet contributions...")
    GMSL_weighting_function = (
        -exact_fp.ice_density
        * exact_fp.one_minus_ocean_function
        * exact_fp.ice_projection(value=0)
        * 1000
        * exact_fp.length_scale
        / (exact_fp.water_density * exact_fp.ocean_area)
    )

    B = sl.averaging_operator(model_space, [GMSL_weighting_function])
    GMSL_true = B(model_true)
    GMSL_posterior_measure = model_posterior_measure.affine_mapping(operator=B)

    fig_gmsl, ax_gmsl = plt.subplots(figsize=(8, 6), layout="constrained")
    plot_1d_distributions(
        GMSL_posterior_measure,
        true_value=GMSL_true[0],
        ax=ax_gmsl,
        xlabel="GMSL Change (mm)",
        title="Global Mean Sea Level Change Inference from Altimetry",
    )

    # --- Corner Plot for Ice Sheet Contributions ---
    GLI_weighting_function = (
        -exact_fp.ice_density
        * exact_fp.one_minus_ocean_function
        * exact_fp.greenland_projection(value=0)
        * 1000
        * exact_fp.length_scale
        / (exact_fp.water_density * exact_fp.ocean_area)
    )
    WAI_weighting_function = (
        -exact_fp.ice_density
        * exact_fp.one_minus_ocean_function
        * exact_fp.west_antarctic_projection(value=0)
        * 1000
        * exact_fp.length_scale
        / (exact_fp.water_density * exact_fp.ocean_area)
    )
    EAI_weighting_function = (
        -exact_fp.ice_density
        * exact_fp.one_minus_ocean_function
        * exact_fp.east_antarctic_projection(value=0)
        * 1000
        * exact_fp.length_scale
        / (exact_fp.water_density * exact_fp.ocean_area)
    )

    C = sl.averaging_operator(
        model_space,
        [GLI_weighting_function, WAI_weighting_function, EAI_weighting_function],
    )
    property_true = C(model_true)
    property_posterior_measure = model_posterior_measure.affine_mapping(operator=C)

    axes_corner = plot_corner_distributions(
        property_posterior_measure,
        true_values=property_true,
        labels=["Greenland (mm)", "West Antarctica (mm)", "East Antarctica (mm)"],
        title="Joint Posterior Distributions of GMSL Contributions",
    )

    """

    plt.show()


if __name__ == "__main__":
    main()
