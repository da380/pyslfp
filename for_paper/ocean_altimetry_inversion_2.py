"""
Bayesian Inversion of Satellite Altimetry Data for Dynamic Topography.

This script performs a joint inversion of sea surface height changes to
infer ocean dynamic topography. It utilizes the pygeoinf library and pyslfp
to handle the massive, dense operator networks by employing spatial and
low-rank surrogate preconditioners.

Usage:
    Run `python altimetry_dyn.py --help` to see all command-line arguments.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import cartopy.crs as ccrs

import pyslfp as sl


def build_physics_components(
    lmax,
    points,
    dyn_order,
    dyn_scale_km,
    dyn_prior_scale_km,
    dyn_prior_std_mm,
    noise_std_mm,
    /,
    *,
    precon=False,
):
    """
    Helper function to build the Earth model, Sobolev spaces, forward
    operators, and prior measures for a specific spherical harmonic truncation degree.
    """
    # 1. Initialize the core fingerprint model at the requested resolution
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # 2. Set up the model space for the ocean dynamic topography
    dyn_scale = 1000.0 * dyn_scale_km / fp.length_scale
    dyn_space = fp.sobolev_load_space(dyn_order, dyn_scale)

    # 3. Set up the operators
    ocean_projection = sl.ocean_projection_operator(
        fp, dyn_space, exclude_ice_shelves=True
    )
    dyn_to_load = sl.sea_level_change_to_load_operator(fp, dyn_space, dyn_space)
    load_to_response = fp.as_sobolev_linear_operator(dyn_order, dyn_scale, rtol=1e-9)
    response_space = load_to_response.codomain
    response_to_ssh = sl.sea_surface_height_operator(fp, response_space)
    ssh_space = response_to_ssh.codomain
    to_data = dyn_space.point_evaluation_operator(points)
    ssh_to_dyn = ssh_space.order_inclusion_operator(dyn_space.order)

    ssh_operator = (
        dyn_space.identity_operator()
        + ssh_to_dyn @ response_to_ssh @ load_to_response @ dyn_to_load
    ) @ ocean_projection

    forward_operator = to_data if precon else to_data @ ssh_operator

    data_space = forward_operator.codomain

    # 4. Set the prior model
    dyn_prior_scale = 1000.0 * dyn_prior_scale_km / fp.length_scale
    dyn_prior_std = dyn_prior_std_mm / (1000 * fp.length_scale)

    dyn_prior_measure = dyn_space.point_value_scaled_heat_kernel_gaussian_measure(
        dyn_prior_scale, std=dyn_prior_std
    )

    # 5. Set up the noise model
    noise_std = noise_std_mm / (1000 * fp.length_scale)
    data_error_measure = inf.GaussianMeasure.from_standard_deviation(
        data_space, noise_std
    )

    return (
        fp,
        dyn_space,
        ssh_operator,
        forward_operator,
        dyn_prior_measure,
        data_error_measure,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of altimetry data for dynamic topography with surrogate preconditioning.",
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
        "--spacing", type=float, default=5.0, help="Altimetry points spacing (degrees)"
    )

    # Preconditioner Parameters
    parser.add_argument(
        "--precond",
        type=str,
        choices=["none", "dense", "block", "spectral", "sparse"],
        default="none",
        help="Type of preconditioner to apply",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=20, help="Bandwidth for banded preconditioner"
    )
    parser.add_argument(
        "--rank", type=int, default=50, help="Rank for spectral preconditioner"
    )
    parser.add_argument(
        "--block-size", type=float, default=10.0, help="Grid size for block partitioner"
    )

    args = parser.parse_args()

    # Physical scaling constants
    dyn_order = 2.0
    dyn_scale_km = 500.0
    dyn_prior_scale_km = 200.0
    dyn_prior_std_mm = 5.0
    noise_std_mm = 2.0

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
    points = dummy_fp.ocean_altimetry_points(spacing_degrees=args.spacing)
    print(f"Using {len(points)} altimetry points for the inversion.")

    # ==========================================
    # 1. Exact Problem Components
    # ==========================================
    print(f"Building exact physical operators (lmax={args.lmax})...")
    (
        fp,
        dyn_space,
        ssh_operator,
        forward_operator,
        dyn_prior_measure,
        data_error_measure,
    ) = build_physics_components(
        args.lmax,
        points,
        dyn_order,
        dyn_scale_km,
        dyn_prior_scale_km,
        dyn_prior_std_mm,
        noise_std_mm,
    )

    # Set up the forward problem
    forward_problem = inf.LinearForwardProblem(
        forward_operator, data_error_measure=data_error_measure
    )

    # Set up the inverse problem
    inverse_problem = inf.LinearBayesianInversion(forward_problem, dyn_prior_measure)

    # Generate synthetic model and data
    print("Generating synthetic truth and data...")
    dyn, data = forward_problem.synthetic_model_and_data(dyn_prior_measure)

    # ==========================================
    # 2. Surrogate Problem Components
    # ==========================================
    preconditioner = None
    if args.precond != "none":
        print(f"Building spatial surrogate operators (lmax={args.surrogate_degree})...")
        _, _, _, surrogate_forward, surrogate_prior, _ = build_physics_components(
            args.surrogate_degree,
            points,
            dyn_order,
            dyn_scale_km,
            dyn_prior_scale_km,
            dyn_prior_std_mm,
            noise_std_mm,
            precon=True,
        )

        surrogate_inv = inverse_problem.surrogate_inversion(
            alternate_forward_operator=surrogate_forward,
            alternate_prior_measure=surrogate_prior,
        )

        surrogate_normal_operator = surrogate_inv.normal_operator

        # ==========================================
        # 3. Preconditioner Routing
        # ==========================================
        print(f"Initializing {args.precond} preconditioner...")
        if args.precond == "block":
            blocks = sl.partition_points_by_grid(points, args.block_size)
            print(f"Forming block preconditioner with {len(blocks)} blocks...")
            solver_wrapper = inf.ExactBlockPreconditioningMethod(
                blocks, incomplete=True
            )
        elif args.precond == "spectral":
            solver_wrapper = inf.SpectralPreconditioningMethod(
                rank=args.rank, method="fixed"
            )
        elif args.precond == "sparse":
            solver_wrapper = inf.ColumnThresholdedPreconditioningMethod(
                1e-3,
                max_nnz=200,
                incomplete=True,
            )
        elif args.precond == "dense":
            solver_wrapper = inf.CholeskySolver()

        preconditioner = solver_wrapper(surrogate_normal_operator)

    # ==========================================
    # 4. Solve the Linear System
    # ==========================================
    class Callback:
        def __init__(self):
            self.iteration = 0

        def __call__(self, xk):
            self.iteration += 1
            print(f"Working... Iteration {self.iteration}", end="\r")

    callback = Callback()

    print("Solving the linear system...")
    solver = inf.CGMatrixSolver(callback=callback)
    dyn_posterior_measure = inverse_problem.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution in {solver.iterations} iterations")

    # ==========================================
    # 5. Visualization and Inference Extraction
    # ==========================================
    print("Visualizing the expected values...")

    # Extract the expectation from the posterior measure
    dyn_posterior_expectation = dyn_posterior_measure.expectation

    # Extract lat/lon for the scatter plots
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    # --- Map Group 1: Dynamic Topography ---
    max_abs_dyn = (
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        (dyn).data.flatten(),
                        (dyn_posterior_expectation).data.flatten(),
                    ]
                )
            )
        )
        * 1000
        * fp.length_scale
    )

    fig_dyn, axes_dyn = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    ocean_mask = fp.ocean_projection(exclude_ice_shelves=True)

    sl.plot(
        1000 * dyn * fp.length_scale * ocean_mask,
        ax=axes_dyn[0],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_dyn,
        vmax=max_abs_dyn,
        colorbar=True,
        colorbar_kwargs={"label": "Dynamic Topography (mm)"},
    )
    # axes_dyn[0].plot(
    #    lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1
    # )
    axes_dyn[0].set_title("a) True Dynamic Topography")

    sl.plot(
        1000 * dyn_posterior_expectation * fp.length_scale * ocean_mask,
        ax=axes_dyn[1],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_dyn,
        vmax=max_abs_dyn,
        colorbar=True,
        colorbar_kwargs={"label": "Dynamic Topography (mm)"},
    )
    # axes_dyn[1].plot(
    #    lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree(), alpha=0.1
    # )
    axes_dyn[1].set_title("b) Posterior Expectation")

    # --- Map Group 2: SSH and Residuals ---
    ssh_true = ssh_operator(dyn)
    ssh_posterior = ssh_operator(dyn_posterior_expectation)

    max_abs_ssh = (
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        (ssh_true).data.flatten(),
                        (ssh_posterior).data.flatten(),
                    ]
                )
            )
        )
        * 1000
        * fp.length_scale
    )

    fig_ssh, axes_ssh = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    sl.plot(
        1000 * ssh_true * fp.length_scale,
        ax=axes_ssh[0],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_ssh,
        vmax=max_abs_ssh,
        colorbar=True,
        colorbar_kwargs={"label": "SSH (mm)"},
    )
    axes_ssh[0].set_title("a) True Sea Surface Height")

    # Scale the synthetic data back to millimeters for the scatter map
    scaled_data = data * 1000 * fp.length_scale

    axes_ssh[0].scatter(
        lons,
        lats,
        c=scaled_data,
        cmap="seismic",
        vmin=-max_abs_ssh,
        vmax=max_abs_ssh,
        s=20,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    sl.plot(
        1000 * ssh_posterior * fp.length_scale,
        ax=axes_ssh[1],
        coasts=True,
        cmap="seismic",
        vmin=-max_abs_ssh,
        vmax=max_abs_ssh,
        colorbar=True,
        colorbar_kwargs={"label": "SSH (mm)"},
    )
    axes_ssh[1].set_title("b) Predicted SSH")

    axes_ssh[1].scatter(
        lons,
        lats,
        c=scaled_data,
        cmap="seismic",
        vmin=-max_abs_ssh,
        vmax=max_abs_ssh,
        s=20,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    plt.show()


if __name__ == "__main__":
    main()
