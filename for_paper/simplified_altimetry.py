"""
Simplified Ocean Altimetry Inversion
====================================
Inverts synthetic satellite altimetry data solely for Dynamic Ocean Topography.

This script demonstrates a powerful preconditioning trick:
The 'True' prior is strictly masked to the ocean (and thus singular), but
the 'Surrogate' prior used for the Woodbury preconditioner is the global,
invariant random field. This ensures the surrogate prior has a mathematically
exact inverse, completely avoiding the need for artificial damping.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

import pygeoinf as inf
import pyslfp as sl
from pyslfp.linear_operators import (
    ocean_altimetry_points,
    ocean_projection_operator,
    remove_ocean_average_operator,
    sea_surface_height_operator,
    sea_level_change_to_load_operator,
)
from pygeoinf.symmetric_space.sphere import Sobolev


def main():
    parser = argparse.ArgumentParser(description="Simplified Dynamic Ocean Inversion.")
    parser.add_argument("--lmax", type=int, default=128, help="Exact model degree.")
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Surrogate model degree."
    )
    parser.add_argument(
        "--scale-km", type=float, default=250.0, help="Correlation length scale."
    )
    parser.add_argument(
        "--std-mm", type=float, default=100.0, help="Ocean dynamic std in mm."
    )
    parser.add_argument(
        "--noise-std-factor", type=float, default=0.25, help="Relative noise std."
    )
    parser.add_argument(
        "--noise-scale-factor", type=float, default=0.1, help="Relative noise scale."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=2.0,
        help="Spacing of observation points in degrees.",
    )
    args = parser.parse_args()

    # =========================================================================
    # 1. EXACT PHYSICS (Masked & Singular)
    # =========================================================================
    print(f"\nBuilding Exact Physics (lmax={args.lmax})...")

    state = sl.EarthState.from_defaults(lmax=args.lmax)

    order = 2.0
    scale = args.scale_km * 1000 / state.model.parameters.length_scale

    fp = sl.linear_operators.FingerPrintOperator(
        state, load_parameters=(order, scale), response_parameters=(order, scale)
    )

    model_space = fp.domain
    response_space = fp.codomain

    prior_std = args.std_mm / (1000 * state.model.parameters.length_scale)
    invariant_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(
        scale, std=prior_std
    )

    masking_op = ocean_projection_operator(
        state, model_space
    ) @ remove_ocean_average_operator(state, model_space)
    masked_prior = invariant_prior.affine_mapping(operator=masking_op)

    points = sl.linear_operators.ocean_altimetry_points(state, spacing=args.spacing)
    point_evaluation_operator = model_space.point_evaluation_operator(points)

    ssh_op = model_space.identity_operator() + sea_surface_height_operator(
        state, response_space
    ) @ fp @ sea_level_change_to_load_operator(state, model_space, model_space)
    forward_op = point_evaluation_operator @ ssh_op

    noise_scale = args.noise_scale_factor * scale
    noise_std = args.noise_std_factor * prior_std

    if noise_scale > 0.0:
        noise_field = model_space.point_value_scaled_heat_kernel_gaussian_measure(
            noise_scale, std=noise_std
        )
        noise_meas = noise_field.affine_mapping(operator=point_evaluation_operator)
    else:
        noise_meas = inf.GaussianMeasure.from_standard_deviation(
            forward_op.codomain, noise_std
        )

    forward_problem = inf.LinearForwardProblem(
        forward_op, data_error_measure=noise_meas
    )

    print(f"Data space dimension is {forward_problem.data_space.dim}")
    print(f"Model space dimension is {forward_problem.model_space.dim}")

    inverse_problem = inf.LinearBayesianInversion(forward_problem, masked_prior)

    # Generate strictly masked synthetic data
    print("Generating synthetic ocean dynamics...")
    true_ocean_model, synthetic_data = forward_problem.synthetic_model_and_data(
        masked_prior
    )

    # =========================================================================
    # 2. SURROGATE PRECONDITIONER (Unmasked & Invertible)
    # =========================================================================
    print(f"\nBuilding Surrogate Preconditioner (lmax={args.surrogate_degree})...")

    surr_state = sl.EarthState.from_defaults(lmax=args.surrogate_degree)

    surr_fp = sl.linear_operators.FingerPrintOperator(
        surr_state,
        load_parameters=(order, scale),
        response_parameters=(order, scale),
        max_iterations=2,
    )

    surr_model_space = surr_fp.domain
    surr_response_space = surr_fp.codomain

    surr_invariant_prior = (
        surr_model_space.point_value_scaled_heat_kernel_gaussian_measure(
            scale, std=prior_std
        )
    )

    surr_point_evaluation_operator = surr_model_space.point_evaluation_operator(points)

    surr_ssh_op = surr_model_space.identity_operator() + sea_surface_height_operator(
        surr_state, surr_response_space
    ) @ surr_fp @ sea_level_change_to_load_operator(
        surr_state, surr_model_space, surr_model_space
    )
    surr_forward_op = surr_point_evaluation_operator @ surr_ssh_op

    surr_noise_meas = inf.GaussianMeasure.from_standard_deviation(
        surr_forward_op.codomain, noise_std
    )

    woodbury_solver = inf.CholeskySolver(galerkin=True)
    woodbury_preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
        woodbury_solver,
        alternate_forward_operator=surr_forward_op,
        alternate_prior_measure=surr_invariant_prior,
        alternate_data_error_measure=surr_noise_meas,
    )

    alpha = 0.1
    preconditioner = (
        1 - alpha
    ) * woodbury_preconditioner + alpha * surr_noise_meas.inverse_covariance

    # =========================================================================
    # 3. SOLVE AND PLOT
    # =========================================================================
    print("\nExecuting Conjugate Gradient Solve...")

    # callback = inverse_problem.normal_residual_callback(synthetic_data)
    callback = inf.ProgressCallback()
    solver = inf.CGSolver(callback=callback, rtol=1e-3)
    model_posterior = inverse_problem.model_posterior_measure(
        synthetic_data, solver, preconditioner=preconditioner
    )
    print(f"Solution converged in {solver.iterations} iterations.")

    print("\nPlotting spatial maps...")
    _, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    ssh_true = ssh_op(true_ocean_model)

    vmax = max(
        np.nanmax(np.abs(ssh_true.data)), np.nanmax(np.abs(true_ocean_model.data))
    )

    sl.plot(
        ssh_true * state.ocean_projection(),
        ax=axes[0, 0],
        vmin=-vmax,
        vmax=vmax,
        colorbar_kwargs={
            "label": "SSH (mm)",
            "shrink": 0.8,
            "orientation": "horizontal",
        },
    )

    axes[0, 1].set_global()
    sl.plot_points(
        points,
        data=synthetic_data,
        ax=axes[0, 1],
        vmin=-vmax,
        vmax=vmax,
        s=10,
        edgecolors="none",
        colorbar=True,
        colorbar_kwargs={
            "label": "SSH observations (mm)",
            "shrink": 0.8,
            "orientation": "horizontal",
        },
    )

    sl.plot(
        true_ocean_model * state.ocean_projection(),
        ax=axes[1, 0],
        vmin=-vmax,
        vmax=vmax,
        colorbar_kwargs={
            "label": "Dynamic SSH (mm)",
            "shrink": 0.8,
            "orientation": "horizontal",
        },
    )

    sl.plot(
        model_posterior.expectation * state.ocean_projection(),
        ax=axes[1, 1],
        vmin=-vmax,
        vmax=vmax,
        colorbar_kwargs={
            "label": "Dynamic SSH (mm): Posterior expectation",
            "shrink": 0.8,
            "orientation": "horizontal",
        },
    )

    plt.show()


if __name__ == "__main__":
    main()
