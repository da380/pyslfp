"""
Optimal Bayesian Data Weighting
===============================

This script evaluates the explicit spatial weighting vector that optimally
estimates Global Mean Sea Level (GMSL) from satellite altimetry data.
By taking the adjoint of the composed [Kalman Gain -> GMSL] operator, we
condense the entire physical and statistical inversion framework into a
single, instantaneous linear functional on the data space.
"""

import argparse
import os
import numpy as np
import matplotlib

# Force headless backend
matplotlib.use("Agg")

import pygeoinf as inf
import pyslfp as sl

import altimetry_utils as utils
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimal Bayesian Weighting for GMSL")
    parser.add_argument(
        "--lmax", type=int, default=256, help="Exact Earth model degree."
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Preconditioner degree."
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Observation point spacing."
    )
    parser.add_argument("--ice-scale-factor", type=float, default=1.0)
    parser.add_argument("--ice-std-mm", type=float, default=10.0)
    parser.add_argument("--ocean-scale-factor", type=float, default=0.2)
    parser.add_argument("--ocean-std-factor", type=float, default=2.0)
    parser.add_argument("--noise-std-factor", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_dir = "output_weights"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating altimetry points (spacing {args.spacing}°)...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    n_points = len(points)
    print(f"Total observation points: {n_points}")

    inf.configure_threading(n_threads=1)

    # ==========================================
    # 1. SETUP EXACT PHYSICS & MEASURES
    # ==========================================
    print("Building exact physics and statistical measures...")
    (
        state,
        load_space,
        _,
        continuous_ssh,
        forward_op,
        scale_mm,
    ) = utils.build_physics_components(
        args.lmax, 2.0, 500.0, points, is_surrogate=False
    )

    # Force prior_shift=0.0 to ensure zero-mean for pure linear functional extraction
    model_prior, noise_meas, GMSL_prior_std = utils.build_measures(
        state,
        load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_scale_factor,
        args.ocean_std_factor,
        0.0,
        args.noise_std_factor,
        points,
        scale_mm,
        prior_shift=0.0,
    )

    forward_problem = inf.LinearForwardProblem(
        forward_op, data_error_measure=noise_meas
    )
    inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

    # ==========================================
    # 2. PRECONDITIONER SETUP (UNMASKED)
    # ==========================================
    print("Building surrogate operators and unmasked prior for preconditioner...")
    (
        surr_state,
        surr_load_space,
        _,
        _,
        surr_forward_op,
        _,
    ) = utils.build_physics_components(
        args.surrogate_degree, 2.0, 500.0, points, is_surrogate=True
    )

    # --- CRITICAL FIX: Manually construct UNMASKED priors for Woodbury ---
    ice_scale = surr_load_space.scale * args.ice_scale_factor
    ice_std = args.ice_std_mm / scale_mm
    surr_ice_prior = surr_load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    ocean_scale = surr_load_space.scale * args.ocean_scale_factor
    ocean_std = args.ocean_std_factor * GMSL_prior_std
    surr_ocean_prior = surr_load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_scale, std=ocean_std
    )

    unmasked_surr_prior = inf.GaussianMeasure.from_direct_sum(
        [surr_ice_prior, surr_ocean_prior]
    )

    noise_std = args.noise_std_factor * GMSL_prior_std
    data_space = inf.EuclideanSpace(n_points)
    surr_noise_meas = inf.GaussianMeasure.from_standard_deviation(data_space, noise_std)

    woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=4)
    woodbury_preconditioner = inverse_problem.surrogate_woodbury_data_preconditioner(
        woodbury_solver,
        alternate_forward_operator=surr_forward_op,
        alternate_prior_measure=unmasked_surr_prior,  # Fixed: Passing the invariant prior
        alternate_data_error_measure=surr_noise_meas,
    )

    alpha = 0.1
    preconditioner = (
        1 - alpha
    ) * woodbury_preconditioner + alpha * surr_noise_meas.inverse_covariance

    # ==========================================
    # 3. EXTRACT LINEAR FUNCTIONALS (WEIGHTS)
    # ==========================================
    print("\nExtracting explicit spatial weights via operator adjoints...")

    # Standard Latitude Averaging Operator
    std_avg_op = altimetry_averaging_operator(points)

    # Bayesian Posterior Expectation Operator
    solver = inf.CGSolver(rtol=0.001)
    post_exp_op = inverse_problem.posterior_expectation_operator(
        solver, preconditioner=preconditioner
    )

    # Handle whether pygeoinf returns an Affine or Linear operator
    bayes_linear_op = (
        post_exp_op.linear_part
        if isinstance(post_exp_op, inf.AffineOperator)
        else post_exp_op
    )

    # Compose: Data -> Posterior Model -> True GMSL
    true_gmsl_op = utils.true_gmsl_operator(state, load_space, continuous_ssh)
    bayes_gmsl_op = true_gmsl_op @ bayes_linear_op

    # Apply the ADJOINT to [1.0] to extract the data-space weights
    unity = np.array([1.0])
    std_weights = std_avg_op.adjoint(unity)
    bayes_weights = bayes_gmsl_op.adjoint(unity)

    """
    # ==========================================
    # 4. ERROR VARIANCE EVALUATION
    # ==========================================
    print("Evaluating theoretical error variances...")

    # Error Operators on the Joint Space [Model, Data Noise]
    data_space = noise_meas.domain

    # Error = True GMSL - Estimated GMSL( F*m + n )
    op_true = inf.RowLinearOperator(
        [true_gmsl_op, data_space.zero_operator(inf.EuclideanSpace(1))]
    )
    op_est_std = inf.RowLinearOperator([std_avg_op @ forward_op, std_avg_op])
    op_est_bayes = inf.RowLinearOperator([bayes_gmsl_op @ forward_op, bayes_gmsl_op])

    err_std_op = op_true - op_est_std
    err_bayes_op = op_true - op_est_bayes

    joint_meas = inf.GaussianMeasure.from_direct_sum([model_prior, noise_meas])

    err_std_meas = joint_meas.affine_mapping(operator=err_std_op)
    err_bayes_meas = joint_meas.affine_mapping(operator=err_bayes_op)

    std_err_mm = np.sqrt(err_std_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    bayes_err_mm = (
        np.sqrt(err_bayes_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    print("-" * 50)
    print(f"Standard Averaging GMSL Error: {std_err_mm:.3f} mm")
    print(f"Optimal Bayesian GMSL Error:   {bayes_err_mm:.3f} mm")
    print("-" * 50)
    """


    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    print("Plotting Spatial Weightings...")

    # ---------------------------------------------------------
    # Plot 1: Standard Weights (Strictly Positive)
    # ---------------------------------------------------------
    # Since these are all positive, use a sequential colormap (like 'viridis' or 'Blues')
    # and clip to the 99th percentile to avoid outliers washing it out.

    fig1, ax1 = sl.create_map_figure(figsize=(16, 10))

    ax1.set_global()
    ax1.coastlines(linewidth=0.5, alpha=0.5)
    sl.plot_points(
        points,
        data=std_weights,
        ax=ax1,
        s=4,
        edgecolors="none",
        colorbar=True,
        colorbar_kwargs={
            "label": "Spatial Weight",
        },
    )
    ax1.set_title("Standard Latitude-Weighted Averages")

    fig2, ax2 = sl.create_map_figure(figsize=(16, 10))

    bayes_max = np.max(np.abs(bayes_weights))

    ax2.set_global()
    ax2.coastlines(linewidth=0.5, alpha=0.5)
    sl.plot_points(
        points,
        data=bayes_weights,
        ax=ax2,
        s=4,
        vmin=-0.25 * bayes_max,
        vmax=0.25 * bayes_max,
        symmetric=True,
        edgecolors="none",
        colorbar=True,
        colorbar_kwargs={
            "label": "Spatial Weight",
        },
    )
    ax2.set_title("Optimal Bayesian Filter Weights")

    out_path = os.path.join(output_dir, "standard.png")
    fig1.savefig(out_path, dpi=600, bbox_inches="tight")
    out_path = os.path.join(output_dir, "bayes.png")
    fig2.savefig(out_path, dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
