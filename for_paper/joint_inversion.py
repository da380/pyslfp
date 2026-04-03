"""
Joint Bayesian Inversion (GRACE + Satellite Altimetry)
======================================================

This script performs a joint Bayesian inversion of synthetic GRACE gravimetry
and satellite altimetry data to simultaneously estimate ice sheet mass loss
and ocean dynamic topography.

It leverages a hybrid block-diagonal preconditioner in the data space:
  - Block 1 (GRACE): Purely spectral WMB analytical approximation.
  - Block 2 (Altimetry): Sparse matrix approximation of a low-resolution surrogate.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian Inversion of GRACE and Altimetry Data."
    )
    # Resolution Parameters
    parser.add_argument("--lmax", type=int, default=64, help="Exact physics lmax.")
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=32,
        help="Surrogate lmax for altimetry preconditioner.",
    )
    parser.add_argument(
        "--obs-degree", type=int, default=32, help="GRACE max observation degree."
    )
    parser.add_argument(
        "--spacing-degrees",
        type=float,
        default=10.0,
        help="Altimetry points spacing (degrees).",
    )

    # Physics Scales
    parser.add_argument("--load-order", type=float, default=2.0)
    parser.add_argument("--load-scale-km", type=float, default=500.0)
    parser.add_argument("--ice-scale-km", type=float, default=500.0)
    parser.add_argument("--ice-std-mm", type=float, default=20.0)
    parser.add_argument("--ocean-scale-km", type=float, default=250.0)

    # Noise Factors (Relative to GMSL signal or Ice Signal)
    parser.add_argument(
        "--ocean-std-factor", type=float, default=0.5, help="Ocean dynamic std factor."
    )
    parser.add_argument(
        "--alt-noise-std-factor",
        type=float,
        default=1.0,
        help="Altimetry noise std factor.",
    )
    parser.add_argument(
        "--grace-noise-std-factor",
        type=float,
        default=0.1,
        help="GRACE noise std factor.",
    )
    parser.add_argument(
        "--grace-noise-scale-factor",
        type=float,
        default=0.25,
        help="GRACE noise length scale factor.",
    )

    parser.add_argument(
        "--sparse-threshold",
        type=float,
        default=1e-3,
        help="Tolerance for sparse altimetry preconditioner.",
    )

    return parser.parse_args()


def build_joint_physics(lmax, obs_degree, points, args):
    """
    Constructs the joint physical operators, priors, and noise measures.
    This function is used to build both the exact High-Res model and the Low-Res Surrogate.
    """
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()
    scale_mm = 1000.0 * fp.length_scale

    # ------------------ 1. SPACES & CORE MAPPINGS ------------------
    load_scale = args.load_scale_km * 1000.0 / fp.length_scale
    fp_op = fp.as_sobolev_linear_operator(args.load_order, load_scale)

    load_space = fp_op.domain
    response_space = fp_op.codomain

    P_ice = sl.ice_projection_operator(fp, load_space)
    P_ocean = sl.ocean_projection_operator(fp, load_space)

    ice_to_load = sl.ice_thickness_change_to_load_operator(fp, load_space)
    ocean_to_load = sl.sea_level_change_to_load_operator(fp, load_space, load_space)

    # Operator mapping [Ice, Ocean] -> Total Direct Load
    total_load_op = inf.RowLinearOperator(
        [ice_to_load @ P_ice, ocean_to_load @ P_ocean]
    )

    # ------------------ 2. GRACE FORWARD BLOCK ------------------
    grace_op = sl.grace_operator(response_space, obs_degree)
    A_grace = grace_op @ fp_op @ total_load_op

    # ------------------ 3. ALTIMETRY FORWARD BLOCK ------------------
    ssh_op = sl.sea_surface_height_operator(fp, response_space)
    ssh_inclusion = ssh_op.codomain.order_inclusion_operator(load_space.order)
    point_eval = load_space.point_evaluation_operator(points)

    gravity_ssh_op = ssh_inclusion @ ssh_op @ fp_op @ total_load_op

    # Extract just the dynamic ocean state to add to the gravitationally consistent SSH
    extract_ocean_op = inf.RowLinearOperator(
        [load_space.zero_operator(), load_space.identity_operator()]
    )

    total_ssh_op = gravity_ssh_op + extract_ocean_op
    A_alt = point_eval @ total_ssh_op

    # ------------------ 4. JOINT FORWARD OPERATOR ------------------
    A_joint = inf.ColumnLinearOperator([A_grace, A_alt])

    # ------------------ 5. PRIORS ------------------
    ice_scale = args.ice_scale_km * 1000.0 / fp.length_scale
    ice_std = args.ice_std_mm / scale_mm
    ice_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    # Calculate expected GMSL variance to scale the ocean and noise priors cleanly
    GMSL_weight = (
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        / (fp.water_density * fp.ocean_area)
    )
    GMSL_meas = ice_prior.affine_mapping(
        operator=sl.averaging_operator(load_space, [GMSL_weight])
    )
    gmsl_std = np.sqrt(GMSL_meas.covariance.matrix(dense=True)[0, 0])

    ocean_scale = args.ocean_scale_km * 1000.0 / fp.length_scale
    ocean_std = args.ocean_std_factor * gmsl_std
    ocean_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_scale, std=ocean_std
    )
    ocean_prior = ocean_prior.affine_mapping(
        operator=sl.remove_ocean_average_operator(fp, load_space)
    )

    # Construct the joint model prior: [Ice Prior, Ocean Prior]
    model_prior = inf.GaussianMeasure.from_direct_sum([ice_prior, ocean_prior])
    model_prior = model_prior.affine_mapping(
        operator=inf.BlockDiagonalLinearOperator([P_ice, P_ocean])
    )

    # ------------------ 6. NOISE MEASURES ------------------
    # GRACE Noise via WMB Scaling
    wmb = sl.WMBMethod.from_finger_print(fp, obs_degree)
    grace_noise_scale = args.grace_noise_scale_factor * ice_scale
    grace_noise_std = args.grace_noise_std_factor * ice_std

    grace_noise_load = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        grace_noise_scale, std=grace_noise_std
    )
    grace_noise = wmb.load_measure_to_observation_measure(grace_noise_load)

    # Altimetry White Noise
    n_points = len(points)
    alt_noise_std = args.alt_noise_std_factor * gmsl_std
    alt_noise = inf.GaussianMeasure.from_standard_deviations(
        inf.EuclideanSpace(n_points), np.full(n_points, alt_noise_std)
    )

    data_noise = inf.GaussianMeasure.from_direct_sum([grace_noise, alt_noise])

    return {
        "fp": fp,
        "load_space": load_space,
        "A_joint": A_joint,
        "A_grace": A_grace,
        "A_alt": A_alt,
        "model_prior": model_prior,
        "data_noise": data_noise,
        "grace_noise": grace_noise,
        "alt_noise": alt_noise,
        "wmb": wmb,
        "total_ssh_op": total_ssh_op,
        "scale_mm": scale_mm,
        "gmsl_std": gmsl_std,
        "ice_scale": ice_scale,
        "ice_std": ice_std,
    }


def main():
    args = parse_arguments()

    print("Generating standard altimetry observation tracks...")
    dummy_fp = sl.FingerPrint(
        lmax=args.lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    dummy_fp.set_state_from_ice_ng()
    points = dummy_fp.ocean_altimetry_points(spacing_degrees=args.spacing_degrees)

    # ==========================================
    # 1. BUILD EXACT PHYSICS
    # ==========================================
    print(f"\nBuilding EXACT joint physical operators (lmax={args.lmax})...")
    exact = build_joint_physics(args.lmax, args.obs_degree, points, args)

    print(
        f"Implied GMSL standard deviation from ice prior: {exact['gmsl_std'] * exact['scale_mm']:.3f} mm"
    )

    fwd_prob = inf.LinearForwardProblem(
        exact["A_joint"], data_error_measure=exact["data_noise"]
    )
    true_model, synthetic_data = fwd_prob.synthetic_model_and_data(exact["model_prior"])
    inv_prob = inf.LinearBayesianInversion(fwd_prob, exact["model_prior"])

    # ==========================================
    # 2. HYBRID BLOCK-DIAGONAL PRECONDITIONER
    # ==========================================
    print(
        f"\nBuilding SURROGATE model for preconditioning (lmax={args.surrogate_degree})..."
    )
    surr_obs_deg = min(args.obs_degree, args.surrogate_degree)
    surr = build_joint_physics(args.surrogate_degree, surr_obs_deg, points, args)

    print("Constructing GRACE Preconditioner Block (WMB)...")
    # WMB requires an InvariantGaussianMeasure (no spatial masks) as a proxy for the total load
    wmb_proxy_prior = surr[
        "load_space"
    ].point_value_scaled_heat_kernel_gaussian_measure(
        surr["ice_scale"], std=surr["ice_std"]
    )
    P_grace = surr["wmb"].bayesian_normal_operator_preconditioner(
        wmb_proxy_prior, surr["grace_noise"]
    )

    print("Constructing Altimetry Preconditioner Block (Sparse Surrogate)...")
    surr_alt_fwd = inf.LinearForwardProblem(
        surr["A_alt"], data_error_measure=surr["alt_noise"]
    )
    surr_alt_inv = inf.LinearBayesianInversion(surr_alt_fwd, surr["model_prior"])

    # P_alt = inf.ColumnThresholdedPreconditioningMethod(
    #    args.sparse_threshold, incomplete=True
    # )(surr_alt_inv.normal_operator)
    P_alt = inf.CholeskySolver()(surr_alt_inv.normal_operator)

    # Combine into a Block Diagonal Matrix
    print("Fusing blocks into Joint Data-Space Preconditioner...")
    joint_preconditioner = inf.BlockDiagonalLinearOperator([P_grace, P_alt])

    # ==========================================
    # 3. SOLVE POSTERIOR
    # ==========================================
    class Callback:
        def __init__(self):
            self.iteration = 0

        def __call__(self, xk):
            self.iteration += 1
            print(f"CG Iteration: {self.iteration}", end="\r")

    print("\nSolving Joint Posterior Equations...")
    solver = inf.CGMatrixSolver(callback=Callback())
    model_posterior = inv_prob.model_posterior_measure(
        synthetic_data, solver, preconditioner=joint_preconditioner
    )
    print(f"\nJoint Solution reached in {solver.iterations} iterations.")

    # ==========================================
    # 4. RESULTS & PLOTTING
    # ==========================================
    scale_mm = exact["scale_mm"]
    true_ice, true_ocean = true_model
    post_ice, post_ocean = model_posterior.expectation

    # Calculate GMSL Posteriors
    true_avg_weight = exact["fp"].ocean_function / exact["fp"].ocean_area
    true_avg_op = sl.averaging_operator(exact["load_space"], [true_avg_weight])
    gmsl_op = true_avg_op @ exact["total_ssh_op"]

    true_gmsl = gmsl_op(true_model)[0] * scale_mm
    post_gmsl_meas = model_posterior.affine_mapping(operator=gmsl_op)
    post_gmsl_mean = post_gmsl_meas.expectation[0] * scale_mm
    post_gmsl_std = (
        np.sqrt(post_gmsl_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    print(f"\n--- Results ---")
    print(f"True GMSL:      {true_gmsl:.3f} mm")
    print(f"Posterior GMSL: {post_gmsl_mean:.3f} ± {post_gmsl_std:.3f} mm")

    # Plot Maps
    ocean_mask = scale_mm * exact["fp"].ocean_projection(value=0)
    ice_mask = scale_mm * exact["fp"].ice_projection(value=0)

    vmax_ice = max(
        np.max(np.abs(true_ice.data * scale_mm)),
        np.max(np.abs(post_ice.data * scale_mm)),
    )
    vmax_ocean = max(
        np.max(np.abs(true_ocean.data * scale_mm)),
        np.max(np.abs(post_ocean.data * scale_mm)),
    )

    fig_maps, axes_maps = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        subplot_kw={"projection": ccrs.Robinson()},
        layout="constrained",
    )

    sl.plot(
        true_ice * ice_mask,
        ax=axes_maps[0, 0],
        colorbar=True,
        vmin=-vmax_ice,
        vmax=vmax_ice,
        symmetric=True,
    )
    axes_maps[0, 0].set_title("True Ice Thickness Change")

    sl.plot(
        post_ice * ice_mask,
        ax=axes_maps[0, 1],
        colorbar=True,
        vmin=-vmax_ice,
        vmax=vmax_ice,
        symmetric=True,
    )
    axes_maps[0, 1].set_title("Posterior Expected Ice Thickness")

    sl.plot(
        true_ocean * ocean_mask,
        ax=axes_maps[1, 0],
        colorbar=True,
        vmin=-vmax_ocean,
        vmax=vmax_ocean,
        symmetric=True,
    )
    axes_maps[1, 0].set_title("True Dynamic Ocean Topography")

    sl.plot(
        post_ocean * ocean_mask,
        ax=axes_maps[1, 1],
        colorbar=True,
        vmin=-vmax_ocean,
        vmax=vmax_ocean,
        symmetric=True,
    )
    axes_maps[1, 1].set_title("Posterior Expected Dynamic Ocean Topography")

    # Plot GMSL PDF
    class MockMeasure:
        def __init__(self, m, s):
            self.mean = np.array([m])
            self.cov = np.array([[s**2]])

    fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5), layout="constrained")
    inf.plot_1d_distributions(
        [MockMeasure(post_gmsl_mean, post_gmsl_std)],
        true_value=true_gmsl,
        ax=ax_pdf,
        xlabel="GMSL Change (mm)",
        title="Joint Inversion Global Mean Sea Level Estimate",
        posterior_labels=["Joint Bayes Posterior"],
    )

    plt.show()


if __name__ == "__main__":
    main()
