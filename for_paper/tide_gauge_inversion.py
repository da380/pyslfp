import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cartopy import crs as crss

import pygeoinf as inf
import pyslfp as sl


def parse_arguments():
    """Parses command-line arguments to toggle simulation and plot options."""
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of Tide Gauge data with Woodbury Preconditioning and 3D parameterization."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all plotting options.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps of true loads, observations, and posterior expectations.",
    )
    parser.add_argument(
        "--plot-corners",
        action="store_true",
        help="Plot the corner distributions for the 3D regional estimators.",
    )
    parser.add_argument(
        "--parameterized-truth",
        action="store_true",
        help="Generate synthetic truth from the 3D parameterized space instead of a full spatial field.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Maximum spherical harmonic degree for the exact Earth model.",
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=64,
        help="Maximum spherical harmonic degree for the surrogate preconditioner model.",
    )
    parser.add_argument(
        "--noise-factor",
        type=float,
        default=0.5,
        help="Observation noise std as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--prior-scale-km",
        type=float,
        default=500.0,
        help="Correlation length scale (in km) for the ice thickness prior.",
    )
    parser.add_argument(
        "--prior-std-mm",
        type=float,
        default=10.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )

    return parser.parse_args()


def build_operators(lmax, tide_gauge_points, args, is_surrogate=False):
    """
    Builds the physical operators and priors.
    If is_surrogate=True, limits SLE iterations for a faster preconditioner.
    """
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # Set up the finger print operator
    load_order = 2.0
    load_scale_km = 500.0
    load_scale = 1000.0 * load_scale_km / fp.length_scale

    # PHYSICS-LITE: Limit SLE iterations for the surrogate
    max_iters = 1 if is_surrogate else None
    finger_print_operator = fp.as_sobolev_linear_operator(
        load_order, load_scale, rtol=1e-9, max_iterations=max_iters
    )

    model_space = finger_print_operator.domain
    ice_projection_operator = sl.ice_projection_operator(fp, model_space)
    ice_to_load_operator = sl.ice_thickness_change_to_load_operator(fp, model_space)

    tide_gauge_operator = sl.tide_gauge_operator(
        finger_print_operator.codomain, tide_gauge_points
    )

    forward_operator = (
        tide_gauge_operator
        @ finger_print_operator
        @ ice_to_load_operator
        @ ice_projection_operator
    )

    # Set up the prior
    prior_scale = 1000.0 * args.prior_scale_km / fp.length_scale
    prior_std = args.prior_std_mm / (1000.0 * fp.length_scale)

    ice_thickness_prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(
        prior_scale, std=prior_std
    )
    model_prior = ice_thickness_prior.affine_mapping(operator=ice_projection_operator)

    # Calculate GMSL Standard Deviation from the prior (used for noise scaling)
    GMSL_weighting_function = (
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        / (fp.water_density * fp.ocean_area)
    )
    B = sl.averaging_operator(model_space, [GMSL_weighting_function])
    GMSL_prior_measure = ice_thickness_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    return fp, forward_operator, model_prior, GMSL_prior_std


def main():
    args = parse_arguments()
    if args.all:
        args.plot_maps = args.plot_corners = True

    # ==========================================
    # 1. Setup Tide Gauge Points
    # ==========================================
    lats, lons = sl.read_gloss_tide_gauge_data()
    tide_gauge_points = list(zip(lats, lons))

    # ==========================================
    # 2. Build Exact Operators
    # ==========================================
    print(f"Building EXACT operators (lmax={args.lmax})...")
    exact_fp, exact_forward_operator, exact_model_prior, GMSL_prior_std = (
        build_operators(
            lmax=args.lmax,
            tide_gauge_points=tide_gauge_points,
            args=args,
            is_surrogate=False,
        )
    )

    to_mm = 1000.0 * exact_fp.length_scale
    print(f"Implied GMSL standard deviation: {GMSL_prior_std * to_mm:.3f} mm")

    # Normalize noise to a fraction of the GMSL standard deviation
    data_std = args.noise_factor * GMSL_prior_std
    data_space = exact_forward_operator.codomain
    data_error_measure = inf.GaussianMeasure.from_standard_deviation(
        data_space, data_std
    )

    forward_problem = inf.LinearForwardProblem(
        exact_forward_operator, data_error_measure=data_error_measure
    )
    inverse_problem = inf.LinearBayesianInversion(forward_problem, exact_model_prior)

    # ==========================================
    # 3. Setup Parameterization & Generate Truth
    # ==========================================
    print(
        f"\n--- Generating Synthetic Data (Parameterized Truth = {args.parameterized_truth}) ---"
    )
    model_space = exact_forward_operator.domain

    # Math masks (MUST be 0 outside ice for the operator algebra to work)
    ice_mask_math = exact_fp.ice_projection(value=0)
    grl_mask = exact_fp.greenland_projection(value=0) * ice_mask_math
    wais_mask = exact_fp.west_antarctic_projection(value=0) * ice_mask_math
    eais_mask = exact_fp.east_antarctic_projection(value=0) * ice_mask_math

    regions = ["Greenland", "W. Antarctica", "E. Antarctica"]
    masks = [grl_mask, wais_mask, eais_mask]

    # Parameterization operators
    basis_op = sl.averaging_operator(model_space, masks).adjoint
    areas = [exact_fp.integrate(m) for m in masks]
    avg_weights = [m * (1.0 / a) for m, a in zip(masks, areas)]
    eval_op = sl.averaging_operator(model_space, avg_weights)

    # Project the prior into the 3D space and densify it
    param_prior = exact_model_prior.affine_mapping(
        operator=basis_op.adjoint
    ).with_dense_covariance()

    if args.parameterized_truth:
        # Draw a 3D parameter vector and broadcast it to the spatial map
        param_truth = param_prior.sample()
        model = basis_op(param_truth)
    else:
        # Draw a fully continuous spatial field
        model = exact_model_prior.sample()

    data = forward_problem.data_measure_from_model(model).sample()

    # Extract the exact 3D regional averages of the true model to plot on the corner plot
    true_3d_mm = eval_op(model) * to_mm

    # ==========================================
    # 4. Build Preconditioner (Surrogate)
    # ==========================================
    print(
        f"\nBuilding SURROGATE operators (lmax={args.surrogate_degree}) for preconditioning..."
    )
    _, surr_forward_operator, surr_model_prior, _ = build_operators(
        lmax=args.surrogate_degree,
        tide_gauge_points=tide_gauge_points,
        args=args,
        is_surrogate=True,
    )

    surrogate_inv = inverse_problem.surrogate_inversion(
        alternate_forward_operator=surr_forward_operator,
        alternate_prior_measure=surr_model_prior,
    )
    print("Constructing preconditioner...")
    preconditioner = inf.EigenSolver()(surrogate_inv.normal_operator)

    # ==========================================
    # 5. Solve via Iterative CG (Full Spatial Model)
    # ==========================================
    print("\nSolving FULL spatial inverse problem...")
    callback = inf.ProgressCallback()
    solver = inf.CGMatrixSolver(callback=callback)

    model_posterior = inverse_problem.model_posterior_measure(
        data, solver, preconditioner=preconditioner
    )

    model_expectation = model_posterior.expectation
    data_expectation = exact_forward_operator(model_expectation)

    normalized_residuals = (data - data_expectation) / data_std

    print(f"Solution reached in {solver.iterations} iterations.")

    # ==========================================
    # 5.5 Solve PARAMETERIZED 3D Inversion
    # ==========================================
    print("\nSolving PARAMETERIZED 3D inversion (via model_space formalism)...")

    param_fp = forward_problem.parameterized_problem(basis_op, dense=True)
    param_inv = inf.LinearBayesianInversion(
        param_fp, param_prior, formalism="model_space"
    )
    param_posterior = param_inv.model_posterior_measure(data, inf.LUSolver())

    print("Extracting 3D regional averages from the full spatial posterior...")
    full_posterior_3d = model_posterior.affine_mapping(operator=eval_op)

    # --- Optimize Measures (Scale to mm AND densify exactly once) ---
    print("Densifying 3D covariances for stats and corner plots...")
    to_mm_op = param_posterior.domain.identity_operator() * to_mm

    param_posterior_mm = param_posterior.affine_mapping(
        operator=to_mm_op
    ).with_dense_covariance()
    full_posterior_3d_mm = full_posterior_3d.affine_mapping(
        operator=to_mm_op
    ).with_dense_covariance()

    # Extract stats directly from the cached dense representations
    param_mean = param_posterior_mm.expectation
    param_std = np.sqrt(np.diag(param_posterior_mm.covariance.matrix(dense=True)))

    full_mean = full_posterior_3d_mm.expectation
    full_std = np.sqrt(np.diag(full_posterior_3d_mm.covariance.matrix(dense=True)))

    print("\nResults (Average Ice Thickness Change):")
    for i, region in enumerate(regions):
        print(f"[{region}]")
        print(
            f"  Parameterized 3D: {param_mean[i]:>6.2f} mm  ± {param_std[i]:>5.2f} mm"
        )
        print(f"  Full Inversion:   {full_mean[i]:>6.2f} mm  ± {full_std[i]:>5.2f} mm")
    print("-----------------------------------------------\n")

    # ==========================================
    # 6. Spatial Plotting
    # ==========================================
    if args.plot_maps:
        # Plotting mask (NaNs outside ice so they render transparently)
        ice_mask_plot = exact_fp.ice_projection(value=np.nan)

        # --- Define the Discrete Residual Colormap ---
        bounds = [-3, -2, -1, 0, 1, 2, 3]
        discrete_cmap = plt.get_cmap("coolwarm", len(bounds) + 1)
        discrete_norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N, extend="both")

        # --- Figure 1: True Model Plot ---
        fig1, ax1 = sl.create_map_figure(figsize=(12, 7))
        _, ice_mappable = sl.plot(
            model * ice_mask_plot * to_mm, ax=ax1, cmap="RdBu", colorbar=False
        )

        scatter_mappable = ax1.scatter(
            lons,
            lats,
            c=data * to_mm,
            cmap="plasma",
            transform=crss.PlateCarree(),
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        cbar_ice1 = fig1.colorbar(
            ice_mappable, ax=ax1, location="bottom", shrink=0.6, pad=0.02
        )
        cbar_ice1.set_label("True Ice Thickness Change (mm)")
        cbar_scatter1 = fig1.colorbar(
            scatter_mappable, ax=ax1, location="bottom", shrink=0.6, pad=0.05
        )
        cbar_scatter1.set_label("Observed Sea Level change (mm)")
        ax1.set_title("True Synthetic Model & Observations")

        # --- Figure 2: Full Posterior Expectation Plot ---
        fig2, ax2 = sl.create_map_figure(figsize=(12, 7))
        _, ice_mappable_exp = sl.plot(
            model_expectation * ice_mask_plot * to_mm,
            ax=ax2,
            cmap="RdBu",
            colorbar=False,
        )

        scatter_mappable_exp = ax2.scatter(
            lons,
            lats,
            c=normalized_residuals,
            cmap=discrete_cmap,
            norm=discrete_norm,
            transform=crss.PlateCarree(),
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        cbar_ice2 = fig2.colorbar(
            ice_mappable_exp, ax=ax2, location="bottom", shrink=0.6, pad=0.02
        )
        cbar_ice2.set_label("Expected Ice Thickness Change (mm)")
        cbar_scatter2 = fig2.colorbar(
            scatter_mappable_exp,
            ax=ax2,
            location="bottom",
            shrink=0.6,
            pad=0.05,
            extend="both",
        )
        cbar_scatter2.set_label("Normalized Residuals (σ)")
        ax2.set_title("Full Posterior Expectation & Data Residuals")

        # --- Figure 3: Parameterized Posterior Expectation Plot ---
        param_model_expectation = basis_op(param_posterior.expectation)
        param_data_expectation = param_fp.forward_operator(param_posterior.expectation)
        param_normalized_residuals = (data - param_data_expectation) / data_std

        fig3, ax3 = sl.create_map_figure(figsize=(12, 7))
        _, ice_mappable_param = sl.plot(
            param_model_expectation * ice_mask_plot * to_mm,
            ax=ax3,
            cmap="RdBu",
            colorbar=False,
        )

        scatter_mappable_param = ax3.scatter(
            lons,
            lats,
            c=param_normalized_residuals,
            cmap=discrete_cmap,
            norm=discrete_norm,
            transform=crss.PlateCarree(),
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

        cbar_ice3 = fig3.colorbar(
            ice_mappable_param, ax=ax3, location="bottom", shrink=0.6, pad=0.02
        )
        cbar_ice3.set_label("Parameterized Expected Ice Thickness Change (mm)")
        cbar_scatter3 = fig3.colorbar(
            scatter_mappable_param,
            ax=ax3,
            location="bottom",
            shrink=0.6,
            pad=0.05,
            extend="both",
        )
        cbar_scatter3.set_label("Parameterized Normalized Residuals (σ)")
        ax3.set_title("3D Parameterized Expectation & Data Residuals")

    # ==========================================
    # 7. Corner Plots
    # ==========================================
    if args.plot_corners:

        def get_width_scaling(mean, std, truth, default_width=3.75):
            """Calculates the required width scaling to ensure truth is on screen."""
            z_scores = np.abs(mean - truth) / std
            max_z = np.max(z_scores)
            return max(default_width, max_z * 1.2)

        param_width = get_width_scaling(param_mean, param_std, true_3d_mm)
        full_width = get_width_scaling(full_mean, full_std, true_3d_mm)

        # --- Figure 4: Parameterized Corner Plot ---
        sl.plot_corner_distributions(
            param_posterior_mm,
            true_values=true_3d_mm,
            labels=regions,
            title="Parameterized 3D Joint Posterior",
        )

        # --- Figure 5: Full Spatial (Mapped) Corner Plot ---
        sl.plot_corner_distributions(
            full_posterior_3d_mm,
            true_values=true_3d_mm,
            labels=regions,
            title="Full Spatial Mapped Joint Posterior",
        )

    if args.plot_maps or args.plot_corners:
        plt.show()


if __name__ == "__main__":
    main()
