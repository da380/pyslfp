"""
Multi-Sensor Bayesian Inversion Comparison
==========================================

This script performs GRACE-only, Altimetry-only, and Joint Bayesian inversions
on the exact same synthetic physical scenario and noise draws. This allows for
a direct, apples-to-apples comparison of how different data constraints
reduce uncertainty in estimating ice mass loss and ocean dynamic topography.
"""

import argparse
import os
import numpy as np
import matplotlib

# Force headless backend to avoid Wayland/Qt display errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl

from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator

import joint_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Head-to-head comparison of Altimetry, GRACE, and Joint Inversions."
    )
    # --- Output Options ---
    parser.add_argument(
        "--all", action="store_true", help="Enable all plotting options."
    )
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Plot 1D analytical PDFs of the GMSL estimate.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot spatial maps of true loads and all posterior expectations.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot regional Corner Plot decomposing the signal for all inversions.",
    )

    # --- Resolution & Physics Settings ---
    parser.add_argument(
        "--lmax",
        type=int,
        default=256,
        help="Maximum spherical harmonic degree for the exact Earth model.",
    )
    parser.add_argument(
        "--surrogate-degree",
        type=int,
        default=32,
        help="Maximum spherical harmonic degree for the surrogate preconditioner.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Maximum spherical harmonic degree of the GRACE observations.",
    )
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

    # --- Preconditioner Options ---
    parser.add_argument(
        "--no-precond",
        action="store_true",
        help="Disable the preconditioners entirely.",
    )

    # --- Observation & Prior Settings ---
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
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
        default=10.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ocean-scale-km",
        type=float,
        default=100.0,
        help="Correlation length scale (in km) for the ocean dynamic prior.",
    )
    parser.add_argument(
        "--ocean-std-factor",
        type=float,
        default=2.0,
        help="Ocean dynamic standard deviation as a factor of GMSL std.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=1.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    # --- Noise Settings ---
    parser.add_argument(
        "--alt-noise-std-factor",
        type=float,
        default=1.0,
        help="Altimetry instrument noise std per point as a factor of GMSL std.",
    )
    parser.add_argument(
        "--grace-noise-scale-km",
        type=float,
        default=50.0,
        help="Correlation length scale (in km) for GRACE spatial noise.",
    )
    parser.add_argument(
        "--grace-noise-std-factor",
        type=float,
        default=0.1,
        help="GRACE spatial noise std as a factor of the ice prior std.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True

    # Setup directory to save plots
    output_dir = "output_plots_comparison"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    inf.configure_threading(n_threads=1)

    # ==========================================
    # 1. BUILD EXACT PHYSICS & MASTER DATA
    # ==========================================
    print(f"\nBuilding EXACT joint physical operators (lmax={args.lmax})...")
    exact_phys = utils.build_physics_components(
        args.lmax,
        args.load_order,
        args.load_scale_km,
        points,
        args.obs_degree,
        is_surrogate=False,
    )
    exact_meas = utils.build_measures(
        exact_phys["state"],
        exact_phys["load_space"],
        args.ice_scale_km,
        args.ice_std_mm,
        args.ocean_scale_km,
        args.ocean_std_factor,
        args.alt_noise_std_factor,
        args.grace_noise_scale_km,
        args.grace_noise_std_factor,
        args.obs_degree,
        points,
        exact_phys["scale_mm"],
        prior_shift=args.prior_shift,
    )
    scale_mm = exact_phys["scale_mm"]
    print(
        f"Implied GMSL prior standard deviation: {exact_meas['gmsl_std'] * scale_mm:.3f} mm"
    )

    print("\nDrawing MASTER synthetic model and dataset...")
    master_forward = inf.LinearForwardProblem(
        exact_phys["joint_forward"], data_error_measure=exact_meas["joint_noise"]
    )

    # Draw exactly one truth and one set of noise for the entire comparison
    true_model, joint_data = master_forward.synthetic_model_and_data(
        exact_meas["model_prior"]
    )
    alt_data, grace_data = joint_data[0], joint_data[1]

    # ==========================================
    # 2. SETUP THE THREE INVERSE PROBLEMS
    # ==========================================
    print("\nSetting up isolated Inverse Problems...")

    # 2a. Altimetry Only
    alt_fwd = inf.LinearForwardProblem(
        exact_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
    )
    alt_inv = inf.LinearBayesianInversion(alt_fwd, exact_meas["model_prior"])

    # 2b. GRACE Only
    grace_fwd = inf.LinearForwardProblem(
        exact_phys["grace_track"], data_error_measure=exact_meas["grace_noise"]
    )
    grace_inv = inf.LinearBayesianInversion(grace_fwd, exact_meas["model_prior"])

    # 2c. Joint
    joint_inv = inf.LinearBayesianInversion(master_forward, exact_meas["model_prior"])

    # ==========================================
    # 3. PRECONDITIONER SETUP
    # ==========================================
    P_alt, P_grace, P_joint = None, None, None
    if not args.no_precond:
        print("\nBuilding Woodbury Preconditioners...")
        surr_phys = utils.build_physics_components(
            args.surrogate_degree,
            args.load_order,
            args.load_scale_km,
            points,
            args.obs_degree,
            is_surrogate=True,
        )
        surr_meas = utils.build_measures(
            surr_phys["state"],
            surr_phys["load_space"],
            args.ice_scale_km,
            args.ice_std_mm,
            args.ocean_scale_km,
            args.ocean_std_factor,
            args.alt_noise_std_factor,
            args.grace_noise_scale_km,
            args.grace_noise_std_factor,
            args.obs_degree,
            points,
            surr_phys["scale_mm"],
            prior_shift=args.prior_shift,
            is_surrogate=True,
        )

        woodbury_solver = inf.LUSolver(galerkin=True, parallel=True, n_jobs=8)
        alpha = 0.1

        # Altimetry Preconditioner (Woodbury)
        surr_alt_fwd = inf.LinearForwardProblem(
            surr_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
        )
        surr_alt_inv = inf.LinearBayesianInversion(
            surr_alt_fwd, surr_meas["unmasked_prior"]
        )
        P_alt = (1 - alpha) * surr_alt_inv.woodbury_data_preconditioner(
            woodbury_solver
        ) + alpha * exact_meas["alt_noise"].inverse_covariance

        # GRACE Preconditioner (Woodbury)
        surr_grace_fwd = inf.LinearForwardProblem(
            surr_phys["grace_track"], data_error_measure=exact_meas["grace_noise"]
        )
        surr_grace_inv = inf.LinearBayesianInversion(
            surr_grace_fwd, surr_meas["unmasked_prior"]
        )
        P_grace = (1 - alpha) * surr_grace_inv.woodbury_data_preconditioner(
            woodbury_solver
        ) + alpha * exact_meas["grace_noise"].inverse_covariance

        # Joint Preconditioner (Full Joint Woodbury)
        woodbury_P_joint = joint_inv.surrogate_woodbury_data_preconditioner(
            woodbury_solver,
            alternate_forward_operator=surr_phys["joint_forward"],
            alternate_prior_measure=surr_meas["unmasked_prior"],
            alternate_data_error_measure=exact_meas["joint_noise"],
        )
        P_joint = (1 - alpha) * woodbury_P_joint + alpha * exact_meas[
            "joint_noise"
        ].inverse_covariance

    # ==========================================
    # 4. SOLVING POSTERIORS
    # ==========================================
    print("\nSolving for Posteriors...")
    callback = inf.ProgressCallback()
    tolerance = 0.01 * min(args.alt_noise_std_factor, args.grace_noise_std_factor)
    solver = inf.CGSolver(callback=callback, rtol=tolerance)

    print(" -> Solving Altimetry-only...")
    post_alt = alt_inv.model_posterior_measure(alt_data, solver, preconditioner=P_alt)
    print(" -> Solving GRACE-only...")
    post_grace = grace_inv.model_posterior_measure(
        grace_data, solver, preconditioner=P_grace
    )
    print(" -> Solving Joint...")
    post_joint = joint_inv.model_posterior_measure(
        joint_data, solver, preconditioner=P_joint
    )

    # ==========================================
    # 5. GMSL EXTRACTION & COMPARISON
    # ==========================================
    true_gmsl_op = utils.true_gmsl_operator(
        exact_phys["state"], exact_phys["load_space"], exact_phys["continuous_ssh"]
    )
    true_gmsl_val_mm = true_gmsl_op(true_model)[0] * scale_mm

    alt_avg_op = altimetry_averaging_operator(points)

    # Calculate GMSL Statistics
    std_noise_measure = exact_meas["alt_noise"].affine_mapping(operator=alt_avg_op)
    (
        np.sqrt(std_noise_measure.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )
    std_alt_gmsl = alt_avg_op(alt_data)[0] * scale_mm

    post_gmsl_alt = post_alt.affine_mapping(operator=true_gmsl_op)
    post_gmsl_grace = post_grace.affine_mapping(operator=true_gmsl_op)
    post_gmsl_joint = post_joint.affine_mapping(operator=true_gmsl_op)

    # ------------------ OPTION 1: MAPS ------------------
    if args.plot_maps:
        print("\nGenerating comparative spatial maps...")
        cmap = "seismic"
        ocean_mask = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
        ice_mask = scale_mm * exact_phys["state"].ice_projection(value=0.0)

        true_ice, true_ocean = true_model

        vmax_ice = max(
            np.max(np.abs(true_ice.data * scale_mm)),
            np.max(np.abs(post_alt.expectation[0].data * scale_mm)),
            np.max(np.abs(post_grace.expectation[0].data * scale_mm)),
            np.max(np.abs(post_joint.expectation[0].data * scale_mm)),
        )

        vmax_ocean = max(
            np.max(np.abs(true_ocean.data * scale_mm)),
            np.max(np.abs(post_alt.expectation[1].data * scale_mm)),
            np.max(np.abs(post_grace.expectation[1].data * scale_mm)),
            np.max(np.abs(post_joint.expectation[1].data * scale_mm)),
        )

        titles = ["True State", "Altimetry Only", "GRACE Only", "Joint Inversion"]
        models = [
            true_model,
            post_alt.expectation,
            post_grace.expectation,
            post_joint.expectation,
        ]

        # Grid indices mapping for a 2x2 layout: True(0,0), Alt(0,1), GRACE(1,0), Joint(1,1)
        grid_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # --- Ice Maps (2x2) ---
        fig_ice, axes_ice = plt.subplots(
            2,
            2,
            figsize=(12, 10),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        for i in range(4):
            idx = grid_indices[i]
            ice = models[i][0]
            sl.plot(
                ice * ice_mask,
                ax=axes_ice[idx],
                colorbar=True,
                colorbar_kwargs={"label": "Ice Thickness (mm)", "shrink": 0.8},
                vmin=-vmax_ice,
                vmax=vmax_ice,
                cmap=cmap,
            )
            axes_ice[idx].set_title(f"{titles[i]}\nIce Thickness Change")

        figures_to_save["comparison_posterior_maps_ice"] = fig_ice

        # --- Ocean Maps (2x2) ---
        fig_ocean, axes_ocean = plt.subplots(
            2,
            2,
            figsize=(12, 10),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        for i in range(4):
            idx = grid_indices[i]
            ocean = models[i][1]
            sl.plot(
                ocean * ocean_mask,
                ax=axes_ocean[idx],
                colorbar=True,
                colorbar_kwargs={"label": "Dynamic Ocean (mm)", "shrink": 0.8},
                vmin=-vmax_ocean,
                vmax=vmax_ocean,
                cmap=cmap,
            )
            axes_ocean[idx].set_title(f"{titles[i]}\nDynamic Ocean Component")

        figures_to_save["comparison_posterior_maps_ocean"] = fig_ocean

    # ------------------ OPTION 2: PDF ------------------
    if args.plot_pdfs:
        print("Plotting Head-to-Head GMSL PDFs...")

        class MockMeasure:
            def __init__(self, meas):
                self.mean = meas.expectation * scale_mm
                self.cov = meas.covariance.matrix(dense=True) * (scale_mm**2)

        results = {
            "Standard Altimetry Averaging": MockMeasure(std_noise_measure),
            "Bayesian Altimetry Only": MockMeasure(post_gmsl_alt),
            "Bayesian GRACE Only": MockMeasure(post_gmsl_grace),
            "Joint Bayesian (Alt + GRACE)": MockMeasure(post_gmsl_joint),
        }

        # Override the mean for the Standard Altimetry (since its mean is the data, not 0)
        results["Standard Altimetry Averaging"].mean = np.array([std_alt_gmsl])

        fig_pdf, ax_pdf = plt.subplots(figsize=(10, 6), layout="constrained")
        inf.plot_1d_distributions(
            list(results.values()),
            true_value=true_gmsl_val_mm,
            ax=ax_pdf,
            xlabel="Global Mean Sea Level Change (mm)",
            title="Comparison of GMSL Estimators Across Instruments",
            posterior_labels=list(results.keys()),
        )
        figures_to_save["comparison_gmsl_pdfs"] = fig_pdf

    # ------------------ OPTION 3: REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals...")

        # Add or remove regions here! The loops below will dynamically adapt.
        regions_to_analyze = ["Tasman Sea"]

        op_dynamic, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        combined_op = inf.ColumnLinearOperator([op_dynamic, op_ice_fp]) * scale_mm
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(true_model)

        prior_meas = exact_meas["model_prior"].affine_mapping(operator=final_op)

        # Dynamically build labels regardless of how many regions are listed
        labels = [f"{region}: Dynamic (mm)" for region in regions_to_analyze] + [
            f"{region}: Ice/SLE (mm)" for region in regions_to_analyze
        ]

        # Because plot_corner_distributions requires a single GaussianMeasure,
        # we generate three distinct figures representing the three inversions.
        posteriors = {
            "altimetry": ("Altimetry-Only", post_alt.affine_mapping(operator=final_op)),
            "grace": ("GRACE-Only", post_grace.affine_mapping(operator=final_op)),
            "joint": ("Joint Inversion", post_joint.affine_mapping(operator=final_op)),
        }

        for key, (title_prefix, meas) in posteriors.items():
            inf.plot_corner_distributions(
                meas,
                prior_measure=prior_meas,
                true_values=true_vals_mm,
                labels=labels,
                title=f"{title_prefix} Signal Separation: Dynamic vs. Ice Melt",
                fill_density=False,
            )
            figures_to_save[f"comparison_regional_corner_{key}"] = plt.gcf()

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
