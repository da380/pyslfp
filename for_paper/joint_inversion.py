"""
Joint Bayesian Inversion (GRACE + Satellite Altimetry)
===============================================================

This script performs a joint Bayesian inversion of synthetic GRACE gravimetry
and satellite altimetry data to estimate ice sheet mass loss, ocean dynamic
topography, and ocean density changes (steric expansion).

It runs Head-to-Head Altimetry-only and Joint (Alt + GRACE) inversions
on the exact same physical scenario to demonstrate the added value of gravimetry.
"""

import argparse
import os
import numpy as np

import matplotlib

# Force headless backend to avoid Wayland/Qt display errors
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf

import joint_utils as utils

import pyslfp as sl
from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points

matplotlib.use("Agg")

# Set publication-quality font sizes globally
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
    }
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian Inversion (Ice, Dyn, Rho) vs Altimetry-Only."
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
        help="Plot spatial maps (True vs Joint Posterior) & Covariances.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Plot regional 3-way signal decomposition.",
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Number of samples for pointwise std estimates (Joint only).",
    )

    # --- Resolution Settings ---
    parser.add_argument(
        "--lmax", type=int, default=256, help="Exact model max SH degree."
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Preconditioner max SH degree."
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Max SH degree of GRACE observations.",
    )
    parser.add_argument(
        "--load-order", type=float, default=2.0, help="Sobolev space order."
    )
    parser.add_argument(
        "--load-scale-km", type=float, default=500.0, help="Sobolev length scale."
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Altimetry observation spacing."
    )

    # --- Prior Settings ---
    parser.add_argument(
        "--ice-scale-factor", type=float, default=1.0, help="Ice correlation scale."
    )
    parser.add_argument(
        "--ice-std-mm", type=float, default=10.0, help="Ice std dev (mm)."
    )

    parser.add_argument(
        "--ocean-dyn-scale-factor",
        type=float,
        default=0.2,
        help="Ocean dynamic correlation scale.",
    )
    parser.add_argument(
        "--ocean-dyn-std-factor",
        type=float,
        default=2.0,
        help="Ocean dynamic std as factor of GMSL std.",
    )

    parser.add_argument(
        "--ocean-rho-scale-factor",
        type=float,
        default=1.0,
        help="Ocean density correlation scale.",
    )
    parser.add_argument(
        "--ocean-rho-std-factor",
        type=float,
        default=0.5,
        help="Effective steric SL std as factor of GMSL std.",
    )

    parser.add_argument(
        "--alt-noise-scale-factor",
        type=float,
        default=0.0,
        help="Alt instrument noise correlation scale.",
    )
    parser.add_argument(
        "--alt-noise-std-factor",
        type=float,
        default=1.0,
        help="Alt instrument noise std as factor of GMSL std.",
    )

    parser.add_argument(
        "--grace-noise-scale-km",
        type=float,
        default=50.0,
        help="GRACE spatial noise correlation scale (km).",
    )
    parser.add_argument(
        "--grace-noise-std-factor",
        type=float,
        default=0.1,
        help="GRACE spatial noise std as factor of ice std.",
    )

    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True

    output_dir = "output_plots_joint_inversion"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    metrics_file = os.path.join(output_dir, "joint_metrics.txt")
    with open(metrics_file, "w") as f_metrics:
        f_metrics.write("Joint Inversion (Alt + GRACE) Metrics\n")
        f_metrics.write("======================================\n\n")

    print("Generating altimetry points...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)
    print(f"Generated {len(points)} ocean altimetry observation points.")

    inf.configure_threading(n_threads=1)
    regions_to_analyze = ["Tasman Sea"]

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(
        f"\nBuilding EXACT 3-Component joint physical operators (lmax={args.lmax})..."
    )
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
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.alt_noise_scale_factor,
        args.alt_noise_std_factor,
        args.grace_noise_scale_km,
        args.grace_noise_std_factor,
        args.obs_degree,
        points,
        exact_phys["scale_mm"],
        prior_shift=args.prior_shift,
        is_surrogate=False,
    )

    scale_mm = exact_phys["scale_mm"]
    print(
        f"Implied GMSL prior standard deviation: {exact_meas['gmsl_std'] * scale_mm:.3f} mm"
    )

    ocean_mask_mm = scale_mm * exact_phys["state"].ocean_projection(value=0.0)
    ice_mask_mm = scale_mm * exact_phys["state"].ice_projection(value=0.0)

    # Unitless masks for covariance plotting
    ocean_mask = exact_phys["state"].ocean_projection(value=0.0)
    ice_mask = exact_phys["state"].ice_projection(value=0.0)

    print("\nDrawing MASTER synthetic model and dataset...")
    master_forward = inf.LinearForwardProblem(
        exact_phys["joint_forward"], data_error_measure=exact_meas["joint_noise"]
    )
    true_model, joint_data = master_forward.synthetic_model_and_data(
        exact_meas["model_prior"]
    )
    alt_data = joint_data[0]

    # ------------------ 2. SETUP INVERSE PROBLEMS ------------------
    joint_inv = inf.LinearBayesianInversion(master_forward, exact_meas["model_prior"])

    alt_fwd = inf.LinearForwardProblem(
        exact_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
    )
    alt_inv = inf.LinearBayesianInversion(alt_fwd, exact_meas["model_prior"])

    # ------------------ 3. PRECONDITIONER SETUP ------------------
    print(
        f"\nBuilding SURROGATE operators (lmax={args.surrogate_degree}) for preconditioning..."
    )
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
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.alt_noise_scale_factor,
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

    P_joint = (1 - alpha) * joint_inv.surrogate_woodbury_data_preconditioner(
        woodbury_solver,
        alternate_forward_operator=surr_phys["joint_forward"],
        alternate_prior_measure=surr_meas["unmasked_prior"],
        alternate_data_error_measure=exact_meas["joint_noise"],
    ) + alpha * exact_meas["joint_noise"].inverse_covariance

    surr_alt_fwd = inf.LinearForwardProblem(
        surr_phys["alt_track"], data_error_measure=exact_meas["alt_noise"]
    )
    surr_alt_inv = inf.LinearBayesianInversion(
        surr_alt_fwd, surr_meas["unmasked_prior"]
    )
    P_alt = (1 - alpha) * surr_alt_inv.woodbury_data_preconditioner(
        woodbury_solver
    ) + alpha * exact_meas["alt_noise"].inverse_covariance

    # ------------------ 4. SOLVING POSTERIORS ------------------
    print("\nSolving for Posteriors...")
    callback = inf.ProgressCallback()
    tolerance = 0.01 * min(args.alt_noise_std_factor, args.grace_noise_std_factor)
    solver = inf.CGSolver(callback=callback, rtol=tolerance)

    print(" -> Solving Altimetry-only...")
    post_alt = alt_inv.model_posterior_measure(alt_data, solver, preconditioner=P_alt)

    print(" -> Solving Joint Inversion...")
    post_joint = joint_inv.model_posterior_measure(
        joint_data, solver, preconditioner=P_joint
    )

    # ------------------ 5. GMSL ------------------
    if args.plot_pdfs:
        print("\nPlotting GMSL PDFs...")
        true_gmsl_op = (
            utils.true_gmsl_operator(
                exact_phys["state"],
                exact_phys["load_space"],
                exact_phys["continuous_sl"],
            )
            * scale_mm
        )

        alt_avg_op = sl.linear_operators.altimetry_averaging_operator(points) * scale_mm

        prior_gmsl_measure = (
            exact_meas["model_prior"]
            .affine_mapping(operator=true_gmsl_op)
            .with_dense_covariance()
        )

        alt_data_measure = exact_meas["alt_noise"].affine_mapping(translation=alt_data)
        alt_gmsl_measure = alt_data_measure.affine_mapping(
            operator=alt_avg_op
        ).with_dense_covariance()

        post_gmsl_alt = post_alt.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        post_gmsl_joint = post_joint.affine_mapping(
            operator=true_gmsl_op
        ).with_dense_covariance()

        true_gmsl_val_mm = true_gmsl_op(true_model)[0]

        # Log GMSL metrics
        prior_var = prior_gmsl_measure.covariance.matrix(dense=True)[0, 0]
        alt_var = post_gmsl_alt.covariance.matrix(dense=True)[0, 0]
        joint_var = post_gmsl_joint.covariance.matrix(dense=True)[0, 0]

        alt_red = 100.0 * (1.0 - (alt_var / prior_var))
        joint_red = 100.0 * (1.0 - (joint_var / prior_var))

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(
                f"{'Target':<12} | {'Estimator':<12} | {'KL Div':<10} | {'Prior Var':<12} | {'Post Var':<12} | {'Reduction'}\n"
            )
            f_metrics.write("-" * 80 + "\n")
            f_metrics.write(
                f"{'GMSL':<12} | {'Alt-Only':<12} | {post_gmsl_alt.kl_divergence(prior_gmsl_measure):<10.4f} | {prior_var:<12.4f} | {alt_var:<12.4f} | {alt_red:>6.2f}%\n"
            )
            f_metrics.write(
                f"{'GMSL':<12} | {'Joint':<12} | {post_gmsl_joint.kl_divergence(prior_gmsl_measure):<10.4f} | {prior_var:<12.4f} | {joint_var:<12.4f} | {joint_red:>6.2f}%\n"
            )

        measures = [alt_gmsl_measure, post_gmsl_alt, post_gmsl_joint]
        labels = ["Simple averaging", "Altimetry-Only", "Joint Bayes"]

        fig_pdf, ax_pdf = plt.subplots(figsize=(10, 6), layout="constrained")
        inf.plot_1d_distributions(
            measures,
            true_value=true_gmsl_val_mm,
            ax=ax_pdf,
            title="",
            posterior_labels=labels,
        )
        figures_to_save["gmsl_pdf"] = fig_pdf

    # ------------------ 6. MAPPING & COVARIANCE ------------------
    if args.plot_maps:
        print("\nGenerating 3-component spatial maps (True vs Joint)...")
        cmap = "seismic"
        gl_kwargs = {"xlabel_style": {"size": 12}, "ylabel_style": {"size": 12}}
        cb_kwargs = {"orientation": "horizontal", "shrink": 0.8, "pad": 0.05}

        state = exact_phys["state"]
        load_space = exact_phys["load_space"]
        scale_mm = exact_phys["scale_mm"]

        mean_ocean_depth = state.model.integrate(state.sea_level) / state.ocean_area
        water_density = state.model.parameters.water_density
        steric_scale = mean_ocean_depth / water_density

        true_ice, true_dyn, true_rho = true_model
        post_ice, post_dyn, post_rho = post_joint.expectation

        plot_std = args.std_samples > 0
        ncols = 3 if plot_std else 2
        fig_width = 22 if plot_std else 14
        cmap_std = "Blues"

        # --- PRE-COMPUTE STD IF REQUESTED ---
        if plot_std:
            print(
                f"  Computing pointwise standard deviation from {args.std_samples} joint posterior samples..."
            )
            samples = post_joint.samples(args.std_samples, parallel=True, n_jobs=10)

            v_ice = load_space.zero
            v_dyn = load_space.zero
            v_rho = load_space.zero

            for s_ice, s_dyn, s_rho in samples:
                diff_ice = load_space.subtract(s_ice, post_ice)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_ice, diff_ice),
                    v_ice,
                )

                diff_dyn = load_space.subtract(s_dyn, post_dyn)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_dyn, diff_dyn),
                    v_dyn,
                )

                diff_rho = load_space.subtract(s_rho, post_rho)
                load_space.axpy(
                    1.0 / args.std_samples,
                    load_space.vector_multiply(diff_rho, diff_rho),
                    v_rho,
                )

            std_ice = load_space.vector_sqrt(v_ice)
            std_dyn = load_space.vector_sqrt(v_dyn)
            std_rho = load_space.vector_sqrt(v_rho)

            vmax_std_ice = np.max(std_ice.data * scale_mm)
            vmax_std_dyn = np.max(std_dyn.data * scale_mm)
            vmax_std_rho = np.max(std_rho.data * steric_scale * scale_mm)

        vmax_ice = max(
            np.max(np.abs(true_ice.data * scale_mm)),
            np.max(np.abs(post_ice.data * scale_mm)),
        )
        vmax_dyn = max(
            np.max(np.abs(true_dyn.data * scale_mm)),
            np.max(np.abs(post_dyn.data * scale_mm)),
        )
        vmax_rho = max(
            np.max(np.abs(true_rho.data * steric_scale * scale_mm)),
            np.max(np.abs(post_rho.data * steric_scale * scale_mm)),
        )

        fig_maps, axes = sl.subplots(
            3, ncols, figsize=(fig_width, 15), gridspec_kw={"hspace": 0.15}
        )

        # --- ROW 1: ICE ---
        sl.plot(
            true_ice * scale_mm,
            ax=axes[0, 0],
            colorbar=True,
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Ice Thickness (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        sl.plot(
            post_ice * scale_mm,
            ax=axes[0, 1],
            colorbar=True,
            vmin=-vmax_ice,
            vmax=vmax_ice,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Ice Thickness (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            sl.plot(
                std_ice * ice_mask_mm,
                ax=axes[0, 2],
                colorbar=True,
                cmap=cmap_std,
                vmin=0,
                vmax=vmax_std_ice,
                colorbar_kwargs={**cb_kwargs, "label": "Ice STD (mm)"},
                gridlines_kwargs=gl_kwargs,
            )

        # --- ROW 2: DYN ---
        sl.plot(
            true_dyn * ocean_mask_mm,
            ax=axes[1, 0],
            colorbar=True,
            vmin=-vmax_dyn,
            vmax=vmax_dyn,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Dynamic Topography (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        sl.plot(
            post_dyn * ocean_mask_mm,
            ax=axes[1, 1],
            colorbar=True,
            vmin=-vmax_dyn,
            vmax=vmax_dyn,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": "Dynamic Topography (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            sl.plot(
                std_dyn * ocean_mask_mm,
                ax=axes[1, 2],
                colorbar=True,
                cmap=cmap_std,
                vmin=0,
                vmax=vmax_std_dyn,
                colorbar_kwargs={**cb_kwargs, "label": "Dynamic STD (mm)"},
                gridlines_kwargs=gl_kwargs,
            )

        # --- ROW 3: STERIC ---
        sl.plot(
            true_rho * steric_scale * ocean_mask_mm,
            ax=axes[2, 0],
            colorbar=True,
            vmin=-vmax_rho,
            vmax=vmax_rho,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": r"Steric SL (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        sl.plot(
            post_rho * steric_scale * ocean_mask_mm,
            ax=axes[2, 1],
            colorbar=True,
            vmin=-vmax_rho,
            vmax=vmax_rho,
            cmap=cmap,
            colorbar_kwargs={**cb_kwargs, "label": r"Steric SL (mm)"},
            gridlines_kwargs=gl_kwargs,
        )
        if plot_std:
            sl.plot(
                std_rho * steric_scale * ocean_mask_mm,
                ax=axes[2, 2],
                colorbar=True,
                cmap=cmap_std,
                vmin=0,
                vmax=vmax_std_rho,
                colorbar_kwargs={**cb_kwargs, "label": "Steric STD (mm)"},
                gridlines_kwargs=gl_kwargs,
            )

        if args.plot_regions:
            for ax in axes.flatten():
                state.plot_boundaries(ax, regions_to_analyze)

        # Labels
        col_labels = ["True State", "Joint Posterior"]
        if plot_std:
            col_labels.append("Pointwise Std. Deviation")

        for j in range(ncols):
            axes[0, j].set_title(col_labels[j], fontsize=16, fontweight="bold", pad=20)

        row_titles = ["Ice thickness", "Dynamic topography", "Steric sea level"]
        for i, row_title in enumerate(row_titles):
            axes[i, 0].annotate(
                row_title,
                xy=(-0.1, 0.5),
                xycoords="axes fraction",
                fontsize=16,
                fontweight="bold",
                va="center",
                ha="center",
                rotation=90,
            )

        figures_to_save["posterior_maps_joint"] = fig_maps

        # =================================================================
        # Point-wise Covariance Maps (Joint Posterior)
        # =================================================================
        print("\nGenerating Point-wise Covariance Maps (Joint)...")

        scenarios = [
            ("Ice", 0, (-78.0, -110.0), "WAIS"),
            ("Ocean Dyn", 1, (30.0, -45.0), "North_Atlantic"),
        ]

        def plot_cov_row(ax_pr, ax_po, pr_field, po_field, pt, label):
            """Helper to plot side-by-side prior/posterior covariance."""
            vmax_pr = np.max(np.abs(pr_field.data))
            vmax_po = np.max(np.abs(po_field.data))

            # Fallback if one or both fields vanish completely
            if vmax_pr == 0 and vmax_po == 0:
                vmax_pr = vmax_po = 1.0
            elif vmax_pr == 0:
                vmax_pr = vmax_po
            elif vmax_po == 0:
                vmax_po = vmax_pr

            _, im_pr = sl.plot(
                pr_field,
                ax=ax_pr,
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-vmax_pr,
                vmax=vmax_pr,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )
            sl.plot_points(
                [pt],
                ax=ax_pr,
                color="black",
                zorder=10,
                gridlines=False,
            )

            _, im_po = sl.plot(
                po_field,
                ax=ax_po,
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-vmax_po,
                vmax=vmax_po,
                colorbar_kwargs={**cb_kwargs, "label": label},
                gridlines_kwargs=gl_kwargs,
            )
            sl.plot_points(
                [pt],
                ax=ax_po,
                color="black",
                zorder=10,
                gridlines=False,
            )

        for comp_name, comp_idx, pt, pt_name in scenarios:
            print(f"  Evaluating joint perturbation in {comp_name} at {pt}...")

            dirac_rep = load_space.dirac_representation(pt)
            test_vec = [load_space.zero, load_space.zero, load_space.zero]
            test_vec[comp_idx] = dirac_rep

            prior_cov = exact_meas["model_prior"].covariance(test_vec)
            post_cov = post_joint.covariance(test_vec)

            pr_ice, pr_dyn, pr_rho = prior_cov
            po_ice, po_dyn, po_rho = post_cov

            perturb_scale = (
                scale_mm if comp_idx in [0, 1] else (steric_scale * scale_mm)
            )

            pr_ice_plot = pr_ice * (perturb_scale * scale_mm) * ice_mask
            po_ice_plot = po_ice * (perturb_scale * scale_mm) * ice_mask

            pr_dyn_plot = pr_dyn * (perturb_scale * scale_mm) * ocean_mask
            po_dyn_plot = po_dyn * (perturb_scale * scale_mm) * ocean_mask

            pr_rho_plot = (
                pr_rho * (perturb_scale * steric_scale * scale_mm) * ocean_mask
            )
            po_rho_plot = (
                po_rho * (perturb_scale * steric_scale * scale_mm) * ocean_mask
            )

            fig_cov, axes_cov = sl.subplots(
                3, 2, figsize=(16, 16), gridspec_kw={"hspace": 0.15}
            )

            plot_cov_row(
                axes_cov[0, 0],
                axes_cov[0, 1],
                pr_ice_plot,
                po_ice_plot,
                pt,
                "Covariance (mm²)",
            )
            plot_cov_row(
                axes_cov[1, 0],
                axes_cov[1, 1],
                pr_dyn_plot,
                po_dyn_plot,
                pt,
                "Covariance (mm²)",
            )
            plot_cov_row(
                axes_cov[2, 0],
                axes_cov[2, 1],
                pr_rho_plot,
                po_rho_plot,
                pt,
                "Covariance (mm²)",
            )

            axes_cov[0, 0].set_title("Prior", fontsize=16, fontweight="bold", pad=20)
            axes_cov[0, 1].set_title(
                "Joint Posterior", fontsize=16, fontweight="bold", pad=20
            )

            for i, row_title in enumerate(row_titles):
                axes_cov[i, 0].annotate(
                    row_title,
                    xy=(-0.1, 0.5),
                    xycoords="axes fraction",
                    fontsize=16,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                )

            if args.plot_regions:
                for ax in axes_cov.flatten():
                    state.plot_boundaries(ax, regions_to_analyze)

            figures_to_save[f"covariance_{comp_name.replace(' ', '_')}_{pt_name}"] = (
                fig_cov
            )

    # ------------------ 7. REGIONAL DECOMPOSITION ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Sea Level Signals (3-way)...")
        op_dyn, op_rho, op_ice_fp = utils.regional_decomposition_operators(
            exact_phys["state"],
            exact_phys["load_space"],
            exact_phys["fp_op"],
            regions_to_analyze,
        )

        mean_ocean_depth = (
            exact_phys["state"].model.integrate(exact_phys["state"].sea_level)
            / exact_phys["state"].ocean_area
        )
        water_density = exact_phys["state"].model.parameters.water_density
        steric_scale = mean_ocean_depth / water_density

        combined_op = (
            inf.ColumnLinearOperator([op_dyn, op_rho * steric_scale, op_ice_fp])
            * scale_mm
        )
        final_op = combined_op.codomain.coordinate_projection @ combined_op

        true_vals_mm = final_op(true_model)
        prior_meas = (
            exact_meas["model_prior"]
            .affine_mapping(operator=final_op)
            .with_dense_covariance(parallel=True, n_jobs=3)
        )
        post_alt_meas = post_alt.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)
        post_joint_meas = post_joint.affine_mapping(
            operator=final_op
        ).with_dense_covariance(parallel=True, n_jobs=3)

        labels = [
            "Dynamic SL (mm)",
            "Steric SL (mm)",
            "Barystatic (mm)",
        ]

        # Log Regional metrics
        prior_cov_mat = prior_meas.covariance.matrix(dense=True)
        alt_cov_mat = post_alt_meas.covariance.matrix(dense=True)
        joint_cov_mat = post_joint_meas.covariance.matrix(dense=True)

        pr_trace = np.trace(prior_cov_mat)
        alt_trace = np.trace(alt_cov_mat)
        joint_trace = np.trace(joint_cov_mat)

        alt_trace_red = 100.0 * (1.0 - (alt_trace / pr_trace)) if pr_trace > 0 else 0.0
        joint_trace_red = (
            100.0 * (1.0 - (joint_trace / pr_trace)) if pr_trace > 0 else 0.0
        )

        pr_det = np.linalg.det(prior_cov_mat)
        alt_det = np.linalg.det(alt_cov_mat)
        joint_det = np.linalg.det(joint_cov_mat)

        alt_det_red = 100.0 * (1.0 - (alt_det / pr_det)) if pr_det > 0 else 0.0
        joint_det_red = 100.0 * (1.0 - (joint_det / pr_det)) if pr_det > 0 else 0.0

        with open(metrics_file, "a") as f_metrics:
            f_metrics.write(
                f"\n\nRegional Signal Separation ({regions_to_analyze[0]})\n"
            )
            f_metrics.write("=" * 105 + "\n")

            f_metrics.write(
                f"{'Metric':<22} | {'Prior':<12} | {'Alt-Only':<12} | {'Alt Red%':<10} | {'Joint':<12} | {'Joint Red%'}\n"
            )
            f_metrics.write("-" * 105 + "\n")
            f_metrics.write(
                f"{'Joint KL Div (nats)':<22} | {'-':<12} | {post_alt_meas.kl_divergence(prior_meas):<12.4f} | {'-':<10} | {post_joint_meas.kl_divergence(prior_meas):<12.4f} | {'-'}\n"
            )
            f_metrics.write(
                f"{'Total Var (Trace) mm²':<22} | {pr_trace:<12.4f} | {alt_trace:<12.4f} | {alt_trace_red:>9.2f}% | {joint_trace:<12.4f} | {joint_trace_red:>9.2f}%\n"
            )
            f_metrics.write(
                f"{'Generalized Var (Det)':<22} | {pr_det:<12.4e} | {alt_det:<12.4e} | {alt_det_red:>9.2f}% | {joint_det:<12.4e} | {joint_det_red:>9.2f}%\n"
            )
            f_metrics.write("-" * 105 + "\n")

            f_metrics.write(
                f"{'Component':<22} | {'Prior Var':<12} | {'Alt Var':<12} | {'Alt Red%':<10} | {'Joint Var':<12} | {'Joint Red%'}\n"
            )
            f_metrics.write("-" * 105 + "\n")

            comp_names = ["Dynamic SL", "Steric SL", "Barystatic"]
            for i, name in enumerate(comp_names):
                pr_v = prior_cov_mat[i, i]
                a_v = alt_cov_mat[i, i]
                j_v = joint_cov_mat[i, i]

                a_red = 100.0 * (1.0 - (a_v / pr_v)) if pr_v > 0 else 0.0
                j_red = 100.0 * (1.0 - (j_v / pr_v)) if pr_v > 0 else 0.0

                f_metrics.write(
                    f"{name:<22} | {pr_v:<12.4f} | {a_v:<12.4f} | {a_red:>9.2f}% | {j_v:<12.4f} | {j_red:>9.2f}%\n"
                )
            f_metrics.write("-" * 105 + "\n")

        # Plots
        inf.plot_corner_distributions(
            post_alt_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner_altimetry"] = plt.gcf()

        inf.plot_corner_distributions(
            post_joint_meas,
            prior_measure=prior_meas,
            true_values=true_vals_mm,
            labels=labels,
            title="",
            fill_density=False,
        )
        figures_to_save["regional_corner_joint"] = plt.gcf()

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
